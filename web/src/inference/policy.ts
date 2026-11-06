/** ControlTransformer expressed through the pinned ruNNtime graph API. */
import type { TgpuRoot } from 'typegpu';
import {
  add,
  ACC_F32,
  f32ArrayToF16Bits,
  setMatmulAccum,
  uploadF16,
  type F16Buffer,
  type F32Buffer,
  cat,
  evalValues,
  gpuExecutor,
  initRunntime,
  layerNorm,
  linear,
  materialized,
  matmul,
  matrix,
  mul,
  sdpa,
  slice,
  softmax,
  transpose,
  uploadF32,
  writeF32,
  type Value,
} from '@jackknife/runntime-core';

export interface Prediction {
  steer: number;
  probabilities: number[];
  attention: number[];
  milliseconds: number;
}
interface Manifest {
  version: number;
  sha256: string;
  tensors: Record<string, { shape: number[]; offset: number; length: number }>;
}
const attentionConfig = {
  qHeads: 8,
  kvHeads: 8,
  headDim: 32,
  windowLeft: Infinity,
  windowRight: Infinity,
};

export type Precision = 'f32' | 'f16';
export async function loadPolicy(
  root: TgpuRoot,
  base = `${import.meta.env.BASE_URL}models/`,
  precision: Precision = 'f32',
) {
  if (precision === 'f16' && !root.device.features.has('shader-f16'))
    throw new Error('FP16 requires shader-f16 support.');
  const [manifestResponse, weightsResponse] = await Promise.all([
    fetch(base + 'manifest.json'),
    fetch(base + 'weights.bin'),
  ]);
  if (!manifestResponse.ok || !weightsResponse.ok)
    throw new Error('Model files missing. Run npm run export-model in web/.');
  const manifest: Manifest = await manifestResponse.json();
  if (manifest.version !== 1) throw new Error('Unsupported model export version.');
  const bytes = await weightsResponse.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const checksum = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  if (checksum !== manifest.sha256)
    throw new Error('Model checksum mismatch. Re-export the checkpoint.');
  initRunntime(root);
  const ex = gpuExecutor(root);
  setMatmulAccum(ACC_F32);
  const buffers: (F32Buffer | F16Buffer)[] = [];
  const upload = (data: Float32Array) => {
    if (precision === 'f32') return uploadF32(root, data);
    const converted = f32ArrayToF16Bits(data);
    if (converted.clamped) throw new Error('Model values exceed FP16 range.');
    return uploadF16(root, converted.data);
  };
  const write = (buffer: F32Buffer | F16Buffer, data: Float32Array) => {
    if (precision === 'f32') writeF32(root, buffer as F32Buffer, data);
    else {
      const converted = f32ArrayToF16Bits(data);
      if (converted.clamped) throw new Error('Policy inputs exceed FP16 range.');
      root.device.queue.writeBuffer(root.unwrap(buffer), 0, converted.data.buffer as ArrayBuffer);
    }
  };
  const weights = new Map<string, Value>();
  for (const [name, spec] of Object.entries(manifest.tensors)) {
    const data = new Float32Array(bytes, spec.offset, spec.length);
    const buffer = upload(data);
    buffers.push(buffer);
    weights.set(
      name,
      materialized({ dtype: precision, elems: spec.length, dims: spec.shape }, buffer),
    );
  }
  const weight = (name: string) => {
    const value = weights.get(name);
    if (!value) throw new Error(`Model tensor missing: ${name}`);
    return value;
  };
  const dense = (x: Value, name: string, residual?: Value, activation?: 'gelu') =>
    linear(x, weight(`${name}.weight`), weight(`${name}.bias`), residual, activation);
  const norm = (x: Value, name: string) =>
    layerNorm(x, weight(`${name}.weight`), 1e-5, weight(`${name}.bias`));
  const attend = (
    query: Value,
    memory: Value,
    name: string,
    returnWeights = false,
  ): [Value, Value?] => {
    const q = dense(query, `${name}.q`),
      k = dense(memory, `${name}.k`),
      v = dense(memory, `${name}.v`);
    const out = sdpa(q, k, v, attentionConfig);
    let attention: Value | undefined;
    if (returnWeights) {
      // The visualized weights are the mean of all eight cross-attention heads.
      for (let head = 0; head < 8; head++) {
        const scores = softmax(
          matmul(
            slice(q, 1, head * 32, (head + 1) * 32),
            transpose(slice(k, 1, head * 32, (head + 1) * 32)),
          ),
        );
        attention = attention ? add(attention, scores) : scores;
      }
      attention = mul(attention!, 1 / 8);
    }
    return [dense(out, `${name}.out_proj`), attention];
  };
  const tokenBuffer = upload(new Float32Array(64 * 7));
  const hitchBuffer = upload(new Float32Array(4));
  buffers.push(tokenBuffer, hitchBuffer);
  const tokens = materialized({ ...matrix(64, 7), dtype: precision }, tokenBuffer);
  const hitch = materialized({ ...matrix(1, 4), dtype: precision }, hitchBuffer);
  const forward = () => {
    let memory = add(dense(tokens, 'token_proj'), weight('pos_embed'));
    for (let i = 0; i < 6; i++) {
      const prefix = `encoder.layers.${i}`;
      memory = norm(
        add(memory, attend(memory, memory, `${prefix}.self_attn`)[0]),
        `${prefix}.norm1`,
      );
      memory = norm(
        dense(dense(memory, `${prefix}.linear1`, undefined, 'gelu'), `${prefix}.linear2`, memory),
        `${prefix}.norm2`,
      );
    }
    let query = dense(dense(hitch, 'state_proj.0', undefined, 'gelu'), 'state_proj.2');
    let attention: Value | undefined;
    for (let i = 1; i <= 2; i++) {
      const [result, weights] = attend(query, memory, `cross_attn${i}`, i === 2);
      query = norm(add(query, result), `norm${i}`);
      query = norm(
        dense(dense(query, `cross_ff${i}.0`, undefined, 'gelu'), `cross_ff${i}.2`, query),
        `norm_ff${i}`,
      );
      if (weights) attention = weights;
    }
    const logits = dense(dense(query, 'head.0', undefined, 'gelu'), 'head.2');
    return cat([logits, attention!], 1);
  };
  // Fixed shapes let ruNNtime reuse pipelines, bind groups and command records.
  const captured = ex.captureFrame(() => {
    const output = forward();
    evalValues([output], ex);
    return output;
  });
  let disposed = false;
  let busy = false;
  return {
    async predict(
      tokens: Float32Array,
      hitch: Float32Array,
    ): Promise<Prediction & { logits: number[] }> {
      if (disposed || busy) throw new Error('Policy is disposed or already running.');
      if (
        tokens.length !== 448 ||
        hitch.length !== 4 ||
        !tokens.every(Number.isFinite) ||
        !hitch.every(Number.isFinite)
      )
        throw new Error('Invalid policy inputs.');
      busy = true;
      const start = performance.now();
      try {
        write(tokenBuffer, tokens);
        write(hitchBuffer, hitch);
        const pending = ex.readbackOnSubmit(captured.targets);
        captured.replay();
        const result = await pending;
        const logits = Array.from(result.slice(0, 15));
        const max = Math.max(...logits);
        const exp = logits.map((x) => Math.exp(x - max)),
          sum = exp.reduce((a, b) => a + b, 0);
        const probabilities = exp.map((x) => x / sum);
        if (!probabilities.every(Number.isFinite)) throw new Error('Non-finite inference output.');
        return {
          logits,
          probabilities,
          attention: Array.from(result.slice(15)),
          steer: probabilities.reduce(
            (total, p, i) => total + (p * (-35 + 5 * i) * Math.PI) / 180,
            0,
          ),
          milliseconds: performance.now() - start,
        };
      } finally {
        busy = false;
      }
    },
    dispose() {
      disposed = true;
      ex.dispose();
      for (const buffer of buffers) buffer.destroy();
    },
  };
}
export type Policy = Awaited<ReturnType<typeof loadPolicy>>;
