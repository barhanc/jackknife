/** Keep GPU readback callbacks and inference timing off the rendering thread. */
import tgpu, { type TgpuRoot } from 'typegpu';
import { loadPolicy, type Policy } from './policy';
import type { WorkerRequest, WorkerResponse } from './workerPolicy';
let root: TgpuRoot | undefined;
let policy: Policy | undefined;
const reply = (message: WorkerResponse) => postMessage(message);
addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === 'init') {
      root = await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } });
      root.device.lost.then((info) =>
        reply({ type: 'fatal', error: `Inference device lost: ${info.message}` }),
      );
      const precision =
        message.precision === 'auto'
          ? root.device.features.has('shader-f16')
            ? 'f16'
            : 'f32'
          : message.precision;
      policy = await loadPolicy(root, message.modelBase, precision);
      console.info(`Inference precision: ${precision.toUpperCase()} (FP32 accumulation)`);
      reply({ type: 'ready' });
    } else {
      if (!policy) throw new Error('Inference worker is not initialized.');
      const result = await policy.predict(message.tokens, message.hitch);
      reply({ type: 'prediction', id: message.id, result });
    }
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    if (message.type === 'predict') reply({ type: 'error', id: message.id, error: text });
    else reply({ type: 'fatal', error: text });
  }
});
