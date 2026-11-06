/** The game's normal policy interface, backed by a dedicated GPU inference worker. */
import type { Policy, Prediction, Precision } from './policy';
type Result = Prediction & { logits: number[] };
export type WorkerRequest =
  | { type: 'init'; modelBase: string; precision: Precision | 'auto' }
  | { type: 'predict'; id: number; tokens: Float32Array; hitch: Float32Array };
export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'prediction'; id: number; result: Result }
  | { type: 'error'; id: number; error: string }
  | { type: 'fatal'; error: string };
export async function createWorkerPolicy(): Promise<Policy> {
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  let nextId = 0,
    disposed = false;
  const pending = new Map<
    number,
    { resolve: (value: Result) => void; reject: (error: Error) => void }
  >();
  let ready!: () => void, initializationError!: (error: Error) => void;
  const initialized = new Promise<void>((resolve, reject) => {
    ready = resolve;
    initializationError = reject;
  });
  const fail = (error: Error) => {
    initializationError(error);
    for (const request of pending.values()) request.reject(error);
    pending.clear();
    disposed = true;
    worker.terminate();
  };
  worker.onerror = (event) => fail(new Error(event.message || 'Inference worker failed.'));
  worker.onmessageerror = () => fail(new Error('Unable to read inference worker response.'));
  worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
    const message = event.data;
    if (message.type === 'ready') {
      ready();
      return;
    }
    if (message.type === 'fatal') {
      fail(new Error(message.error));
      return;
    }
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.type === 'error') request.reject(new Error(message.error));
    else request.resolve(message.result);
  };
  worker.postMessage({
    type: 'init',
    precision:
      new URLSearchParams(location.search).get('precision') === 'fp32'
        ? 'f32'
        : new URLSearchParams(location.search).get('precision') === 'fp16'
          ? 'f16'
          : 'auto',
    modelBase: new URL(`${import.meta.env.BASE_URL}models/`, location.href).href,
  } satisfies WorkerRequest);
  await initialized;
  return {
    predict(tokens, hitch) {
      if (disposed) return Promise.reject(new Error('Inference worker is disposed.'));
      const id = nextId++;
      return new Promise<Result>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ type: 'predict', id, tokens, hitch } satisfies WorkerRequest);
      });
    },
    dispose() {
      if (!disposed) fail(new Error('Inference worker disposed.'));
    },
  };
}
