// Node support: `CpuRenderer` on `worker_threads`. `import { createNodeWorker } from "@tinypoly/cpu-renderer/node"`.
import { Worker, type TransferListItem } from "node:worker_threads";
import type { WorkerResponse } from "./protocol.js";
import type { RenderWorker } from "./renderer.js";

/** Gives a `worker_threads` thread the shape of a Web Worker, which is what `CpuRenderer` talks to. */
export function wrapNodeWorker(worker: Worker): RenderWorker {
  const adapter: RenderWorker = {
    postMessage: (message, transfer = []) => worker.postMessage(message, transfer as unknown as TransferListItem[]),
    terminate: () => void worker.terminate(),
    onmessage: null,
    onerror: null,
    onmessageerror: null,
  };

  worker.on("message", (data: WorkerResponse) => adapter.onmessage?.({ data } as MessageEvent<WorkerResponse>));
  worker.on("error", (error: Error) => adapter.onerror?.({ message: error.message } as ErrorEvent));
  worker.on("messageerror", () => adapter.onmessageerror?.({} as MessageEvent));

  return adapter;
}

/** Starts `workerNode.js` next to this module in a worker thread. Pass it as `createWorker`. */
export function createNodeWorker(): RenderWorker {
  return wrapNodeWorker(new Worker(new URL("./workerNode.js", import.meta.url)));
}
