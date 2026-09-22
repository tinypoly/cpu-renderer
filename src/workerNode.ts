import { parentPort, type TransferListItem } from "node:worker_threads";
import type { WorkerRequest } from "./protocol.js";
import { createWorkerHost } from "./workerHost.js";

// The `worker_threads` entry: the render worker on this thread's parent port. `createNodeWorker` starts it.
if (!parentPort)
  throw new Error("CPU renderer: workerNode.js must run in a worker thread.");

const port = parentPort;

const handle = createWorkerHost({
  post: (message, transfer) => port.postMessage(message, transfer as unknown as TransferListItem[]),
  close: () => port.close(),
});

port.on("message", (request: WorkerRequest) => handle(request));
