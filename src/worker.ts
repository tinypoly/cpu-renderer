/// <reference lib="webworker" />
import type { WorkerRequest } from "./protocol.js";
import { createWorkerHost } from "./workerHost.js";

// The Web Worker entry: the render worker on this worker's global scope.
const handle = createWorkerHost({
  post: (message, transfer) => self.postMessage(message, transfer),
  close: () => self.close(),
});

self.onmessage = (event: MessageEvent<WorkerRequest>) => handle(event.data);
