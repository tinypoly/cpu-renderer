import { afterEach, describe, expect, it, vi } from "vitest";
import { attachCanvas } from "./canvas.js";
import { CpuRenderer, type RenderWorker } from "./renderer.js";
import type { WorkerRequest, WorkerResponse } from "./protocol.js";

/** A worker that only answers what the test replies for it. */
class FakeWorker {
  postMessage = vi.fn<(message: WorkerRequest) => void>();
  terminate = vi.fn();
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;

  reply(message: WorkerResponse) {
    this.onmessage?.({ data: message } as MessageEvent<WorkerResponse>);
  }

  get revision() {
    return this.postMessage.mock.calls.map(call => call[0]).find(message => message.type === "frame")!.revision;
  }
}

function fakeCanvas() {
  const context = { clearRect: vi.fn(), putImageData: vi.fn() };

  return { getContext: () => context, width: 300, height: 150, context } as
    unknown as HTMLCanvasElement & { context: typeof context };
}

const settle = () => new Promise<void>(resolve => setTimeout(resolve, 0));

afterEach(() => vi.unstubAllGlobals());

describe("attachCanvas", () => {
  it("sizes the canvas at start and paints each pixels event at its position", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    vi.stubGlobal("ImageData", class {
      constructor(public data: Uint8ClampedArray, public width: number, public height: number) {}
    });

    const worker = new FakeWorker();
    const renderer = new CpuRenderer({ width: 32, height: 32, createWorker: () => worker as unknown as RenderWorker });
    const canvas = fakeCanvas();
    const detach = attachCanvas(renderer, canvas);
    void renderer.render().catch(() => {});
    await settle();
    worker.reply({ type: "start", revision: worker.revision, width: 40, height: 20, order: [0], denoise: false, prepass: [] });
    expect([canvas.width, canvas.height]).toEqual([40, 20]);
    expect(canvas.context.clearRect).toHaveBeenCalledWith(0, 0, 40, 20);
    const bucket = { x: 8, y: 4, width: 4, height: 4, index: 0 };
    worker.reply({ type: "pixels", revision: worker.revision, bucket, row: 2, pixels: new Uint8ClampedArray(4 * 4) });
    const [image, x, y] = canvas.context.putImageData.mock.calls.at(-1)!;
    expect([image.width, image.height, x, y]).toEqual([4, 1, 8, 6]);
    worker.reply({ type: "bucket", revision: worker.revision, bucket, pixels: new Uint8ClampedArray(4 * 4 * 4) });
    expect(canvas.context.putImageData).toHaveBeenCalledTimes(2);

    // Detached, the canvas stops following; attached later, it starts from what the image holds.
    detach();
    worker.reply({ type: "pixels", revision: worker.revision, bucket, row: 0, pixels: new Uint8ClampedArray(4 * 4) });
    expect(canvas.context.putImageData).toHaveBeenCalledTimes(2);
    const late = fakeCanvas();
    attachCanvas(renderer, late);
    expect([late.width, late.height]).toEqual([40, 20]);
    const [whole, wx, wy] = late.context.putImageData.mock.calls[0]!;
    expect([whole.width, whole.height, wx, wy]).toEqual([40, 20, 0, 0]);
    expect(whole.data).toBe(renderer.image!.data);
    renderer.dispose();
  });

  it("refuses a canvas without a 2D context", () => {
    vi.stubGlobal("crossOriginIsolated", false);
    const createWorker = () => new FakeWorker() as unknown as RenderWorker;
    const renderer = new CpuRenderer({ width: 1, height: 1, createWorker });
    expect(() => attachCanvas(renderer, { getContext: () => null } as unknown as HTMLCanvasElement)).toThrow("Canvas 2D");
    renderer.dispose();
  });
});
