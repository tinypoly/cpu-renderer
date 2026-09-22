import { afterEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { serializeCamera } from "./camera.js";
import { serializeScene, type SerializedScene } from "./sceneSerialization.js";
import { DEFAULT_FRAME_SETTINGS, type FrameSettings } from "./frameSettings.js";
import type { PreparedFrame, WorkerRequest, WorkerResponse } from "./protocol.js";

type Response<T extends WorkerResponse["type"]> = Extract<WorkerResponse, { type: T }>;

type FrameRequest = Extract<WorkerRequest, { type: "frame" }>;

const settings: FrameSettings = { ...DEFAULT_FRAME_SETTINGS, maxSamples: 1, tileSize: 8, ambientOcclusion: false };
const giSettings: FrameSettings = { ...settings, globalIllumination: true, giSamples: 1, giBounces: 1 };

/** A worker module on a fake global scope: requests go in through `send`, responses collect in `messages`. */
async function startWorker() {
  const messages: WorkerResponse[] = [];

  const scope = {
    postMessage: (message: WorkerResponse) => messages.push(message),
    close: vi.fn(),
    onmessage: null as ((event: MessageEvent<WorkerRequest>) => void) | null,
  };

  vi.stubGlobal("self", scope);
  vi.resetModules();
  await import("./worker.js");

  const send = (message: WorkerRequest) => scope.onmessage!({ data: message } as MessageEvent<WorkerRequest>);

  /** Waits for the next response of that type and revision that has not been taken yet. */
  const taken = new Set<WorkerResponse>();

  async function next<T extends WorkerResponse["type"]>(type: T, revision?: number, timeout = 10000): Promise<Response<T>> {
    const deadline = performance.now() + timeout;

    while (performance.now() < deadline) {
      const found = messages.find(message => !taken.has(message) && message.type === type
        && (revision === undefined || message.revision === revision));

      if (found) {
        taken.add(found);

        return found as Response<T>;
      }

      if (type !== "error" && messages.some(message => message.type === "error"))
        throw new Error(`worker error: ${(messages.find(message => message.type === "error") as Response<"error">).message}`);
      await new Promise(resolve => setTimeout(resolve, 2));
    }

    throw new Error(`no ${type} response for revision ${revision}`);
  }

  /** Lets the worker run for a while, for asserting that something does not happen. */
  const idle = (ms = 150) => new Promise(resolve => setTimeout(resolve, ms));

  return { scope, messages, send, next, idle };
}

async function litBox(): Promise<SerializedScene> {
  const scene = new THREE.Scene();
  const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: "#3080ff" }));
  const light = new THREE.DirectionalLight("#ffffff", 3);
  light.position.set(2, 4, 3);
  scene.add(box, light);

  return (await serializeScene(scene)).scene;
}

function view(x = 2) {
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 50);
  camera.position.set(x, 1.5, 3);
  camera.lookAt(0, 0, 0);

  return serializeCamera(camera);
}

async function frame(revision: number, options: Partial<FrameRequest> = {}): Promise<FrameRequest> {
  return {
    type: "frame", revision, publishPrepared: false, width: 16, height: 16, pixelRatio: 1, settings, camera: view(),
    scene: await litBox(), environment: { kind: "gradient", topColor: "#ffffff", bottomColor: "#202020" },
    ...options,
  };
}

/** Renders every bucket of a started frame and returns their pixels by bucket index. */
async function renderAll(worker: Awaited<ReturnType<typeof startWorker>>, start: Response<"start">) {
  const pixels = new Map<number, number[]>();

  for (const bucket of start.order) {
    worker.send({ type: "render", revision: start.revision, bucket, phase: "shade" });
    const result = await worker.next("bucket", start.revision);
    pixels.set(result.bucket.index, Array.from(result.pixels));
  }

  return pixels;
}

afterEach(() => vi.unstubAllGlobals());

describe("render worker", () => {
  it("prepares a frame, announces its buckets and renders the ones it is handed", async() => {
    const worker = await startWorker();
    worker.send(await frame(1));
    const start = await worker.next("start", 1);
    expect(start).toMatchObject({ width: 16, height: 16, denoise: false, prepass: [] });
    expect([...start.order].sort()).toEqual([0, 1, 2, 3]);
    // A single worker keeps its preparation instead of publishing it.
    expect(worker.messages.some(message => message.type === "prepared")).toBe(false);

    worker.send({ type: "render", revision: 1, bucket: start.order[0], phase: "shade" });
    const active = await worker.next("active", 1);
    const bucket = await worker.next("bucket", 1);
    expect(active.bucket).toEqual(bucket.bucket);
    expect(bucket.bucket).toMatchObject({ index: start.order[0], width: 8, height: 8 });
    expect(bucket.pixels).toHaveLength(8 * 8 * 4);
    expect(bucket.aov).toBeUndefined();
  });

  it("applies pixelRatio and renderScale to the render size", async() => {
    const worker = await startWorker();
    worker.send(await frame(1, { width: 10, height: 6, pixelRatio: 2, settings: { ...settings, renderScale: 0.5 } }));
    expect(await worker.next("start", 1)).toMatchObject({ width: 10, height: 6 });
  });

  it("coalesces frames that arrive together and prepares only the last", async() => {
    const worker = await startWorker();
    worker.send(await frame(1));
    worker.send(await frame(2, { scene: undefined, environment: undefined }));
    worker.send(await frame(3, { scene: undefined, environment: undefined, camera: view(-2) }));
    await worker.next("start", 3);
    await worker.idle();
    expect(worker.messages.filter(message => message.type === "start").map(message => message.revision)).toEqual([3]);
  });

  it("keeps the scene and environment of earlier frames when a frame leaves them out", async() => {
    const worker = await startWorker();
    worker.send(await frame(1));
    const first = await renderAll(worker, await worker.next("start", 1));
    worker.send(await frame(2, { scene: undefined, environment: undefined }));
    const second = await renderAll(worker, await worker.next("start", 2));
    expect(second).toEqual(first);
    worker.send(await frame(3, { scene: undefined, environment: undefined, camera: view(-2) }));
    const moved = await renderAll(worker, await worker.next("start", 3));
    expect(moved).not.toEqual(first);
  });

  it("does not start before it has a scene", async() => {
    const worker = await startWorker();
    worker.send(await frame(1, { scene: undefined }));
    await worker.idle();
    expect(worker.messages).toEqual([]);
  });

  it("ignores buckets of an older revision and of a revision it was reset out of", async() => {
    const worker = await startWorker();
    worker.send(await frame(1));
    const start = await worker.next("start", 1);
    worker.send(await frame(2, { scene: undefined, environment: undefined }));
    await worker.next("start", 2);
    // Revision 2 is ready to render: a late request of revision 1 must not run in it.
    worker.send({ type: "render", revision: 1, bucket: start.order[0], phase: "shade" });
    await worker.idle();
    worker.send({ type: "reset", revision: 3 });
    worker.send({ type: "render", revision: 3, bucket: start.order[0], phase: "shade" });
    await worker.idle();
    expect(worker.messages.filter(message => message.type === "bucket")).toEqual([]);
  });

  it("holds its work while paused and continues on resume", async() => {
    const worker = await startWorker();
    worker.send(await frame(1));
    const start = await worker.next("start", 1);
    worker.send({ type: "pause", revision: 1, paused: true });
    worker.send({ type: "render", revision: 1, bucket: start.order[0], phase: "shade" });
    await worker.idle();
    expect(worker.messages.some(message => message.type === "bucket")).toBe(false);
    worker.send({ type: "pause", revision: 1, paused: false });
    await worker.next("bucket", 1);
  });

  it("reports an unknown bucket as an error of the frame", async() => {
    const worker = await startWorker();
    worker.send(await frame(1));
    await worker.next("start", 1);
    worker.send({ type: "render", revision: 1, bucket: 99, phase: "shade" });
    expect((await worker.next("error", 1)).message).toContain("unknown bucket index");
  });

  it("reports an environment that fails to load", async() => {
    vi.stubGlobal("fetch", vi.fn(async() => new Response(null, { status: 404 })));
    const worker = await startWorker();
    worker.send(await frame(1, { environment: { kind: "hdri", url: "missing.hdr" } }));
    expect((await worker.next("error", 1)).message).toBe("HDRI: HTTP 404");
  });

  it("publishes its preparation so a reader renders the same image without preparing", async() => {
    const owner = await startWorker();
    owner.send(await frame(1, { publishPrepared: true }));
    const prepared = await owner.next("prepared", 1);
    const expected = await renderAll(owner, await owner.next("start", 1));
    // Structured clone, as postMessage copies the frame to another worker.
    const copy = structuredClone(prepared.frame) as PreparedFrame;

    const reader = await startWorker();
    reader.send({ type: "reset", revision: 1 });
    reader.send({ type: "adopt", revision: 1, frame: copy });
    const start = await reader.next("start", 1);
    expect(reader.messages.some(message => message.type === "prepared")).toBe(false);
    expect(await renderAll(reader, start)).toEqual(expected);
  });

  it("returns the denoiser accumulations with each bucket and denoises the whole image on request", async() => {
    const worker = await startWorker();

    worker.send(await frame(1, {
      publishPrepared: true, settings: { ...giSettings, giDenoise: true, giCache: false },
    }));

    const start = await worker.next("start", 1);
    expect(start.denoise).toBe(true);
    const slices = [];

    for (const index of start.order) {
      worker.send({ type: "render", revision: 1, bucket: index, phase: "shade" });
      const { bucket, aov } = await worker.next("bucket", 1);
      // Without shared memory the accumulations travel with the bucket, for the client to return them.
      expect(aov?.length).toBeGreaterThan(0);
      slices.push({ bucket, data: aov! });
    }

    worker.send({ type: "denoise", revision: 1, slices });
    await worker.next("denoised", 1);

    const bands = worker.messages.filter((message): message is Response<"pixels"> =>
      message.type === "pixels" && message.bucket.index === -1);

    // The denoised image comes back in row bands that cover it once.
    expect(bands.reduce((rows, band) => rows + band.pixels.length / 4 / 16, 0)).toBe(16);
  });

  it("fills the irradiance cache grid in bands before shading, and exports records for the other workers", async() => {
    const worker = await startWorker();

    worker.send(await frame(1, {
      publishPrepared: true, settings: { ...giSettings, giDenoise: false, giCache: true, giCacheSpacing: 4 },
    }));

    const start = await worker.next("start", 1);
    const jobs = start.prepass.reduce((a, b) => a + b, 0);
    expect(start.prepass.length).toBeGreaterThan(0);
    const cells: Float32Array[] = [];

    for (let job = 0; job < jobs; job++) {
      worker.send({ type: "render", revision: 1, bucket: job, phase: "cache" });
      const cached = await worker.next("cached", 1);
      expect(cached.job).toBe(job);
      if (cached.cells)
        cells.push(cached.cells);
    }

    // The cache phase shows its progress through its points, not through an active bucket frame.
    expect(worker.messages.some(message => message.type === "active")).toBe(false);
    expect(cells.length).toBeGreaterThan(0);
    const merged = new Float32Array(cells.reduce((total, chunk) => total + chunk.length, 0));
    cells.reduce((offset, chunk) => (merged.set(chunk, offset), offset + chunk.length), 0);
    worker.send({ type: "records", revision: 1, cells: merged });
    const shaded = await renderAll(worker, start);
    expect(shaded.size).toBe(start.order.length);
  });

  it("closes on dispose and ignores anything sent afterwards", async() => {
    const worker = await startWorker();
    worker.send(await frame(1));
    await worker.next("start", 1);
    worker.send({ type: "dispose", revision: 1 });
    expect(worker.scope.close).toHaveBeenCalledOnce();
    worker.send(await frame(2));
    await worker.idle(200);
    expect(worker.messages.filter(message => message.revision === 2)).toEqual([]);
  });
});
