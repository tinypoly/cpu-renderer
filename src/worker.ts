/// <reference lib="webworker" />
import type { CameraState, RenderEnvironment } from "./settings.js";
import type { PreparedFrame, WorkerRequest, WorkerResponse } from "./protocol.js";
import type { FrameSettings } from "./frameSettings.js";
import type { SerializedMaterial, SerializedScene } from "./sceneSerialization.js";
import { CpuEnvironment } from "./environment.js";
import { CpuScenePipeline } from "./pipeline.js";
import { CpuRasterizer, type Bucket } from "./rasterizer.js";
import { allocateFrameAov, denoiseIndirect, packBucketAov, unpackBucketAov } from "./denoise.js";
import { shareable } from "./sharedBuffers.js";
import { allocateIrradianceGrid } from "./irradianceCache.js";

// The first pool worker receives the scene and prepares everything once: simulation, captures, environment,
// geometry, buckets and BVH. The others adopt what it published and only render the buckets the client hands them.
let revision = 0;
let settings: FrameSettings | null = null;
let scene: SerializedScene | null = null;
let pipeline: CpuScenePipeline | null = null;
let environment: CpuEnvironment | null = null;
// Keep the last completed atmosphere across camera, quality and unrelated geometry changes.
let skyCache: { key: string; environment: CpuEnvironment } | null = null;

let size = {
    width: 1,
    height: 1,
    pixelRatio: 1,
  }, camera: CameraState | null = null;

let timer: ReturnType<typeof setTimeout> | undefined;
let job: Generator<void> | null = null;
let renderer: CpuRasterizer | null = null;
let bucketsByIndex = new Map<number, Bucket>();

type BucketRequest = Extract<WorkerRequest, { type: "render" }>;

let queue: BucketRequest[] = [];
let paused = false, disposed = false;
let publishPrepared = false;
let environmentAbort: AbortController | null = null;

function post(message: WorkerResponse, transfer: Transferable[] = []) {
  self.postMessage(message, transfer);
}

function cancel() {
  clearTimeout(timer);
  job = null;
  renderer = null;
  queue = [];
}

function fail(error: unknown) {
  cancel();
  post({
    type: "error",
    revision,
    message: error instanceof Error ? error.message : String(error),
  });
}

/** Slices work into short windows so messages (cancellation, pause, buckets) can get in between them. */
function pump() {
  if (paused || disposed)
    return;

  try {
    const deadline = performance.now() + 8;

    while (performance.now() < deadline) {
      if (!job) {
        if (!renderer || !queue.length)
          return;
        job = renderBucket(renderer, queue.shift()!, revision);
      }

      if (job.next().done)
        job = null;
    }

    timer = setTimeout(pump, 0);
  } catch(error) {
    fail(error);
  }
}

function start(ready: CpuRasterizer, frameRevision: number) {
  renderer = ready;
  // A bucket's index is its grid position, not its position in the ordered list: the client requests it by that.
  bucketsByIndex = new Map(ready.buckets.map(bucket => [bucket.index, bucket]));
  post({
    type: "start",
    revision: frameRevision,
    width: ready.width,
    height: ready.height,
    order: ready.buckets.map(bucket => bucket.index),
    denoise: Boolean(ready.frameAov),
    prepass: ready.irradianceGrid
      ? CpuRasterizer.prepassJobs(ready.width, ready.height, ready.settings.giCacheSpacing ?? 4)
        .map(level => level.length)
      : [],
  });
}

/** Adopting workers do not rerun the simulation: its saved state does not need to travel to the other workers. */
function withoutComputeSnapshots(source: SerializedScene): SerializedScene {
  const copies = new Map<SerializedMaterial, SerializedMaterial>();

  return { ...source, meshes: source.meshes.map(mesh => ({ ...mesh, materials: mesh.materials.map(material => {
    let copy = copies.get(material);

    if (!copy) {
      copy = material.compute?.snapshot ? { ...material, compute: { ...material.compute, snapshot: null } } : material;
      copies.set(material, copy);
    }

    return copy;
  }) })) };
}

function* prepare(base: CpuRasterizer, frameRevision: number): Generator<void> {
  let lighting = base.environment;

  if (base.scene.sky) {
    const key = JSON.stringify(base.scene.sky);

    if (skyCache?.key !== key) {
      const atmosphere = yield* CpuEnvironment.fromProcedural(base.scene.sky);
      skyCache = { key, environment: atmosphere };
    }

    lighting = skyCache.environment;
  }

  const prepared = yield* pipeline!.prepare(base.camera, base.settings, base.width, base.height, lighting);
  const ready = new CpuRasterizer(prepared, base.camera, base.settings, base.width, base.height, lighting);
  yield* ready.prepare();

  // The denoiser buffers are created here, shared when the page allows it: each worker writes its own region.
  const aov = base.settings.globalIllumination && base.settings.giDenoise
    ? allocateFrameAov(base.width, base.height) : undefined;

  if (aov)
    ready.attachAov(aov);

  // Whole-image irradiance cache grid: the cache phase writes it, shading reads it.
  const grid = ready.usesCache
    ? allocateIrradianceGrid(base.width, base.height, base.settings.giCacheSpacing ?? 4)
    : undefined;

  if (grid)
    ready.attachIrradianceGrid(grid);

  if (publishPrepared) {
    const frame: PreparedFrame = {
      scene: withoutComputeSnapshots(prepared),
      camera: base.camera,
      settings: base.settings,
      width: base.width,
      height: base.height,
      environment: lighting.snapshot(),
      geometry: ready.preparedGeometry,
      aov,
      grid,
    };

    post({ type: "prepared", revision: frameRevision, frame });
  }

  start(ready, frameRevision);
}

/** Builds the renderer on another worker's preparation: compiles the materials and reads the shared memory. */
function* adopt(frame: PreparedFrame, frameRevision: number): Generator<void> {
  const ready = new CpuRasterizer(frame.scene, frame.camera, frame.settings, frame.width, frame.height,
    CpuEnvironment.fromSnapshot(frame.environment), {}, frame.geometry);

  yield* ready.prepare();
  if (frame.aov)
    ready.attachAov(frame.aov);
  if (frame.grid)
    ready.attachIrradianceGrid(frame.grid);
  start(ready, frameRevision);
}

/** After the last bucket: denoises the whole-image indirect light and returns the recomposed rows in bands. */
function* denoise(target: CpuRasterizer, frameRevision: number,
  slices: Extract<WorkerRequest, { type: "denoise" }>["slices"]): Generator<void> {
  const aov = target.frameAov;
  if (!aov)
    return;
  for (const slice of slices)
    unpackBucketAov(aov, slice.bucket, slice.data);
  const filtered = yield* denoiseIndirect(aov, target.noiseCorrelation);
  const whole: Bucket = { x: 0, y: 0, width: aov.width, height: aov.height, index: -1 };
  yield* target.resolveDenoised(filtered, (row, pixels) => {
    const copy = new Uint8ClampedArray(pixels);
    post({ type: "pixels", revision: frameRevision, bucket: whole, row, pixels: copy }, [copy.buffer]);
  });
  post({ type: "denoised", revision: frameRevision });
}

// Minimum interval between sends of the finished pixels of an in-progress bucket.
const PROGRESS_INTERVAL_MS = 30;

function* renderBucket(target: CpuRasterizer, request: BucketRequest, frameRevision: number): Generator<void> {
  let bucket: Bucket | undefined;
  if (request.phase === "cache") {
    // Whole-image band covered by the job: it is what the preview paints and the activity indicator shows.
    const job = target.prepassJob(request.bucket);
    bucket = { x: 0, y: job.y, width: target.width, height: job.height, index: -1 };
  } else
    bucket = bucketsByIndex.get(request.bucket);
  if (!bucket)
    throw new Error("CPU renderer: unknown bucket index.");
  // In the cache phase the points themselves show progress: no active-bucket frame over the image.
  if (request.phase !== "cache")
    post({
      type: "active",
      revision: frameRevision,
      bucket,
    });
  // Row band changed since the last send; with depth of field each pass starts again from the top.
  let lastPost = performance.now(), firstRow = -1, lastRow = -1, latest: Uint8ClampedArray | null = null;

  const flush = () => {
    if (firstRow < 0 || !latest)
      return;
    const rows = latest.slice(firstRow * bucket.width * 4, (lastRow + 1) * bucket.width * 4);
    post({
      type: "pixels",
      revision: frameRevision,
      bucket,
      row: firstRow,
      pixels: rows,
    }, [rows.buffer]);
    firstRow = lastRow = -1;
  };

  const progress = (live: Uint8ClampedArray, pixel: number) => {
    const row = Math.floor(pixel / bucket.width);
    latest = live;
    if (firstRow < 0 || row < firstRow)
      firstRow = row;
    if (row > lastRow)
      lastRow = row;
    const now = performance.now();
    if (now - lastPost < PROGRESS_INTERVAL_MS)
      return;
    lastPost = now;
    flush();
  };

  if (request.phase === "cache") {
    // Grid only: the points stay on screen and the records stay in the grid (or travel, without shared memory).
    const written = yield* target.cachePrepass(request.bucket, progress);
    flush();
    const cells = !publishPrepared || shareable() ? undefined : target.irradianceGrid!.exportCells(written);
    post({ type: "cached", revision: frameRevision, job: request.bucket, cells }, cells ? [cells.buffer] : []);

    return;
  }

  const result = yield* target.renderBucket(bucket, progress);
  const pixels = new Uint8ClampedArray(result.pixels);
  // Without shared memory the denoiser accumulations travel with the bucket and the client returns them in `denoise`.
  const aov = target.frameAov && publishPrepared && !shareable() ? packBucketAov(target.frameAov, bucket) : undefined;
  post({
    type: "bucket",
    revision: frameRevision,
    bucket,
    pixels,
    aov,
  }, aov ? [pixels.buffer, aov.buffer] : [pixels.buffer]);
}

function restart() {
  cancel();
  if (!settings || !scene || !camera || !environment || disposed)
    return;
  // Coalesce orbit/resize events before touching geometry or allocating buffers.
  timer = setTimeout(() => {
    try {
      const width = Math.max(1, Math.floor(size.width * size.pixelRatio * settings!.renderScale));
      const height = Math.max(1, Math.floor(size.height * size.pixelRatio * settings!.renderScale));
      job = prepare(new CpuRasterizer(scene!, camera!, settings!, width, height, environment!), revision);
      pump();
    } catch(error) {
      fail(error);
    }
  }, 60);
}

async function loadEnvironment(descriptor: RenderEnvironment) {
  environmentAbort?.abort();
  const controller = new AbortController();
  environmentAbort = controller;
  environment = null;
  restart();

  try {
    const loaded = await CpuEnvironment.load(descriptor, controller.signal);
    if (controller.signal.aborted || disposed)
      return;
    environment = loaded;
    restart();
  } catch(error) {
    if (!controller.signal.aborted && !disposed)
      fail(error);
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (disposed)
    return;

  try {
    if (!["render", "pause", "dispose"].includes(request.type))
      revision = request.revision;

    switch (request.type) {
      case "frame":
        publishPrepared = request.publishPrepared;
        size = { width: request.width, height: request.height, pixelRatio: request.pixelRatio };
        settings = request.settings;
        camera = request.camera;

        if (request.scene) {
          scene = request.scene;
          pipeline = new CpuScenePipeline(scene);
        }

        // Loading the environment is asynchronous and restarts on its own once it is ready.
        if (request.environment)
          void loadEnvironment(request.environment);
        else
          restart();
        break;
      case "adopt":
        cancel();
        job = adopt(request.frame, revision);
        pump();
        break;
      case "reset":
        cancel();
        break;
      case "render":
        // A bucket from an old revision arriving after a restart: ignore it.
        if (request.revision !== revision || !renderer)
          break;
        queue.push(request);
        pump();
        break;
      case "records":
        if (request.revision === revision)
          renderer?.irradianceGrid?.importCells(request.cells);
        break;
      case "denoise":
        if (request.revision !== revision || !renderer)
          break;
        job = denoise(renderer, revision, request.slices);
        pump();
        break;
      case "pause":
        paused = request.paused;
        if (paused)
          clearTimeout(timer);
        else
          pump();
        break;
      case "dispose":
        disposed = true;
        cancel();
        environmentAbort?.abort();
        scene = null;
        pipeline = null;
        environment = null;
        skyCache = null;
        self.close();
        break;
    }
  } catch(error) {
    fail(error);
  }
};
