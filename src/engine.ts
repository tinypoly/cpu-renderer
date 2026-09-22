// The engine behind `CpuRenderer`, for custom workers, Node scripts and tests. It follows the package version but is
// lower level: its classes take flat `FrameSettings` and serialized scenes, and may change more often than the main API.

// Rasterizer, environment and scene preparation.
export { CpuRasterizer, createBuckets } from "./rasterizer.js";
export type { Bucket, BucketResult, PrepassJob, RasterTargetOptions } from "./rasterizer.js";
export { CpuEnvironment } from "./environment.js";
export type { CpuEnvironmentSnapshot } from "./environment.js";
export { CpuScenePipeline } from "./pipeline.js";
export { CpuVolumetricLight, type VolumetricMedium } from "./volumetricLight.js";
export type { BvhData } from "./bvh.js";

// Frame settings: the flat form the engine reads.
export { DEFAULT_FRAME_SETTINGS } from "./frameSettings.js";
export type { FrameSettings } from "./frameSettings.js";
export { environmentSource, toFrameSettings } from "./settings.js";

// GLSL interpreter and compute pipelines.
export { CpuShader, ShaderError, matrixValue } from "./glsl.js";
export type { DerivativeContext, ShaderContext, Value } from "./glsl.js";
export { CpuComputePipeline, computeConfigKey } from "./compute.js";
export type { CpuComputeSnapshot } from "./compute.js";

// Serialized scene data.
export { prepareCpuMaterials, serializeTexture, serializeValue } from "./sceneSerialization.js";

export type {
  SerializedAttribute,
  SerializedFog,
  SerializedLight,
  SerializedMaterial,
  SerializedMesh,
  SerializedMipLevel,
  SerializedScene,
  SerializedShader,
  SerializedTexture,
  UniformValue,
} from "./sceneSerialization.js";

// Color and textures.
export { encodeColor, toneMap, toneMap3 } from "./color.js";
export type { TextureSampling } from "./texture.js";
export { linearToSrgb, sampleTexture, srgbToLinear, textureFootprint, textureLod } from "./texture.js";

// Shared memory (SharedArrayBuffer when the page is cross-origin isolated).
export { shareable, sharedCopy, sharedFloat32, sharedFloat64, sharedInt32, sharedUint32, sharedUint8 } from "./sharedBuffers.js";

// Worker pool and the protocol between `CpuRenderer` and its workers.
export { BucketScheduler } from "./bucketScheduler.js";
export { effectiveWorkerCount } from "./renderer.js";
export { EventEmitter } from "./events.js";
export type { BucketAovSlice, FramePhase, PreparedFrame, WorkerRequest, WorkerResponse } from "./protocol.js";
