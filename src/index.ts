// The public API: render a Three.js scene on the CPU, into `image` and the `pixels` event. `attachCanvas` paints on
// a canvas; `@tinypoly/cpu-renderer/node` runs the workers in Node. The engine behind it is `@tinypoly/cpu-renderer/engine`.

export { CpuRenderer, MAX_RENDER_WORKERS, createDefaultWorker, defaultWorkerCount, isAbortError } from "./renderer.js";

export type {
  CpuRendererEvents,
  CpuRendererOptions,
  RenderCamera,
  RenderImage,
  RenderPhase,
  RenderPixels,
  RenderScene,
  RenderWorker,
  WorkerFactory,
} from "./renderer.js";

export { attachCanvas } from "./canvas.js";

export type { Listener } from "./events.js";
export type { Bucket } from "./rasterizer.js";

// Settings and environment.
export { DEFAULT_RENDER_ENVIRONMENT, DEFAULT_RENDER_SETTINGS, mergeSettings } from "./settings.js";

export type {
  CameraState,
  RenderBackgroundMode,
  RenderEnvironment,
  RenderSettings,
  RenderSettingsInput,
  RenderToneMapping,
} from "./settings.js";

// Scene snapshots and change detection.
export { serializeScene } from "./sceneSerialization.js";
export type { SerializedScene, SerializedSceneResult } from "./sceneSerialization.js";
export { serializeCamera } from "./camera.js";
export { sceneSignature } from "./sceneSignature.js";
export { shareable } from "./sharedBuffers.js";

// Options read from `userData.cpuRenderer` on scenes, objects, lights and materials.
export { USER_DATA_KEY } from "./extensions.js";

export type {
  CpuRendererLightData,
  CpuRendererMaterialData,
  CpuRendererObjectData,
  CpuRendererSceneData,
  MaterialCompute,
  ProceduralSky,
} from "./extensions.js";

export type { CpuComputeSnapshot } from "./compute.js";
export type { VolumetricMedium } from "./volumetricLight.js";

export type {
  GpgpuBufferSpec,
  GpgpuPassSpec,
  GpgpuPipelineDescriptor,
  GpgpuSize,
  ShaderUniformBinding,
  ShaderUniformType,
} from "./shaderTypes.js";
