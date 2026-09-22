import type { Object3D, Scene } from "three";
import type { CpuComputeSnapshot } from "./compute.js";
import type { GpgpuPipelineDescriptor } from "./shaderTypes.js";
import type { UniformValue } from "./sceneSerialization.js";
import type { VolumetricMedium } from "./volumetricLight.js";

/**
 * Options the renderer reads from Three objects, all under `userData.cpuRenderer`. Everything here is optional:
 * a plain Three scene renders without any of it.
 */
export const USER_DATA_KEY = "cpuRenderer";

/** `scene.userData.cpuRenderer` */
export interface CpuRendererSceneData {
  /** A sky computed by a shader, used instead of the renderer's environment. */
  sky?: ProceduralSky;
}

/** `object.userData.cpuRenderer`, on any object. */
export interface CpuRendererObjectData {
  /** Leaves the object and its children out of the render, for helpers that only make sense on the GPU. */
  exclude?: boolean;
}

/** `light.userData.cpuRenderer`, on a spot, point or directional light. */
export interface CpuRendererLightData extends CpuRendererObjectData {
  /** Scatters the light in the air along every view ray: `true` for the default medium, or the medium itself. */
  volumetric?: boolean | Partial<VolumetricMedium>;
}

/** `material.userData.cpuRenderer`, on a `ShaderMaterial` or a material patched in `onBeforeCompile`. */
export interface CpuRendererMaterialData {
  /** Value of the shader's `time` uniform in the render. */
  time?: number;
  /**
   * Renders planar reflection and refraction captures for a flat surface, as `tReflectionMap`,
   * `tEnvironmentReflectionMap`, `tRefractionMap` and `tRefractionDepth`, with the matrices `reflectionMatrix`
   * and `refractionMatrix`. The surface's local +Y is its normal. Planar surfaces never appear in captures.
   */
  planarCapture?: boolean;
  /** A simulation baked on the CPU before rendering; its outputs become uniforms of this material. */
  compute?: MaterialCompute;
  /** The live uniforms of a material patched in `onBeforeCompile`, read where `material.uniforms` does not exist. */
  uniforms?: Record<string, { value: unknown }>;
  /** Called before the scene is serialized, to bring the material up to date without `onBeforeRender` or a GPU. */
  prepare?: (scene: Scene) => void;
}

export interface MaterialCompute {
  pipeline: GpgpuPipelineDescriptor;
  /** Names the simulation: materials with the same id share one bake, kept until the scene changes. */
  id: string;
  /** What the pipeline's sizes and uniform bindings read by path. */
  data?: Record<string, unknown>;
  /** Outputs already computed, for example read back from the GPU: restored instead of simulated. */
  snapshot?: CpuComputeSnapshot | null;
}

/** A sky shader: `fragmentShader` receives `vDirection` and writes linear radiance to `gl_FragColor`. */
export interface ProceduralSky {
  fragmentShader: string;
  /** Painted over the baked sky in the background only, per pixel (sun and moon discs, for example). */
  foregroundShader?: string;
  uniforms: Record<string, UniformValue>;
  /** Strength of the light the sky casts. */
  intensity: number;
  /** Whether the sky also paints the background. */
  background: boolean;
}

/** The renderer's options on a Three object, or an empty object. */
export function cpuRendererData<T extends object = CpuRendererObjectData>(
  object: Pick<Object3D, "userData">): T {
  return (object.userData[USER_DATA_KEY] ?? {}) as T;
}
