/**
 * Compute pipelines described as data: float buffers and the fragment passes that fill them. A material runs one
 * through `userData.cpuRenderer.compute` (see `CpuRendererMaterialData`).
 */

export type ShaderUniformType = "float" | "vec2" | "vec3" | "vec4" | "color" | "texture" | "int" | "bool";

/** A uniform whose value is read from the object's data by dot-path, with a fallback. */
export interface ShaderUniformBinding {
  name: string;
  type: ShaderUniformType;
  arrayLength?: number;
  path?: string;
  paths?: string[];
  components?: string[];
  default: number | number[] | number[][] | string | boolean;
}

/** A fixed size, or one read from the object's data (`log2` derives FFT stage counts). */
export type GpgpuSize = number | { path: string; default: number; transform?: "log2" };

/** One float RGBA buffer of a compute pipeline. */
export interface GpgpuBufferSpec {
  name: string;
  width: GpgpuSize;
  height: GpgpuSize;
  /** Serialized into runtime snapshots so a paused simulation resumes exactly. */
  snapshot?: boolean;
  /** Two alternating slots: a pass that reads and writes this buffer reads the previous one. */
  pingPong?: boolean;
}

/** One full-screen fragment pass over the pipeline buffers. */
export interface GpgpuPassSpec {
  name: string;
  /** Full fragment shader, `main()` included. Inputs are declared as same-named `sampler2D` uniforms. */
  fragmentShader: string;
  inputs: string[];
  output: string;
  uniforms?: ShaderUniformBinding[];
  /** Repeats the pass, exposing `uniform int uIteration`. */
  iterations?: GpgpuSize;
  trigger: "onConfigChange" | "everyFrame";
}

/** A compute pipeline described entirely as data: buffers, passes and the material uniforms they feed. */
export interface GpgpuPipelineDescriptor {
  buffers: GpgpuBufferSpec[];
  passes: GpgpuPassSpec[];
  /** Material uniform name -> buffer bound to it. */
  outputs: Record<string, string>;
}
