import type { RenderBackgroundMode, RenderToneMapping } from "./settings.js";

/**
 * The flat settings the engine reads for one frame. `CpuRenderer` builds them from `RenderSettings` and the
 * environment's intensity and rotation (`toFrameSettings`); use them directly only with the engine classes.
 * Missing optional fields disable their effect.
 */
export interface FrameSettings {
  renderScale: number;
  maxSamples: number;
  tileSize: number;
  shadows: boolean;
  environmentIntensity: number;
  /** Degrees. */
  environmentRotation: number;
  backgroundMode: RenderBackgroundMode;
  backgroundColor: string;
  dofEnabled: boolean;
  dofFocusDistance: number;
  dofAperture: number;
  bokehBlades: number;
  tonemapping: RenderToneMapping;
  exposure: number;
  /** Diffuse path tracing. */
  globalIllumination?: boolean;
  giSamples?: number;
  giBounces?: number;
  giIntensity?: number;
  /** Maximum indirect sample luminance in linear light; zero disables clamping. */
  giClamp?: number;
  /** Irradiance cache: indirect light computed on a pixel grid and interpolated between the points. */
  giCache?: boolean;
  /** Cache grid spacing, in pixels. */
  giCacheSpacing?: number;
  /** Indirect-light denoiser over the whole image, after the last bucket. */
  giDenoise?: boolean;
  ambientOcclusion?: boolean;
  aoDistance?: number;
  aoIntensity?: number;
  /** Apparent angular diameter of the lights, in degrees; zero keeps hard shadows. */
  shadowSoftness?: number;
  fogEnabled?: boolean;
  fogColor?: string;
  fogNear?: number;
  fogFar?: number;
  vignette?: number;
  grain?: number;
}

/** `DEFAULT_RENDER_SETTINGS` in the engine's flat form, with the default environment's intensity and rotation. */
export const DEFAULT_FRAME_SETTINGS: Readonly<Required<FrameSettings>> = Object.freeze({
  renderScale: 1,
  maxSamples: 4,
  tileSize: 64,
  shadows: true,
  environmentIntensity: 1,
  environmentRotation: 0,
  backgroundMode: "environment",
  backgroundColor: "#1a1a1a",
  dofEnabled: false,
  dofFocusDistance: 10,
  dofAperture: 2.8,
  bokehBlades: 0,
  tonemapping: "aces",
  exposure: 1,
  globalIllumination: false,
  giSamples: 64,
  giBounces: 3,
  giIntensity: 1,
  giClamp: 0,
  giCache: true,
  giCacheSpacing: 4,
  giDenoise: true,
  ambientOcclusion: true,
  aoDistance: 1.5,
  aoIntensity: 1,
  shadowSoftness: 1,
  fogEnabled: false,
  fogColor: "#b8c4d0",
  fogNear: 10,
  fogFar: 60,
  vignette: 0,
  grain: 0,
});
