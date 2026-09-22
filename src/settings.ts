import type { FrameSettings } from "./frameSettings.js";

export type RenderToneMapping = "aces" | "agx" | "neutral" | "linear";
export type RenderBackgroundMode = "transparent" | "environment" | "color";

/** How the renderer shades a frame. Start from `DEFAULT_RENDER_SETTINGS`; setters take a `RenderSettingsInput`. */
export interface RenderSettings {
  /** Multiplier on the image size. */
  renderScale: number;
  /** Antialiasing samples per pixel. */
  samples: number;
  /** Bucket size in pixels. */
  tileSize: number;
  toneMapping: RenderToneMapping;
  exposure: number;
  background: {
    /** `"environment"` paints the environment, `"color"` a solid color, `"transparent"` nothing. */
    mode: RenderBackgroundMode;
    /** Used by `mode: "color"`. */
    color: string;
  };
  /** Ray-traced shadows. Only meshes with `castShadow` / `receiveShadow` take part. */
  shadows: {
    enabled: boolean;
    /** Angular diameter of the lights in degrees. `0` gives hard shadows. */
    softness: number;
  };
  /** Ray-traced ambient occlusion. Skipped while global illumination is on, which accounts for it. */
  ambientOcclusion: {
    enabled: boolean;
    /** Ray length, in world units. */
    distance: number;
    intensity: number;
  };
  /** Diffuse path tracing. */
  globalIllumination: {
    enabled: boolean;
    /** Paths per shaded pixel, 1 to 1024. */
    samples: number;
    /** 1 to 8. */
    bounces: number;
    intensity: number;
    /** Maximum indirect sample luminance. `0` disables clamping. */
    clamp: number;
    /** Compute indirect light on a pixel grid and interpolate. */
    cache: boolean;
    /** Grid spacing in pixels. */
    cacheSpacing: number;
    /** Denoise the indirect light after the last bucket. */
    denoise: boolean;
  };
  depthOfField: {
    enabled: boolean;
    /** Distance to the focal plane, in world units. */
    focusDistance: number;
    /** f-number. */
    aperture: number;
    /** Polygon sides of the bokeh. `0` is circular. */
    blades: number;
  };
  /** Linear fog in view depth, over every surface. Without it, `scene.fog` applies to the materials that accept it. */
  fog: {
    enabled: boolean;
    color: string;
    near: number;
    far: number;
  };
  /** `0` to `1`. */
  vignette: number;
  /** `0` to `1`. */
  grain: number;
}

/** Any subset of the settings, group by group: `{ depthOfField: { enabled: true } }` keeps the other fields. */
export type RenderSettingsInput = {
  [K in keyof RenderSettings]?: RenderSettings[K] extends object ? Partial<RenderSettings[K]> : RenderSettings[K];
};

/** Balanced defaults: four antialiasing samples, soft shadows and ambient occlusion, global illumination off. */
export const DEFAULT_RENDER_SETTINGS: Readonly<RenderSettings> = deepFreeze({
  renderScale: 1,
  samples: 4,
  tileSize: 64,
  toneMapping: "aces",
  exposure: 1,
  background: { mode: "environment", color: "#1a1a1a" },
  shadows: { enabled: true, softness: 1 },
  ambientOcclusion: { enabled: true, distance: 1.5, intensity: 1 },
  globalIllumination: {
    enabled: false, samples: 64, bounces: 3, intensity: 1, clamp: 0, cache: true, cacheSpacing: 4, denoise: true,
  },
  depthOfField: { enabled: false, focusDistance: 10, aperture: 2.8, blades: 0 },
  fog: { enabled: false, color: "#b8c4d0", near: 10, far: 60 },
  vignette: 0,
  grain: 0,
});

/** Settings with `input` applied over `base`, group by group. Neither argument is modified. */
export function mergeSettings(base: RenderSettings, input: RenderSettingsInput = {}): RenderSettings {
  const merged = { ...base } as Record<string, unknown>;

  for (const [key, value] of Object.entries(input)) {
    if (value === undefined)
      continue;
    const current = merged[key];
    merged[key] = value && typeof value === "object" && current && typeof current === "object"
      ? { ...current, ...value } : value;
  }

  return merged as unknown as RenderSettings;
}

/** Light that surrounds the scene, lights it, and paints the background when `background.mode` is `"environment"`. */
export type RenderEnvironment = ({
  /** Equirectangular `.hdr` image. */
  kind: "hdri";
  url: string;
} | {
  /** Vertical gradient from `topColor` (default white) to `bottomColor` (default black), shaped by `exponent`. */
  kind: "gradient";
  topColor?: string;
  bottomColor?: string;
  /** Defaults to 2. */
  exponent?: number;
}) & {
  /** Strength of the light it casts. Defaults to 1. */
  intensity?: number;
  /** Rotation around the vertical axis, in degrees. Defaults to 0. */
  rotation?: number;
};

export const DEFAULT_RENDER_ENVIRONMENT: Readonly<RenderEnvironment> = Object.freeze({
  kind: "gradient",
  topColor: "#9cb8d8",
  bottomColor: "#2b2a28",
  exponent: 2,
});

/** A camera snapshot: see `serializeCamera`. */
export interface CameraState {
  matrixWorld: number[];
  type?: "perspective" | "orthographic";
  fov?: number;
  aspect?: number;
  zoom?: number;
  left?: number;
  right?: number;
  top?: number;
  bottom?: number;
  /** Actual Three.js projection; takes precedence over the individual fields. */
  projectionMatrix?: number[];
  near: number;
  far: number;
}

/** The environment without its intensity and rotation: what has to be loaded again when it changes. */
export function environmentSource(environment: RenderEnvironment): RenderEnvironment {
  const { intensity: _intensity, rotation: _rotation, ...source } = environment;

  return source;
}

/** The engine's flat frame settings for these settings and environment. */
export function toFrameSettings(settings: RenderSettings, environment: RenderEnvironment): Required<FrameSettings> {
  const { shadows, ambientOcclusion: ao, globalIllumination: gi, depthOfField: dof, fog } = settings;

  return {
    renderScale: settings.renderScale,
    maxSamples: settings.samples,
    tileSize: settings.tileSize,
    tonemapping: settings.toneMapping,
    exposure: settings.exposure,
    backgroundMode: settings.background.mode,
    backgroundColor: settings.background.color,
    environmentIntensity: environment.intensity ?? 1,
    environmentRotation: environment.rotation ?? 0,
    shadows: shadows.enabled,
    shadowSoftness: shadows.softness,
    ambientOcclusion: ao.enabled,
    aoDistance: ao.distance,
    aoIntensity: ao.intensity,
    globalIllumination: gi.enabled,
    giSamples: gi.samples,
    giBounces: gi.bounces,
    giIntensity: gi.intensity,
    giClamp: gi.clamp,
    giCache: gi.cache,
    giCacheSpacing: gi.cacheSpacing,
    giDenoise: gi.denoise,
    dofEnabled: dof.enabled,
    dofFocusDistance: dof.focusDistance,
    dofAperture: dof.aperture,
    bokehBlades: dof.blades,
    fogEnabled: fog.enabled,
    fogColor: fog.color,
    fogNear: fog.near,
    fogFar: fog.far,
    vignette: settings.vignette,
    grain: settings.grain,
  };
}

function deepFreeze<T extends object>(value: T): T {
  for (const child of Object.values(value))
    if (child && typeof child === "object")
      deepFreeze(child);

  return Object.freeze(value);
}
