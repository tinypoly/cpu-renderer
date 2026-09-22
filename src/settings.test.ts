import { describe, expect, it } from "vitest";
import { DEFAULT_FRAME_SETTINGS } from "./frameSettings.js";
import {
  DEFAULT_RENDER_ENVIRONMENT, DEFAULT_RENDER_SETTINGS, environmentSource, mergeSettings, toFrameSettings,
} from "./settings.js";

describe("render settings", () => {
  it("convert to the engine's flat defaults", () => {
    expect(toFrameSettings(DEFAULT_RENDER_SETTINGS, DEFAULT_RENDER_ENVIRONMENT)).toEqual(DEFAULT_FRAME_SETTINGS);
  });

  it("map every group to its flat fields", () => {
    const settings = mergeSettings(DEFAULT_RENDER_SETTINGS, {
      renderScale: 0.5, samples: 16, tileSize: 32, toneMapping: "neutral", exposure: 2, vignette: 0.3, grain: 0.1,
      background: { mode: "transparent", color: "#ffffff" },
      shadows: { enabled: false, softness: 3 },
      ambientOcclusion: { enabled: false, distance: 2, intensity: 0.5 },
      globalIllumination: { enabled: true, samples: 8, bounces: 2, intensity: 0.8, clamp: 10, cache: false,
        cacheSpacing: 8, denoise: false },
      depthOfField: { enabled: true, focusDistance: 4, aperture: 1.8, blades: 6 },
      fog: { enabled: true, color: "#000000", near: 1, far: 5 },
    });

    expect(toFrameSettings(settings, { kind: "gradient", intensity: 2, rotation: 45 })).toEqual({
      renderScale: 0.5, maxSamples: 16, tileSize: 32, tonemapping: "neutral", exposure: 2, vignette: 0.3, grain: 0.1,
      backgroundMode: "transparent", backgroundColor: "#ffffff", environmentIntensity: 2, environmentRotation: 45,
      shadows: false, shadowSoftness: 3, ambientOcclusion: false, aoDistance: 2, aoIntensity: 0.5,
      globalIllumination: true, giSamples: 8, giBounces: 2, giIntensity: 0.8, giClamp: 10, giCache: false,
      giCacheSpacing: 8, giDenoise: false, dofEnabled: true, dofFocusDistance: 4, dofAperture: 1.8, bokehBlades: 6,
      fogEnabled: true, fogColor: "#000000", fogNear: 1, fogFar: 5,
    });
  });

  it("merge group by group without modifying either side", () => {
    const input = { fog: { enabled: true }, exposure: 1.5, shadows: undefined };
    const merged = mergeSettings(DEFAULT_RENDER_SETTINGS, input);
    expect(merged.fog).toEqual({ ...DEFAULT_RENDER_SETTINGS.fog, enabled: true });
    expect(merged.exposure).toBe(1.5);
    expect(merged.shadows).toBe(DEFAULT_RENDER_SETTINGS.shadows);
    expect(DEFAULT_RENDER_SETTINGS.fog.enabled).toBe(false);
    expect(input).toEqual({ fog: { enabled: true }, exposure: 1.5, shadows: undefined });
  });

  it("keep the defaults frozen, groups included", () => {
    expect(Object.isFrozen(DEFAULT_RENDER_SETTINGS)).toBe(true);
    expect(Object.isFrozen(DEFAULT_RENDER_SETTINGS.globalIllumination)).toBe(true);
  });

  it("separate an environment's source from its intensity and rotation", () => {
    expect(environmentSource({ kind: "hdri", url: "a.hdr", intensity: 2, rotation: 30 })).toEqual({ kind: "hdri", url: "a.hdr" });
  });
});
