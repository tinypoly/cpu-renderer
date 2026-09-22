import type { SerializedLight } from "./sceneSerialization.js";
import { add, clamp, dot, normalize, scale, sub } from "./math.js";

const smoothstep = (a: number, b: number, value: number) => {
  const t = b > a ? clamp((value - a) / (b - a)) : Number(value >= b);

  return t * t * (3 - 2 * t);
};

/**
 * The air a volumetric light shines through, set with `light.userData.cpuRenderer = { volumetric: { density, ... } }`
 * (`true` takes every default).
 */
export interface VolumetricMedium {
  /** Scattering and absorption per unit length: denser air glows brighter near the light and dims farther off. */
  density?: number;
  /**
   * Henyey-Greenstein asymmetry, from -0.95 to 0.95. Fog and haze scatter forward (positive): the beam is brightest
   * looking toward the light and fainter seen from the side. Zero scatters evenly, as before.
   */
  anisotropy?: number;
  /**
   * Multiple scattering, from 0 to 1: light that bounces between droplets escapes a spot light's cone, softening its
   * edge into a wide, fading glow around the beam. Zero keeps the cone sharp.
   */
  spread?: number;
}

/** The medium and falloff a GPU volumetric light uses, integrated in linear light before output tone mapping. */
export class CpuVolumetricLight {
  private lights;

  constructor(lights: SerializedLight[]) {
    this.lights = lights.filter(l => l.volumetric
      && (l.type === "PointLight" || l.type === "SpotLight")).map(light => {
      const density = Math.max(0, light.medium?.density ?? .01), spread = clamp(light.medium?.spread ?? 0);
      // Scattered light reaches past the cone: up to 1.5 rad (86°) off the axis with a full spread.
      const outer = light.angle + spread * Math.max(0, 1.5 - light.angle);

      return {
        light, density, spread, outer, anisotropy: clamp(light.medium?.anisotropy ?? 0, -.95, .95),
        axis: normalize(sub(light.target, light.position)),
        radius: light.distance > 0 ? light.distance : -3 * Math.log(.0001),
      };
    });
  }

  get active() {
    return this.lights.length > 0;
  }

  sample(origin: number[], direction: number[], distance: number,
    visible: (p: number[], d: number[], distance: number) => boolean, jitter = .5,
    fogVisibility?: (p: number[]) => number): number[] {
    const result = [0, 0, 0];
    let tStart = Math.min(distance, 40), tEnd = 0;

    for (const { light, axis, radius, outer } of this.lights) {
      const delta = sub(origin, light.position), b = dot(delta, direction);
      const disc = b * b - dot(delta, delta) + radius * radius;
      if (disc <= 0) continue;
      const start = Math.max(0, -b - Math.sqrt(disc));
      const end = Math.min(distance, 40, -b + Math.sqrt(disc));
      if (end <= start) continue;
      const cuts = [start, end];
      const cone = Math.cos(outer);

      if (light.type === "SpotLight") {
        const ar = dot(axis, direction), ac = dot(axis, delta), cos2 = cone * cone;
        const qa = ar * ar - cos2, qb = ar * ac - dot(direction, delta) * cos2;
        const qc = ac * ac - dot(delta, delta) * cos2, discriminant = qb * qb - qa * qc;

        const roots = Math.abs(qa) < 1e-8 ? (Math.abs(qb) > 1e-8 ? [-qc / (2 * qb)] : [])
          : discriminant >= 0 ? [(-qb - Math.sqrt(discriminant)) / qa, (-qb + Math.sqrt(discriminant)) / qa] : [];

        cuts.push(...roots.filter(t => t > start && t < end));
        cuts.sort((a, b) => a - b);
      }

      for (let segment = 1; segment < cuts.length; segment++) {
        const lo = cuts[segment - 1], hi = cuts[segment];
        if (light.type === "SpotLight"
          && dot(normalize(add(delta, scale(direction, (lo + hi) / 2))), axis) < cone) continue;
        tStart = Math.min(tStart, lo);
        tEnd = Math.max(tEnd, hi);
      }
    }

    if (tEnd <= tStart) return result;
    // The GPU uses one shared march across the union of the light intervals; each light's medium dims on its own.
    const step = (tEnd - tStart) / 64;
    const transmittance = this.lights.map(({ density }) => Math.exp(-density * tStart));
    const extinction = this.lights.map(({ density }) => Math.exp(-density * step));

    for (let i = 0; i < 64; i++) {
      const t = tStart + (i + jitter) * step, p = add(origin, scale(direction, t));
      // Scene fog between the camera and this sample dims what it scatters, as it dims a surface there.
      const fogged = fogVisibility ? fogVisibility(p) : 1;
      if (fogged <= .0001) break;
      let lit = false;

      this.lights.forEach(({ light, axis, density, anisotropy, spread, outer }, index) => {
        if (transmittance[index] <= .003) return;
        lit = true;
        const toLight = sub(light.position, p), dist = Math.hypot(...toLight), toward = scale(toLight, 1 / dist);
        let attenuation = Math.exp(-dist / ((light.distance > 0 ? light.distance : 10) * .3));
        if (light.distance > 0) attenuation *= 1 - smoothstep(light.distance * .8, light.distance, dist);

        if (light.type === "SpotLight") {
          const along = -dot(toward, axis);
          const core = smoothstep(Math.cos(light.angle), Math.cos(light.angle * (1 - light.penumbra)), along);
          // The light scattered out of the cone fades from its edge to the outer angle.
          attenuation *= spread > 0 ? Math.max(core, spread * smoothstep(Math.cos(outer),
            Math.cos(light.angle * (1 - light.penumbra)), along) ** 2) : core;
        }

        // Shadowing follows GPU volumetric lights, which only have shadow maps for spots, not point lights.
        if (attenuation <= .0001 || (light.type === "SpotLight" && light.castShadow
          && !visible(p, toward, dist))) return;
        // Henyey-Greenstein, relative to isotropic scattering: the angle between the light's travel and the view.
        const g = anisotropy, cosine = dot(toward, direction);
        const phase = g === 0 ? 1 : (1 - g * g) / (1 + g * g - 2 * g * cosine) ** 1.5;
        const weight = attenuation * density * step * transmittance[index] * fogged * phase * 1.5 * light.intensity;
        for (let c = 0; c < 3; c++) result[c] += light.color[c] * weight;
      });

      if (!lit) break;
      for (let index = 0; index < transmittance.length; index++) transmittance[index] *= extinction[index];
    }

    // The shader compresses the sum of all lights BEFORE additive composition and frame tone mapping.
    return result.map(v => v / (1 + v));
  }
}
