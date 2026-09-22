import type { RenderToneMapping } from "./settings.js";
import { clamp, transform3 } from "./math.js";
import { linearToSrgb } from "./texture.js";

/** CPU equivalents of Three's ACES, AgX and Khronos neutral operators (Three MIT license). */
export function toneMap(rgb: number[], exposure: number, mode: RenderToneMapping): number[] {
  let c = rgb.map(v => Math.max(0, v * exposure));

  if (mode === "aces") {
    c = transform3([.59719, .076, .0284, .35458, .90834, .13383, .04823, .01566, .83777], c.map(v => v / .6));
    c = c.map(v => (v * (v + .0245786) - .000090537) / (v * (.983729 * v + .432951) + .238081));
    c = transform3([1.60475, -.10208, -.00327, -.53108, 1.10813, -.07276, -.07367, -.00605, 1.07602], c);
  } else if (mode === "agx") {
    c = transform3([.6274, .0691, .0164, .3293, .9195, .088, .0433, .0113, .8956], c);
    c
      = transform3(
        [
          .856627153315983,
          .137318972929847,
          .11189821299995,
          .0951212405381588,
          .761241990602591,
          .0767994186031903,
          .0482516061458583,
          .101439036467562,
          .811302368396859,
        ],
        c,
      );
    c = c.map(v => clamp((Math.log2(Math.max(v, 1e-10)) + 12.47393) / 16.499999));
    c
      = c.map(x =>
        15.5 * x ** 6 - 40.14 * x ** 5 + 31.96 * x ** 4 - 6.868 * x ** 3 + .4298 * x ** 2 + .1191 * x
        - .00232);
    c
      = transform3(
        [
          1.1271005818144368,
          -.1413297634984383,
          -.14132976349843826,
          -.11060664309660323,
          1.157823702216272,
          -.11060664309660294,
          -.016493938717834573,
          -.016493938717834257,
          1.2519364065950405,
        ],
        c,
      ).map(v => Math.max(
        0,
        v,
      ) ** 2.2);
    c = transform3([1.6605, -.1246, -.0182, -.5876, 1.1329, -.1006, -.0728, -.0083, 1.1187], c);
  } else if (mode === "neutral") {
    const x = Math.min(...c), offset = x < .08 ? x - 6.25 * x * x : .04;
    c = c.map(v => v - offset);
    const peak = Math.max(...c);

    if (peak >= .76) {
      const newPeak = 1 - .24 * .24 / (peak + .24 - .76), g = 1 - 1 / (.15 * (peak - newPeak) + 1);
      c = c.map(v => v * newPeak / peak * (1 - g) + newPeak * g);
    }
  }

  return c.map(v => clamp(v));
}

export function encodeColor(rgb: number[], exposure: number, mode: RenderToneMapping, mapped = true) {
  return (mapped ? toneMap(rgb, exposure, mode) : rgb).map(v => Math.round(clamp(linearToSrgb(v)) * 255));
}

/** Same curve as toneMap, without temporary arrays on the ACES path, the default for the per-pixel resolve. */
export function toneMap3(r: number, g: number, b: number, exposure: number, mode: RenderToneMapping): number[] {
  if (mode !== "aces")
    return toneMap([r, g, b], exposure, mode);
  r = Math.max(0, r * exposure) / .6; g = Math.max(0, g * exposure) / .6; b = Math.max(0, b * exposure) / .6;
  let x = .59719 * r + .35458 * g + .04823 * b;
  let y = .076 * r + .90834 * g + .01566 * b;
  let z = .0284 * r + .13383 * g + .83777 * b;
  x = (x * (x + .0245786) - .000090537) / (x * (.983729 * x + .432951) + .238081);
  y = (y * (y + .0245786) - .000090537) / (y * (.983729 * y + .432951) + .238081);
  z = (z * (z + .0245786) - .000090537) / (z * (.983729 * z + .432951) + .238081);

  return [
    clamp(1.60475 * x - .53108 * y - .07367 * z),
    clamp(-.10208 * x + 1.10813 * y - .00605 * z),
    clamp(-.00327 * x - .07276 * y + 1.07602 * z),
  ];
}
