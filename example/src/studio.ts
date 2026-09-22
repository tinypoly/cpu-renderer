import * as THREE from "three";

// A photo studio as an HDR environment, painted in code so the example ships no .hdr file: a neutral gray room
// with four softboxes. The same pixels reach the CPU renderer (as a Radiance file behind a blob: URL, which its
// workers fetch) and the WebGL preview (as a texture).

export interface StudioEnvironment {
  /** A Radiance .hdr file, for `{ kind: "hdri", url }`. */
  url: string;
  /** The same picture for WebGL, ready for `scene.environment` and `scene.background`. */
  texture: THREE.DataTexture;
}

interface Softbox {
  /** Center: azimuth as in Three's equirectangular maps, atan2(z, x), and elevation, in degrees. */
  azimuth: number;
  elevation: number;
  /** Half extent across and up, in degrees. */
  width: number;
  height: number;
  radiance: [number, number, number];
}

// The key light sits where the scenes put their sun, (-3, 6, 4), so reflections and shadows agree.
const SOFTBOXES: Softbox[] = [
  { azimuth: 127, elevation: 48, width: 24, height: 15, radiance: [7, 6.6, 6.1] },
  { azimuth: 25, elevation: 18, width: 7, height: 24, radiance: [1.8, 1.9, 2.1] },
  { azimuth: -90, elevation: 42, width: 34, height: 5, radiance: [4, 4, 4] },
  { azimuth: 0, elevation: 84, width: 60, height: 5, radiance: [1.5, 1.5, 1.6] },
];

/** 1 inside `edge`, 0 outside, over a 3° feathered border. */
const feather = (edge: number, value: number) => Math.min(1, Math.max(0, (edge - value) / 3 + .5));

function radiance(azimuth: number, elevation: number): number[] {
  // A dim room: a lighter ceiling, a darker floor.
  const up = Math.sin(elevation * Math.PI / 180), room = .12 + .07 * Math.max(0, up) - .08 * Math.max(0, -up);
  const color = [room * .98, room, room * 1.04];

  for (const box of SOFTBOXES) {
    const across = ((azimuth - box.azimuth) % 360 + 540) % 360 - 180, lift = elevation - box.elevation;

    const weight = feather(box.width, Math.abs(across) * Math.cos(elevation * Math.PI / 180))
      * feather(box.height, Math.abs(lift));

    for (let c = 0; c < 3; c++) color[c] += box.radiance[c] * weight;
  }

  return color;
}

export function studioEnvironment(width = 1024): StudioEnvironment {
  const height = width / 2;
  const header = new TextEncoder().encode(`#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y ${height} +X ${width}\n`);
  // Radiance stores flat scanlines top row first; WebGL reads its texture bottom row first.
  const file = new Uint8Array(header.length + width * height * 4), halves = new Uint16Array(width * height * 4);
  file.set(header);

  for (let y = 0; y < height; y++) {
    const elevation = (.5 - (y + .5) / height) * 180;

    for (let x = 0; x < width; x++) {
      const rgb = radiance(((x + .5) / width - .5) * 360, elevation), top = Math.max(...rgb);
      const byte = header.length + (y * width + x) * 4;

      if (top > 1e-32) {
        const exponent = Math.ceil(Math.log2(top) + 1e-9), scale = 256 / 2 ** exponent;
        file.set([...rgb.map(c => Math.min(255, Math.floor(c * scale))), exponent + 128], byte);
      }

      halves.set([...rgb, 1].map(THREE.DataUtils.toHalfFloat), ((height - 1 - y) * width + x) * 4);
    }
  }

  const texture = new THREE.DataTexture(halves, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.magFilter = texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;

  return { url: URL.createObjectURL(new Blob([file], { type: "image/vnd.radiance" })), texture };
}
