import { RepeatWrapping, MirroredRepeatWrapping, SRGBColorSpace, NearestFilter, LinearFilter,
  NearestMipmapNearestFilter, NearestMipmapLinearFilter, LinearMipmapNearestFilter, LinearMipmapLinearFilter } from "three";
import { sharedFloat32, sharedUint8 } from "./sharedBuffers.js";
import type { SerializedTexture, SerializedMipLevel } from "./sceneSerialization.js";
import { clamp } from "./math.js";

export const srgbToLinear = (v: number) => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;

export const linearToSrgb = (v: number) => v <= 0.0031308 ? v * 12.92 : 1.055 * Math.max(
  0,
  v,
) ** (1 / 2.4) - 0.055;

// Match the old Float32 snapshot exactly, including rounding before sRGB decoding.
const linearBytes = Float32Array.from({ length: 256 }, (_, i) => i / 255);
const srgbBytes = Float32Array.from(linearBytes, srgbToLinear);

function wrap(index: number, size: number, mode: number) {
  if (mode === RepeatWrapping)
    return ((index % size) + size) % size;

  if (mode === MirroredRepeatWrapping) {
    const i = ((index % (size * 2)) + size * 2) % (size * 2);

    return i < size ? i : size * 2 - 1 - i;
  }

  return clamp(index, 0, size - 1);
}

export interface TextureSampling {
  lod?: number;
  dx?: number[];
  dy?: number[];
  bias?: number;
}

export function isMipmapFilter(filter: number | undefined): boolean {
  return filter === NearestMipmapNearestFilter || filter === NearestMipmapLinearFilter
    || filter === LinearMipmapNearestFilter || filter === LinearMipmapLinearFilter;
}

/**
 * Level of detail and anisotropic taps of a lookup. The footprint uses unwrapped UV gradients, including the
 * material texture's linear UV transform. With `texture.anisotropy` above one, the level follows the footprint's
 * minor axis and up to that many taps spread along the major axis (`axis`, in the texture's UV space, spans the
 * whole footprint), like GPU anisotropic filtering. Explicit LOD disables anisotropy, as on the GPU.
 */
export function textureFootprint(texture: SerializedTexture, sampling: TextureSampling, transform = true):
{ lod: number; taps: number; axis: number[] } {
  const bias = sampling.bias ?? 0;
  if (sampling.lod !== undefined) return { lod: sampling.lod + bias, taps: 1, axis: [0, 0] };
  if (!sampling.dx || !sampling.dy) return { lod: bias, taps: 1, axis: [0, 0] };
  const m = texture.matrix;

  const uvSpace = (d: number[]) => transform ? [m[0] * d[0] + m[3] * d[1], m[1] * d[0] + m[4] * d[1]] : d;
  const dx = uvSpace(sampling.dx), dy = uvSpace(sampling.dy);
  const px = Math.hypot(dx[0] * texture.width, dx[1] * texture.height);
  const py = Math.hypot(dy[0] * texture.width, dy[1] * texture.height);
  const major = Math.max(px, py, 1e-20), minor = Math.min(px, py);
  // A magnified minor axis counts as one texel: the taps then spread only what the major axis minifies.
  const ratio = clamp(major / Math.max(minor, 1), 1, Math.max(1, texture.anisotropy ?? 1));

  return { lod: Math.log2(major / ratio) + bias, taps: Math.ceil(ratio - 1e-6), axis: px >= py ? dx : dy };
}

/** LOD of a lookup; see `textureFootprint`. */
export function textureLod(texture: SerializedTexture, sampling: TextureSampling, transform = true): number {
  return textureFootprint(texture, sampling, transform).lod;
}

/** Filtering and mipmap interpolation happen in linear light; shader UVs bypass material transforms. */
export function sampleTexture(texture: SerializedTexture, uv: number[], transform = true,
  sampling?: TextureSampling): number[] {
  const m = texture.matrix;
  const u = transform ? m[0] * uv[0] + m[3] * uv[1] + m[6] : uv[0];
  let v = transform ? m[1] * uv[0] + m[4] * uv[1] + m[7] : uv[1];
  if (!Number.isFinite(u) || !Number.isFinite(v)) return [0, 0, 0, 0];
  if (texture.flipY) v = 1 - v;
  if (!sampling) return sampleLod(texture, u, v, 0);
  const { lod, taps, axis } = textureFootprint(texture, sampling, transform);
  if (taps < 2) return sampleLod(texture, u, v, lod);
  // Taps are symmetric about the center, so the flipped v direction needs no sign change.
  const sum = [0, 0, 0, 0];

  for (let i = 0; i < taps; i++) {
    const t = (i + .5) / taps - .5, tap = sampleLod(texture, u + t * axis[0], v + t * axis[1], lod);
    for (let c = 0; c < 4; c++) sum[c] += tap[c];
  }

  return sum.map(c => c / taps);
}

/** Filters the transformed coordinates at `lod`: min/mag filters, mipmap selection and interpolation. */
function sampleLod(texture: SerializedTexture, u: number, v: number, lod: number): number[] {
  const mag = texture.magFilter ?? (texture.nearest ? NearestFilter : LinearFilter);
  const min = texture.minFilter ?? mag;

  const transition = mag === LinearFilter && (min === NearestMipmapNearestFilter || min === NearestMipmapLinearFilter)
    ? .5 : 0;

  if (Number.isNaN(lod) || lod <= transition) return sampleLevel(texture, texture, u, v, mag === NearestFilter);
  const nearest = min === NearestFilter || min === NearestMipmapNearestFilter || min === NearestMipmapLinearFilter;
  if (!isMipmapFilter(min) || !texture.mipmaps?.length) return sampleLevel(texture, texture, u, v, nearest);
  const level = clamp(lod, 0, texture.mipmaps.length);
  const at = (i: number) => sampleLevel(texture, i === 0 ? texture : texture.mipmaps![i - 1], u, v, nearest);
  if (min === NearestMipmapNearestFilter || min === LinearMipmapNearestFilter) return at(Math.round(level));
  const low = Math.floor(level), weight = level - low;
  const a = at(low);
  if (!weight) return a;
  const b = at(low + 1);

  return a.map((v, c) => v * (1 - weight) + b[c] * weight);
}

function sampleLevel(texture: SerializedTexture, level: SerializedMipLevel,
  u: number, v: number, nearest: boolean): number[] {
  const { width, height, data } = level;
  const byte = data instanceof Uint8Array;
  const rgb = texture.colorSpace === SRGBColorSpace ? srgbBytes : linearBytes;
  u = u * width - 0.5;
  v = v * height - 0.5;

  if (nearest) {
    const o = (wrap(Math.floor(v + 0.5), height, texture.wrapT) * width
      + wrap(Math.floor(u + 0.5), width, texture.wrapS)) * 4;

    if (byte) return [rgb[data[o]], rgb[data[o + 1]], rgb[data[o + 2]], linearBytes[data[o + 3]]];

    return [data[o], data[o + 1], data[o + 2], data[o + 3]];
  }

  const x = Math.floor(u), y = Math.floor(v), tx = u - x, ty = v - y;
  // The four coordinates are the same for every channel: resolve each one only once.
  const x0 = wrap(x, width, texture.wrapS), x1 = wrap(x + 1, width, texture.wrapS);
  const y0 = wrap(y, height, texture.wrapT) * width, y1 = wrap(y + 1, height, texture.wrapT) * width;
  const o00 = (y0 + x0) * 4, o10 = (y0 + x1) * 4, o01 = (y1 + x0) * 4, o11 = (y1 + x1) * 4;
  const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;

  if (byte) {
    const result = new Array<number>(4);

    for (let c = 0; c < 4; c++) {
      const table = c === 3 ? linearBytes : rgb;
      result[c] = table[data[o00 + c]] * w00 + table[data[o10 + c]] * w10
        + table[data[o01 + c]] * w01 + table[data[o11 + c]] * w11;
    }

    return result;
  }

  return [
    data[o00] * w00 + data[o10] * w10 + data[o01] * w01 + data[o11] * w11,
    data[o00 + 1] * w00 + data[o10 + 1] * w10 + data[o01 + 1] * w01 + data[o11 + 1] * w11,
    data[o00 + 2] * w00 + data[o10 + 2] * w10 + data[o01 + 2] * w01 + data[o11 + 2] * w11,
    data[o00 + 3] * w00 + data[o10 + 3] * w10 + data[o01 + 3] * w01 + data[o11 + 3] * w11,
  ];
}

/** Area box reduction, including odd dimensions, with sRGB decoded before averaging. */
export function generateMipmaps(texture: Pick<SerializedTexture, "width" | "height" | "data" | "colorSpace">): SerializedMipLevel[] {
  const levels: SerializedMipLevel[] = [];
  let source: SerializedMipLevel = texture;
  const byte = texture.data instanceof Uint8Array, srgb = byte && texture.colorSpace === SRGBColorSpace;

  while (source.width > 1 || source.height > 1) {
    const width = Math.max(1, Math.floor(source.width / 2)), height = Math.max(1, Math.floor(source.height / 2));
    const data = byte ? sharedUint8(width * height * 4) : sharedFloat32(width * height * 4);
    const sx = source.width / width, sy = source.height / height;

    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const sums = [0, 0, 0, 0], x0 = x * sx, x1 = (x + 1) * sx, y0 = y * sy, y1 = (y + 1) * sy;

      for (let j = Math.floor(y0); j < Math.min(source.height, Math.ceil(y1)); j++)
        for (let i = Math.floor(x0); i < Math.min(source.width, Math.ceil(x1)); i++) {
          const weight = (Math.min(i + 1, x1) - Math.max(i, x0)) * (Math.min(j + 1, y1) - Math.max(j, y0)) / (sx * sy);
          const offset = (j * source.width + i) * 4;

          for (let c = 0; c < 4; c++) {
            const value = source.data[offset + c];
            sums[c] += weight * (byte ? (srgb && c < 3 ? srgbBytes[value] : linearBytes[value]) : value);
          }
        }

      const offset = (y * width + x) * 4;
      for (let c = 0; c < 4; c++) data[offset + c] = byte
        ? Math.round(clamp(srgb && c < 3 ? linearToSrgb(sums[c]) : sums[c]) * 255) : sums[c];
    }

    source = { width, height, data };
    levels.push(source);
  }

  return levels;
}
