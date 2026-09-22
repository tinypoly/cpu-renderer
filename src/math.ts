export type Vec3 = [
  number,
  number,
  number,
];

export const add = (a: number[], b: number[]): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: number[], b: number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: number[], b: number): Vec3 => [a[0] * b, a[1] * b, a[2] * b];
export const mul = (a: number[], b: number[]): Vec3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
export const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const cross = (a: number[], b: number[]): Vec3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
];

export const length = (a: number[]) => Math.sqrt(dot(a, a));
export const normalize = (a: number[]) => scale(a, 1 / (length(a) || 1));
export const mix = (a: number[], b: number[], t: number): Vec3 => add(scale(a, 1 - t), scale(b, t));
export const reflect = (i: number[], n: number[]) => sub(i, scale(n, 2 * dot(i, n)));

/** GLSL `refract`: zero vector on total internal reflection. */
export const refract = (i: number[], n: number[], eta: number): Vec3 => {
  const ni = dot(n, i), k = 1 - eta * eta * (1 - ni * ni);

  return k < 0 ? [0, 0, 0] : sub(scale(i, eta), scale(n, eta * ni + Math.sqrt(k)));
};

export const clamp = (v: number, min = 0, max = 1) => Math.max(min, Math.min(max, v));

export const transform4 = (m: number[], v: number[]): number[] =>
  [0, 1, 2, 3].map(r => m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * (v[3] ?? 1));

export const transform3 = (m: number[], v: number[]): Vec3 =>
  [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];

export const halton = (index: number, base: number) => {
  let result = 0, fraction = 1;

  while (index > 0) {
    fraction /= base;
    result += fraction * (index % base);
    index = Math.floor(index / base);
  }

  return result;
};
