/**
 * When the page is cross-origin isolated (COEP), buffers are allocated in a SharedArrayBuffer: every worker
 * in the pool reads the same memory instead of receiving its own copy. Otherwise, plain arrays.
 */
export const shareable = () => typeof SharedArrayBuffer !== "undefined" && Boolean(globalThis.crossOriginIsolated);

export const sharedFloat32 = (length: number): Float32Array =>
  shareable() ? new Float32Array(new SharedArrayBuffer(length * 4)) : new Float32Array(length);

export const sharedFloat64 = (length: number): Float64Array =>
  shareable() ? new Float64Array(new SharedArrayBuffer(length * 8)) : new Float64Array(length);

export const sharedUint32 = (length: number): Uint32Array =>
  shareable() ? new Uint32Array(new SharedArrayBuffer(length * 4)) : new Uint32Array(length);

export const sharedInt32 = (length: number): Int32Array =>
  shareable() ? new Int32Array(new SharedArrayBuffer(length * 4)) : new Int32Array(length);

export const sharedUint8 = (length: number): Uint8Array =>
  shareable() ? new Uint8Array(new SharedArrayBuffer(length)) : new Uint8Array(length);

type NumericArray = Float32Array | Float64Array | Uint32Array | Int32Array | Uint8Array;

/** Copies the values into a new array of the requested type, shared when possible. */
export function sharedCopy<T extends NumericArray>(create: (length: number) => T, values: ArrayLike<number>): T {
  const out = create(values.length);
  out.set(values);

  return out;
}
