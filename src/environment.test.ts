import { afterEach, describe, expect, it, vi } from "vitest";
import { CpuEnvironment } from "./environment.js";

/** A 2 × 1 Radiance HDR image, uncompressed (runs only apply from 8 pixels wide). */
function hdr(pixels: number[][]): ArrayBuffer {
  const header = new TextEncoder().encode("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n");
  const body = Uint8Array.from(pixels.flat());
  const bytes = new Uint8Array(header.length + body.length);
  bytes.set(header);
  bytes.set(body, header.length);

  return bytes.buffer;
}

afterEach(() => vi.unstubAllGlobals());

describe("CpuEnvironment.load", () => {
  it("decodes an equirectangular HDR image into linear floats", async() => {
    const pixels = [[128, 64, 32, 129], [128, 128, 128, 128]];
    const body = hdr(pixels);
    const fetch = vi.fn(async() => new Response(body));
    vi.stubGlobal("fetch", fetch);
    const signal = new AbortController().signal;
    const environment = await CpuEnvironment.load({ kind: "hdri", url: "studio.hdr" }, signal);
    expect(fetch).toHaveBeenCalledWith("studio.hdr", { signal });
    const { texture } = environment.snapshot();
    expect([texture!.width, texture!.height, texture!.colorSpace]).toEqual([2, 1, "srgb-linear"]);
    // Three's RGBE decoding: mantissa / 255 × 2^(exponent − 128), with alpha 1.
    const expected = pixels.flatMap(([r, g, b, e]) => [...[r, g, b].map(m => m / 255 * 2 ** (e - 128)), 1]);
    Array.from(texture!.data).forEach((value, i) => expect(value).toBeCloseTo(expected[i], 6));
  });

  it("rejects an HTTP error and a body that is not an HDR image", async() => {
    vi.stubGlobal("fetch", vi.fn(async() => new Response(null, { status: 403 })));
    await expect(CpuEnvironment.load({ kind: "hdri", url: "private.hdr" })).rejects.toThrow("HDRI: HTTP 403");
    vi.stubGlobal("fetch", vi.fn(async() => new Response("<html></html>")));
    await expect(CpuEnvironment.load({ kind: "hdri", url: "page.html" })).rejects.toThrow();
    await expect(CpuEnvironment.load({ kind: "hdri", url: "" })).rejects.toThrow("HDRI URL is missing.");
  });

  it("stops a download when its signal aborts", async() => {
    vi.stubGlobal("fetch", vi.fn((_url: string, { signal }: { signal: AbortSignal }) => new Promise((_, reject) =>
      signal.addEventListener("abort", () => reject(signal.reason)))));
    const controller = new AbortController();
    const loading = CpuEnvironment.load({ kind: "hdri", url: "slow.hdr" }, controller.signal);
    controller.abort();
    await expect(loading).rejects.toMatchObject({ name: "AbortError" });
  });

  it("builds a gradient with white over black and exponent 2 by default", async() => {
    const snapshot = (await CpuEnvironment.load({ kind: "gradient" })).snapshot();
    expect([snapshot.top, snapshot.bottom, snapshot.exponent, snapshot.texture])
      .toEqual([[1, 1, 1], [0, 0, 0], 2, null]);
  });
});
