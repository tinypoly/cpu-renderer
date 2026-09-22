import type { CpuRenderer } from "./renderer.js";

/**
 * Paints the renderer's output on a Canvas 2D: the canvas takes the render size at `start` and each `pixels` event
 * lands at its position. Attached after a frame started, it paints what `image` already holds. Returns the function
 * that detaches it; the canvas keeps its last contents.
 */
export function attachCanvas(renderer: CpuRenderer, canvas: HTMLCanvasElement): () => void {
  const context = canvas.getContext("2d");
  if (!context)
    throw new Error("Canvas 2D is unavailable.");

  const resize = ({ width, height }: { width: number; height: number }) => {
    canvas.width = width;
    canvas.height = height;
    context.clearRect(0, 0, width, height);
  };

  const paint = ({ x, y, width, height, data }: { x: number; y: number; width: number; height: number;
    data: Uint8ClampedArray<ArrayBuffer> }) => context.putImageData(new ImageData(data, width, height), x, y);

  const image = renderer.image;

  if (image) {
    resize(image);
    paint({ x: 0, y: 0, ...image });
  }

  const detach = [renderer.on("start", resize), renderer.on("pixels", paint)];

  return () => detach.forEach(off => off());
}
