import { Worker } from "node:worker_threads";
import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { CpuRenderer, type RenderPixels } from "./renderer.js";
import { wrapNodeWorker } from "./node.js";
import { shareable } from "./sharedBuffers.js";

// The worker entry is TypeScript here, so the thread registers tsx before loading it; `createNodeWorker` loads the
// built file directly.
const bootstrap = `import { register } from ${JSON.stringify(import.meta.resolve("tsx/esm/api"))};
register();
await import(${JSON.stringify(new URL("./workerNode.ts", import.meta.url).href)});`;

const createWorker = () => wrapNodeWorker(new Worker(new URL(`data:text/javascript,${encodeURIComponent(bootstrap)}`)));

describe("CpuRenderer in Node", () => {
  it("renders a scene on worker threads sharing memory, into image and the pixels event", async() => {
    expect(shareable()).toBe(true);
    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.MeshBasicMaterial({ color: "#ff0000" })));
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(0, 0, 2);
    camera.lookAt(0, 0, 0);

    const renderer = new CpuRenderer({
      width: 12, height: 8, pixelRatio: 2, scene, camera, workers: 2, createWorker,
      settings: { samples: 1, tileSize: 8, ambientOcclusion: { enabled: false } },
      environment: { kind: "gradient", topColor: "#000000", bottomColor: "#000000" },
    });

    expect(renderer.workerCount).toBe(2);
    const painted: RenderPixels[] = [];
    renderer.on("pixels", pixels => painted.push(pixels));

    try {
      const image = await renderer.render();
      expect(image).toBe(renderer.image);
      expect([image.width, image.height]).toEqual([24, 16]);

      // A red plane fills the view: every pixel is red (after tone mapping) and opaque, and every one arrived
      // through the event.
      for (let i = 0; i < image.width * image.height; i++) {
        const [r, g, b, a] = image.data.subarray(i * 4, i * 4 + 4);
        expect(r).toBeGreaterThan(200);
        expect(Math.max(g, b)).toBeLessThan(60);
        expect(a).toBe(255);
      }

      const covered = new Set<number>();
      for (const { x, y, width, height } of painted)
        for (let j = 0; j < height; j++)
          for (let i = 0; i < width; i++)
            covered.add((y + j) * image.width + x + i);
      expect(covered.size).toBe(image.width * image.height);
    } finally {
      renderer.dispose();
    }
  }, 60000);
});
