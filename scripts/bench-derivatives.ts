/** Small reproducible comparison of scalar shading and fragment derivatives (no GI or texture IO). */
import * as THREE from "three";
import { CpuRasterizer } from "../src/rasterizer.js";
import { CpuEnvironment } from "../src/environment.js";
import { serializeScene } from "../src/sceneSerialization.js";
import { serializeCamera } from "../src/camera.js";
import { DEFAULT_FRAME_SETTINGS } from "../src/frameSettings.js";

const size = 64;
const results: Record<string, number> = {};

for (const [name, expression] of Object.entries({
  scalar: "abs(sin(vUv.x*20.0))+abs(cos(vUv.y*20.0))",
  derivatives: "fwidth(sin(vUv.x*20.0)+cos(vUv.y*20.0))",
})) {
  const material = new THREE.ShaderMaterial({
    vertexShader: `varying vec2 vUv;
void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
    fragmentShader: `varying vec2 vUv;void main(){gl_FragColor=vec4(vec3(${expression}),1.0);}`,
  });

  const scene = new THREE.Scene();
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  mesh.position.z = -2;
  scene.add(mesh);
  const serialized = (await serializeScene(scene)).scene;

  const renderer = new CpuRasterizer(serialized,
    serializeCamera(new THREE.OrthographicCamera(-1, 1, 1, -1, .1, 10)),
    { ...DEFAULT_FRAME_SETTINGS, maxSamples: 1, tileSize: 32,
      ambientOcclusion: false, shadows: false, backgroundMode: "transparent" }, size, size, new CpuEnvironment());

  for (const _ of renderer.prepare()) { /* drain */ }

  const elapsed: number[] = [];

  for (let run = 0; run < 4; run++) {
    const start = performance.now();

    for (const bucket of renderer.buckets)
      for (const _ of renderer.renderBucket(bucket)) { /* drain */ }

    if (run) elapsed.push(performance.now() - start);
  }

  results[name] = elapsed.sort((a, b) => a - b)[1];
  console.log(`${name}: ${results[name].toFixed(1)} ms (median of 3, ${size}x${size})`);
}

console.log(`Derivative/scalar ratio: ${(results.derivatives / results.scalar).toFixed(2)}x`);
