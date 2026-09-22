/**
 * Benchmark of the CPU rasterizer, single-threaded, in Node.
 *
 * It provides a baseline for renderer optimizations: each case reports preparation and shading
 * time separately, plus a signature of the final image. The signature is what guarantees that an
 * optimization did not change the result: if it changes, the optimization altered the order of
 * floating point operations and must be revisited.
 *
 * Usage:
 *   pnpm bench
 *   pnpm bench --case=gi --runs=5
 *   pnpm bench --json
 *
 * How to read the numbers without fooling yourself, since each of the three pitfalls below has
 * already produced false results:
 *
 * 1. Wall-clock time fluctuates by more than 10% with the browser open. That is why this measures
 *    the process CPU time, which ignores contention for scheduling.
 * 2. Short cases need warm-up. With few runs the JIT has not settled yet and the reading comes
 *    out high: use `--runs=12` for the ~150 ms cases and 2 or 3 for the GI one.
 * 3. CPU time still goes up when the machine throttles its clock due to heat. Right after running
 *    the test suite the reading is inflated. Run again and keep the lowest value.
 *
 * Water is left out: it depends on module-environment and its preparation in the interpreter
 * takes minutes, which is useless for measuring every iteration. The "glsl" case covers the same
 * interpreter.
 */
import * as THREE from "three";
import { CpuRasterizer } from "../src/rasterizer.js";
import { CpuEnvironment } from "../src/environment.js";
import { serializeScene } from "../src/sceneSerialization.js";
import type { FrameSettings } from "../src/frameSettings.js";
import type { BvhData } from "../src/bvh.js";

const baseSettings: FrameSettings = {
  renderScale: 1,
  maxSamples: 4,
  tileSize: 64,
  shadows: true,
  environmentIntensity: 1,
  environmentRotation: 0,
  backgroundMode: "environment",
  backgroundColor: "#000000",
  dofEnabled: false,
  dofFocusDistance: 3,
  dofAperture: 2.8,
  bokehBlades: 0,
  tonemapping: "aces",
  exposure: 1,
};

interface BenchCase {
  name: string;
  description: string;
  width: number;
  height: number;
  settings: Partial<FrameSettings>;
  build: () => THREE.Scene;
  camera: () => THREE.PerspectiveCamera;
}

/** Default camera for the cases, looking straight at the origin. */
function defaultCamera(width: number, height: number, distance = 6) {
  const camera = new THREE.PerspectiveCamera(50, width / height, .1, 100);
  camera.position.set(0, 1.2, distance);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  return camera;
}

/** Procedural checkerboard texture, without a canvas, for the textured case. */
function checkerTexture(size = 64) {
  const data = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const on = ((x >> 3) + (y >> 3)) % 2 === 0;
      const offset = (y * size + x) * 4;
      data[offset] = on ? 230 : 40;
      data[offset + 1] = on ? 120 : 40;
      data[offset + 2] = on ? 60 : 200;
      data[offset + 3] = 255;
    }

  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.needsUpdate = true;

  return texture;
}

/** Six Physical spheres on a floor, lit by a directional light. Base for the native cases. */
function sphereScene(textured: boolean) {
  const scene = new THREE.Scene();
  const geometry = new THREE.SphereGeometry(.8, 24, 16);
  const map = textured ? checkerTexture() : undefined;

  for (let i = 0; i < 6; i++) {
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color().setHSL(i / 6, .6, .5),
      roughness: .15 + i * .14,
      metalness: i % 2 === 0 ? .9 : .1,
      clearcoat: i % 3 === 0 ? 1 : 0,
      ...(map ? { map } : {}),
    });

    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.set((i % 3 - 1) * 2, i < 3 ? .9 : 2.6, i < 3 ? 0 : -1.5);
    sphere.castShadow = sphere.receiveShadow = true;
    scene.add(sphere);
  }

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshStandardMaterial({ color: 0xbbbbbb, roughness: .8 }),
  );

  floor.rotation.x = -Math.PI / 2;
  floor.castShadow = floor.receiveShadow = true;
  scene.add(floor);
  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.castShadow = true;
  light.position.set(3, 6, 4);
  scene.add(light, new THREE.AmbientLight(0xffffff, .2));

  return scene;
}

/** Closed box with an emissive panel on the ceiling: exercises the indirect lighting paths. */
function cornellScene() {
  const scene = new THREE.Scene();

  const wall = (color: number, position: [number, number, number], rotation: [number, number, number]) => {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(6, 6),
      new THREE.MeshStandardMaterial({ color, roughness: 1, side: THREE.DoubleSide }),
    );

    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    mesh.castShadow = mesh.receiveShadow = true;

    return mesh;
  };

  scene.add(
    wall(0xcccccc, [0, 0, -3], [0, 0, 0]),
    wall(0xcccccc, [0, -3, 0], [-Math.PI / 2, 0, 0]),
    wall(0xcccccc, [0, 3, 0], [Math.PI / 2, 0, 0]),
    wall(0xcc3333, [-3, 0, 0], [0, Math.PI / 2, 0]),
    wall(0x33cc33, [3, 0, 0], [0, -Math.PI / 2, 0]),
  );

  const panel = new THREE.Mesh(
    new THREE.PlaneGeometry(2.4, 2.4),
    new THREE.MeshStandardMaterial({
      color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 6, side: THREE.DoubleSide,
    }),
  );

  panel.position.set(0, 2.9, 0);
  panel.rotation.x = Math.PI / 2;

  const box = new THREE.Mesh(
    new THREE.BoxGeometry(1.4, 2.6, 1.4),
    new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: .9 }),
  );

  box.position.set(-1, -1.7, -.8);
  box.rotation.y = .4;

  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(.9, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: .6 }),
  );

  ball.position.set(1.2, -2.1, .4);
  for (const mesh of [panel, box, ball])
    mesh.castShadow = mesh.receiveShadow = true;
  scene.add(panel, box, ball);

  return scene;
}

/** Many triangles: the tree no longer fits in cache and the cost per visited node starts to matter. */
function heavyScene() {
  const scene = new THREE.Scene();
  const knot = new THREE.TorusKnotGeometry(.9, .3, 256, 32);

  for (let i = 0; i < 9; i++) {
    const mesh = new THREE.Mesh(knot, new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(i / 9, .5, .5), roughness: .5, metalness: .2,
    }));

    mesh.position.set((i % 3 - 1) * 3, 1.2, Math.floor(i / 3) * -3 + 3);
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: 0xaaaaaa, roughness: .9 }),
  );

  floor.rotation.x = -Math.PI / 2;
  floor.castShadow = floor.receiveShadow = true;
  scene.add(floor);
  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.castShadow = true;
  light.position.set(4, 8, 5);
  scene.add(light, new THREE.AmbientLight(0xffffff, .25));

  return scene;
}

/** Custom GLSL material with per-fragment vector arithmetic: measures the interpreter. */
function glslScene() {
  const scene = new THREE.Scene();

  const material = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec2 vUv;
      varying vec3 vWorld;
      void main() {
        vUv = uv;
        vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      varying vec2 vUv;
      varying vec3 vWorld;
      uniform float uTime;
      void main() {
        vec3 color = vec3(0.0);
        vec2 p = vUv * 6.0 - 3.0;
        for (int i = 0; i < 6; i++) {
          float fi = float(i);
          vec2 q = p + vec2(sin(uTime + fi), cos(uTime * 0.7 + fi));
          float d = length(q) + 0.001;
          color += vec3(0.6, 0.3, 0.9) * (0.35 / d);
          p = p * 1.15 + vec2(0.1, -0.05);
        }
        vec3 normal = normalize(vWorld);
        color *= 0.5 + 0.5 * max(dot(normal, vec3(0.0, 1.0, 0.0)), 0.0);
        gl_FragColor = vec4(color, 1.0);
      }`,
    uniforms: { uTime: { value: 1.25 } },
  });

  const mesh = new THREE.Mesh(new THREE.SphereGeometry(2.2, 48, 32), material);
  scene.add(mesh);

  return scene;
}

const cases: BenchCase[] = [
  {
    name: "spheres",
    description: "six Physical spheres with hard shadows",
    width: 480, height: 320,
    settings: {},
    build: () => sphereScene(false),
    camera: () => defaultCamera(480, 320),
  },
  {
    name: "textured",
    description: "the same spheres with a color map",
    width: 480, height: 320,
    settings: {},
    build: () => sphereScene(true),
    camera: () => defaultCamera(480, 320),
  },
  {
    name: "aoShadows",
    description: "ambient occlusion and soft shadows, 12 and 8 rays per fragment",
    width: 320, height: 240,
    settings: { ambientOcclusion: true, aoDistance: 2, aoIntensity: 1, shadowSoftness: 3 },
    build: () => sphereScene(false),
    camera: () => defaultCamera(320, 240),
  },
  {
    name: "gi",
    description: "diffuse global illumination, 16 samples and 3 bounces",
    width: 320, height: 213,
    settings: { globalIllumination: true, giSamples: 16, giBounces: 3, giIntensity: 1 },
    build: cornellScene,
    camera: () => {
      const camera = new THREE.PerspectiveCamera(50, 320 / 213, .1, 100);
      camera.position.set(0, 0, 7.5);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();

      return camera;
    },
  },
  {
    name: "heavy",
    description: "nine tessellated torus knots with shadows and occlusion",
    width: 320, height: 240,
    settings: { ambientOcclusion: true, aoDistance: 2, aoIntensity: 1, shadowSoftness: 2 },
    build: heavyScene,
    camera: () => {
      const camera = new THREE.PerspectiveCamera(50, 320 / 240, .1, 100);
      camera.position.set(0, 4, 11);
      camera.lookAt(0, .8, 0);
      camera.updateMatrixWorld();

      return camera;
    },
  },
  {
    name: "glsl",
    description: "custom GLSL material in the interpreter",
    width: 320, height: 240,
    settings: {},
    build: glslScene,
    camera: () => defaultCamera(320, 240, 7),
  },
];

/**
 * SAH cost of the tree, in the form of equation 1 from Karras and Aila (2013), the metric used in the
 * literature to compare BVH builders. Lower is better. The constants are the ones from the paper.
 */
function bvhReport(geometry: { bvh: BvhData }) {
  const { bounds, nodes } = geometry.bvh;
  const Ci = 1.2, Ct = 1;
  const count = nodes.length / 2;
  if (!count)
    return { sah: 0, nodes: 0, leaves: 0, triangles: 0, maxLeaf: 0, depth: 0 };

  const area = (n: number) => {
    const x = bounds[n * 6 + 3] - bounds[n * 6], y = bounds[n * 6 + 4] - bounds[n * 6 + 1];
    const z = bounds[n * 6 + 5] - bounds[n * 6 + 2];

    return 2 * (x * y + y * z + z * x);
  };

  const root = area(0);
  let internal = 0, leaves = 0, triangles = 0, maxLeaf = 0, depth = 0, sah = 0;

  const walk = (node: number, level: number) => {
    depth = Math.max(depth, level);
    const first = nodes[node * 2], second = nodes[node * 2 + 1];

    if (first >= 0) {
      internal++;
      sah += Ci * area(node) / root;
      walk(first, level + 1);
      walk(second, level + 1);

      return;
    }

    leaves++;
    triangles += second;
    maxLeaf = Math.max(maxLeaf, second);
    sah += Ct * area(node) / root * second;
  };

  walk(0, 1);

  return { sah, nodes: internal, leaves, triangles, maxLeaf, depth };
}

/** 32-bit FNV-1a over the image bytes: changes on the slightest deviation in the result. */
function checksum(pixels: Uint8ClampedArray) {
  let hash = 0x811c9dc5;

  for (let i = 0; i < pixels.length; i++) {
    hash ^= pixels[i];
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }

  return hash.toString(16).padStart(8, "0");
}

function drain<T>(job: Generator<void, T>): T {
  let step = job.next();
  while (!step.done)
    step = job.next();

  return step.value;
}

interface RunResult {
  prepare: number;
  shade: number;
  total: number;
  signature: string;
}

/** CPU time of this process, in milliseconds: immune to unrelated load on the machine. */
function cpuMillis() {
  const usage = process.cpuUsage();

  return (usage.user + usage.system) / 1000;
}

function runOnce(benchCase: BenchCase, serialized: Awaited<ReturnType<typeof serializeScene>>["scene"],
  camera: THREE.PerspectiveCamera): RunResult {
  const settings = { ...baseSettings, ...benchCase.settings };

  const renderer = new CpuRasterizer(serialized, {
    matrixWorld: camera.matrixWorld.toArray(),
    fov: camera.fov,
    near: camera.near,
    far: camera.far,
  }, settings, benchCase.width, benchCase.height, new CpuEnvironment());

  const prepareStart = cpuMillis();
  drain(renderer.prepare());
  const prepare = cpuMillis() - prepareStart;

  const image = new Uint8ClampedArray(benchCase.width * benchCase.height * 4);
  const shadeStart = cpuMillis();

  for (const bucket of renderer.buckets) {
    const result = drain(renderer.renderBucket(bucket));
    for (let y = 0; y < bucket.height; y++)
      image.set(
        result.pixels.subarray(y * bucket.width * 4, (y + 1) * bucket.width * 4),
        ((bucket.y + y) * benchCase.width + bucket.x) * 4,
      );
  }

  const shade = cpuMillis() - shadeStart;

  return { prepare, shade, total: prepare + shade, signature: checksum(image) };
}

function parseArgs() {
  const args = process.argv.slice(2);

  const value = (name: string) => {
    const found = args.find(a => a.startsWith(`--${name}=`));

    return found ? found.slice(name.length + 3) : undefined;
  };

  return {
    filter: value("case"),
    runs: Number(value("runs") ?? 3),
    json: args.includes("--json"),
    bvh: args.includes("--bvh"),
  };
}

async function main() {
  const { filter, runs, json, bvh } = parseArgs();
  const selected = filter ? cases.filter(c => c.name === filter) : cases;

  if (!selected.length) {
    console.error(`Unknown case: ${filter}. Available: ${cases.map(c => c.name).join(", ")}`);
    process.exitCode = 1;

    return;
  }

  if (!Number.isInteger(runs) || runs < 1) {
    console.error("The number of runs must be an integer greater than zero.");
    process.exitCode = 1;

    return;
  }

  const report: Record<string, unknown>[] = [];

  for (const benchCase of selected) {
    const scene = benchCase.build();
    const camera = benchCase.camera();
    const { scene: serialized } = await serializeScene(scene);

    if (bvh) {
      const settings = { ...baseSettings, ...benchCase.settings };

      const renderer = new CpuRasterizer(serialized, {
        matrixWorld: camera.matrixWorld.toArray(), fov: camera.fov, near: camera.near, far: camera.far,
      }, settings, benchCase.width, benchCase.height, new CpuEnvironment());

      drain(renderer.prepare());
      const report = bvhReport(renderer.preparedGeometry);
      console.log(
        `${benchCase.name.padEnd(10)} SAH ${report.sah.toFixed(2).padStart(8)}  `
        + `internal ${String(report.nodes).padStart(6)}  leaves ${String(report.leaves).padStart(6)}  `
        + `triangles ${String(report.triangles).padStart(6)}  max leaf ${report.maxLeaf}  `
        + `depth ${report.depth}`,
      );
      continue;
    }

    const results: RunResult[] = [];
    for (let run = 0; run < runs; run++)
      results.push(runOnce(benchCase, serialized, camera));

    // The lowest CPU time is the most stable one: machine noise only ever adds to it.
    const best = results.reduce((a, b) => b.total < a.total ? b : a);
    const signatures = new Set(results.map(r => r.signature));

    const row = {
      case: benchCase.name,
      description: benchCase.description,
      resolution: `${benchCase.width}x${benchCase.height}`,
      prepareMs: Math.round(best.prepare),
      shadeMs: Math.round(best.shade),
      totalMs: Math.round(best.total),
      signature: best.signature,
      stable: signatures.size === 1,
    };

    report.push(row);
    if (!json)
      console.log(
        `${row.case.padEnd(10)} ${row.resolution.padEnd(9)} `
        + `prepare ${String(row.prepareMs).padStart(6)} ms  `
        + `shade ${String(row.shadeMs).padStart(7)} ms  `
        + `total ${String(row.totalMs).padStart(7)} ms  `
        + `signature ${row.signature}${row.stable ? "" : "  UNSTABLE"}`,
      );
  }

  if (json)
    console.log(JSON.stringify(report, null, 2));
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
