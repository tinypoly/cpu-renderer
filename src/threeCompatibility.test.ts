import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { serializeCamera, cameraProjection } from "./camera.js";
import { serializeScene } from "./sceneSerialization.js";
import { sceneSignature } from "./sceneSignature.js";
import { CpuRasterizer } from "./rasterizer.js";
import { CpuEnvironment } from "./environment.js";
import { DEFAULT_FRAME_SETTINGS } from "./frameSettings.js";

const settings = { ...DEFAULT_FRAME_SETTINGS, maxSamples: 1, tileSize: 64,
  shadows: false, ambientOcclusion: false, backgroundMode: "transparent" as const, tonemapping: "linear" as const };

const ortho = () => new THREE.OrthographicCamera(-2, 2, 2, -2, .1, 20);

const geometry = (positions: number[]) => new THREE.BufferGeometry()
  .setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));

function pixels(renderer: CpuRasterizer) {
  for (const _ of renderer.prepare()) { /* drain */ }

  const job = renderer.renderBucket(renderer.buckets[0]);
  let step = job.next();
  while (!step.done) step = job.next();

  return step.value.pixels;
}

async function render(object: THREE.Object3D, camera: THREE.PerspectiveCamera | THREE.OrthographicCamera = ortho()) {
  const scene = new THREE.Scene();
  scene.add(object);
  const snapshot = (await serializeScene(scene)).scene;
  const renderer = new CpuRasterizer(snapshot, serializeCamera(camera), settings, 64, 64, new CpuEnvironment());

  return { renderer, image: pixels(renderer), scene };
}

const alpha = (image: Uint8ClampedArray, x: number, y: number) => image[(y * 64 + x) * 4 + 3];
const coverage = (image: Uint8ClampedArray) => image.filter((v, i) => i % 4 === 3 && v > 0).length;

describe("Three camera and primitive compatibility", () => {
  it("preserves zoom, asymmetric frusta and view offsets in camera snapshots", () => {
    for (const camera of [ortho(), new THREE.PerspectiveCamera(45, 2, .2, 100)]) {
      camera.zoom = 2;
      camera.setViewOffset(800, 600, 100, 50, 400, 300);
      const snapshot = serializeCamera(camera);
      expect(cameraProjection(snapshot, 1).toArray()).toEqual(camera.projectionMatrix.toArray());
    }

    const camera = ortho();
    camera.zoom = 2;
    const snapshot = serializeCamera(camera);
    delete snapshot.projectionMatrix;
    expect(cameraProjection(snapshot, 3).toArray()).toEqual(camera.projectionMatrix.toArray());
  });

  it("keeps orthographic objects the same size at different depths and honors zoom", async() => {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial());
    mesh.position.z = -2;
    const a = await render(mesh);
    mesh.position.z = -10;
    const b = await render(mesh);
    expect(a.image).toEqual(b.image);
    expect(coverage(a.image)).toBe(256);
    const camera = ortho();
    camera.zoom = 2;
    expect(coverage((await render(mesh, camera)).image)).toBe(1024);
  });

  it("renders connected, segmented and closed lines with the correct topology", async() => {
    const g = geometry([-1, -1, -3, -1, 1, -3, 1, 1, -3, 1, -1, -3]);
    const material = new THREE.LineBasicMaterial({ linewidth: 2 });
    const line = (await render(new THREE.Line(g, material))).image;
    const segments = (await render(new THREE.LineSegments(g, material))).image;
    const loop = (await render(new THREE.LineLoop(g, material))).image;
    expect(alpha(line, 32, 16)).toBe(255);
    expect(alpha(segments, 32, 16)).toBe(0);
    expect(alpha(line, 32, 48)).toBe(0);
    expect(alpha(loop, 32, 48)).toBe(255);
  });

  it("interpolates dash distance, vertex color and clips lines crossing the near plane", async() => {
    const line = new THREE.Line(geometry([-1, 0, -3, 1, 0, -3]),
      new THREE.LineDashedMaterial({ dashSize: .5, gapSize: .5, linewidth: 2 }));

    line.computeLineDistances();
    const dashed = (await render(line)).image;
    expect(alpha(dashed, 18, 32)).toBe(255);
    expect(alpha(dashed, 28, 32)).toBe(0);

    const clipped = new THREE.Line(geometry([-1, 0, 1, 1, 0, -3]),
      new THREE.LineBasicMaterial({ linewidth: 3, vertexColors: true }));

    clipped.geometry.setAttribute("color", new THREE.Float32BufferAttribute([1, 0, 0, 0, 1, 0], 3));
    const { image } = await render(clipped, new THREE.PerspectiveCamera(60, 1, .1, 20));
    expect(coverage(image)).toBeGreaterThan(0);
    expect(image[(32 * 64 + 40) * 4 + 1]).toBeGreaterThan(0);
  });

  it("billboards sprites with center, rotation, scale and transparent maps", async() => {
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ color: 0xff0000 }));
    sprite.position.z = -3;
    sprite.scale.set(2, 1, 1);
    sprite.rotation.y = Math.PI / 2;
    const first = await render(sprite);
    expect(coverage(first.image)).toBe(512);
    const signature = sceneSignature(first.scene);
    sprite.center.set(0, 0);
    expect(sceneSignature(first.scene)).not.toBe(signature);
    sprite.material.rotation = Math.PI / 2;
    const rotated = (await render(sprite)).image;
    expect(alpha(rotated, 24, 16)).toBe(255);
    expect(alpha(rotated, 40, 32)).toBe(0);
    sprite.material.map = new THREE.DataTexture(new Uint8Array([255, 255, 255, 0]), 1, 1);
    expect(coverage((await render(sprite)).image)).toBe(0);
  });

  it("implements perspective sprite attenuation and native point size", async() => {
    const camera = new THREE.PerspectiveCamera(60, 1, .1, 20);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ sizeAttenuation: false }));
    sprite.scale.setScalar(.5);
    sprite.position.z = -2;
    const first = (await render(sprite, camera)).image;
    sprite.position.z = -6;
    expect((await render(sprite, camera)).image).toEqual(first);

    const points = new THREE.Points(geometry([0, 0, -2]),
      new THREE.PointsMaterial({ size: 8, sizeAttenuation: false }));

    expect(coverage((await render(points, camera)).image)).toBe(64);
    expect(coverage((await render(points)).image)).toBe(64);
    points.material.size = 1;
    points.material.sizeAttenuation = true;
    expect(coverage((await render(points, camera)).image)).toBe(256);
  });

  it("uses parallel orthographic background rays, including rotated cameras", () => {
    const camera = ortho();
    camera.rotation.y = .6;
    const environment = new CpuEnvironment();
    environment.sampleBackground = d => [(d[0] + 1) / 2, (d[1] + 1) / 2, (d[2] + 1) / 2];

    const renderer = new CpuRasterizer({ meshes: [], lights: [], textures: [] }, serializeCamera(camera),
      { ...settings, backgroundMode: "environment" }, 64, 64, environment);

    const image = pixels(renderer);
    expect(image.slice(0, 4)).toEqual(image.slice(-4));
    expect(image.slice(0, 4)).toEqual(image.slice(32 * 64 * 4, 32 * 64 * 4 + 4));
    expect(image[3]).toBe(255);
  });

  it("samples native point textures with generated point coordinates and alpha tests", async() => {
    const texture = new THREE.DataTexture(new Uint8Array([
      255, 0, 0, 255, 0, 255, 0, 0,
      0, 0, 255, 255, 255, 255, 255, 0,
    ]), 2, 2);

    const points = new THREE.Points(geometry([0, 0, -2]), new THREE.PointsMaterial({
      size: 16, sizeAttenuation: false, map: texture, alphaTest: .5,
    }));

    const { image } = await render(points);
    expect(coverage(image)).toBe(128);
    expect(alpha(image, 26, 26)).toBe(255);
    expect(alpha(image, 36, 26)).toBe(0);
  });

  it("clips orthographic geometry at near and far and billboards against rotated cameras", async() => {
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial());
    const camera = ortho();
    camera.position.set(3, 0, 0);
    camera.lookAt(0, 0, 0);
    expect(coverage((await render(sprite, camera)).image)).toBe(256);
    sprite.position.set(0, 0, -.05);
    expect(coverage((await render(sprite)).image)).toBe(0);
    sprite.position.z = -21;
    expect(coverage((await render(sprite)).image)).toBe(0);
  });

  it("honors indexed line draw ranges, occlusion and shared prepared geometry", async() => {
    const group = new THREE.Group();
    const g = geometry([-1, 0, -3, 1, 0, -3, 0, -1, -3, 0, 1, -3]);
    g.setIndex([2, 3, 0, 1]);
    g.setDrawRange(2, 2);
    group.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ linewidth: 2 })));
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xff0000 }));
    mesh.position.z = -2;
    group.add(mesh);
    const { renderer, image } = await render(group);
    expect(alpha(image, 32, 16)).toBe(0);
    expect(alpha(image, 18, 32)).toBe(255);
    expect(Array.from(image.slice((32 * 64 + 32) * 4, (32 * 64 + 32) * 4 + 4))).toEqual([255, 0, 0, 255]);

    const adopted = new CpuRasterizer(renderer.scene, renderer.camera, settings, 64, 64,
      new CpuEnvironment(), {}, renderer.preparedGeometry);

    expect(pixels(adopted)).toEqual(image);
  });
});
