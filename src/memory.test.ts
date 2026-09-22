import { afterEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { serializeScene, serializeTexture } from "./sceneSerialization.js";
import { sampleTexture, srgbToLinear } from "./texture.js";
import { GeometryWriter } from "./preparedGeometry.js";

afterEach(() => vi.unstubAllGlobals());

describe("CPU snapshot memory", () => {
  it.each([false, true])("keeps byte textures four times smaller with identical filtering (shared=%s)", async shared => {
    vi.stubGlobal("crossOriginIsolated", shared);
    const bytes = Uint8Array.from({ length: 16 * 16 * 4 }, (_, i) => (i * 37) % 256);

    for (const colorSpace of [THREE.SRGBColorSpace, THREE.NoColorSpace]) {
      const texture = new THREE.DataTexture(bytes, 16, 16, THREE.RGBAFormat, THREE.UnsignedByteType);
      texture.colorSpace = colorSpace;
      const compact = await serializeTexture(texture);
      const expanded = Float32Array.from(bytes, value => value / 255);
      if (colorSpace === THREE.SRGBColorSpace)
        for (let i = 0; i < expanded.length; i += 4)
          for (let c = 0; c < 3; c++) expanded[i + c] = srgbToLinear(expanded[i + c]);
      expect(compact.data.byteLength).toBe(expanded.byteLength / 4);
      expect(compact.data.buffer).not.toBe(bytes.buffer);
      expect(compact.data.buffer instanceof SharedArrayBuffer).toBe(shared);

      for (const nearest of [true, false])
        for (const wrap of [THREE.ClampToEdgeWrapping, THREE.RepeatWrapping, THREE.MirroredRepeatWrapping])
          for (const flipY of [true, false]) {
            const settings = { ...compact, nearest, wrapS: wrap, wrapT: wrap, flipY,
              matrix: [2, 0, 0, 0, 3, 0, .17, -.2, 1] };

            for (const uv of [[0, 0], [.13, .62], [.5, .5], [1, 1], [-.2, 1.7]])
              expect(sampleTexture(settings, uv)).toEqual(sampleTexture({ ...settings, data: expanded,
                colorSpace: THREE.LinearSRGBColorSpace }, uv));
          }
    }
  });

  it("preserves HDR range and missing channel defaults", async() => {
    const hdr = new THREE.DataTexture(new Uint16Array([THREE.DataUtils.toHalfFloat(8)]), 1, 1,
      THREE.RedFormat, THREE.HalfFloatType);

    const snapshot = await serializeTexture(hdr);
    expect(snapshot.data).toBeInstanceOf(Float32Array);
    expect(sampleTexture(snapshot, [.5, .5])).toEqual([8, 0, 0, 1]);
    const red = await serializeTexture(new THREE.DataTexture(new Uint8Array([128]), 1, 1, THREE.RedFormat));
    expect(Array.from(red.data)).toEqual([128, 0, 0, 255]);
    expect(sampleTexture(red, [.5, .5])).toEqual([Math.fround(128 / 255), 0, 0, 1]);
  });

  it("copies shared geometry and material once, without detaching editor buffers", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const geometry = new THREE.BoxGeometry(), material = new THREE.MeshStandardMaterial();
    const scene = new THREE.Scene();

    for (let i = 0; i < 20; i++) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.x = i;
      scene.add(mesh);
    }

    const snapshot = await serializeScene(scene), first = snapshot.scene.meshes[0];

    for (const mesh of snapshot.scene.meshes) {
      expect(mesh.attributes).toBe(first.attributes);
      expect(mesh.index).toBe(first.index);
      expect(mesh.materials[0]).toBe(first.materials[0]);
    }

    const uniqueBytes = snapshot.transfer.reduce<number>((sum, buffer) => sum + (buffer as ArrayBuffer).byteLength, 0);

    const expectedBytes = Object.values(first.attributes)
      .reduce((sum, a) => sum + a.data.byteLength, first.index!.byteLength);

    expect(uniqueBytes).toBe(expectedBytes);
    structuredClone(snapshot.scene, { transfer: snapshot.transfer });
    expect(geometry.attributes.position.array.byteLength).toBeGreaterThan(0);
    expect(geometry.index!.array.byteLength).toBeGreaterThan(0);
  });

  it("does not reuse one mesh's morphed positions for another mesh", async() => {
    const geometry = new THREE.PlaneGeometry();
    geometry.morphAttributes.position = [geometry.attributes.position.clone()];
    geometry.morphTargetsRelative = true;
    const a = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    const b = new THREE.Mesh(geometry, a.material);
    a.morphTargetInfluences![0] = 1;
    const scene = new THREE.Scene(); scene.add(a, b);
    const snapshot = (await serializeScene(scene)).scene;
    expect(snapshot.meshes[0].attributes.position.data[0]).toBe(geometry.attributes.position.getX(0) * 2);
    expect(snapshot.meshes[1].attributes.position.data[0]).toBe(geometry.attributes.position.getX(0));
    expect(snapshot.meshes[0].attributes.position).not.toBe(snapshot.meshes[1].attributes.position);
  });
  it("shares texture pixels across sampler clones while preserving UV transforms and updates", async() => {
    vi.stubGlobal("crossOriginIsolated", false);
    const source = new THREE.DataTexture(new Uint8Array([128, 64, 32, 255]), 1, 1);
    const other = source.clone(); other.repeat.set(2, 3); other.flipY = true;
    const scene = new THREE.Scene(), geometry = new THREE.PlaneGeometry();
    scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: source })),
      new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: other })));
    const first = await serializeScene(scene), [a, b] = first.scene.textures;
    expect(a.id).not.toBe(b.id);
    expect(a.data).toBe(b.data);
    expect(a.matrix).not.toEqual(b.matrix);
    expect(b.flipY).toBe(true);
    expect(first.transfer.filter(buffer => buffer === a.data.buffer)).toHaveLength(1);
    source.image.data![0] = 255;
    const next = await serializeScene(scene);
    expect(next.scene.textures[0].data[0]).toBe(255);
    expect(a.data[0]).toBe(128);
  });

  it("bounds canvas readbacks to strips and releases the backing canvas", async() => {
    const canvases: FakeCanvas[] = [];
    const reads: number[] = [], starts: number[] = [];

    class FakeCanvas {
      row = 0;

      constructor(public width: number, public height: number) {
        canvases.push(this);
      }

      getContext() {
        return { clearRect: () => {},
          drawImage: (_image: unknown, _x: number, y: number) => {
            this.row = y; starts.push(y);
          },
          getImageData: (_x: number, _y: number, width: number, height: number) => {
            reads.push(height);

            return { data: Uint8ClampedArray.from({ length: width * height * 4 },
              (_, i) => (this.row + Math.floor(i / (width * 4))) % 256) };
          } };
      }
    }

    vi.stubGlobal("OffscreenCanvas", FakeCanvas);
    const texture = new THREE.Texture({ width: 2, height: 130 } as TexImageSource);
    const result = await serializeTexture(texture);
    expect(reads).toEqual([64, 64, 2]);
    expect(starts).toEqual([0, 64, 128]);
    expect(result.data[129 * 2 * 4]).toBe(129);
    expect(canvases[0].width * canvases[0].height).toBe(0);
  });

  it("releases geometry staging buffers when publishing prepared geometry", () => {
    const writer = new GeometryWriter();
    const a = writer.allocateVertex(11), b = writer.allocateVertex(11), c = writer.allocateVertex(11);
    writer.vertices.set([1, 0, 0], b + 4); writer.vertices.set([0, 1, 0], c + 4);
    writer.addTriangle(0, a, b, c, 0);
    const bins = [[0]], result = writer.finish(bins, true);
    expect(writer.vertices.byteLength).toBe(0);
    expect(bins).toEqual([[]]);
    expect(Array.from(result.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(Array.from(result.binTriangles)).toEqual([0]);
  });
});
