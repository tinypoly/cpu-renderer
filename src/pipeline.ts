import { cameraProjection } from "./camera.js";
import { Matrix3, Matrix4, Plane, Vector3 } from "three";
import type { SerializedMaterial, SerializedScene, SerializedTexture } from "./sceneSerialization.js";
import type { CameraState } from "./settings.js";
import type { FrameSettings } from "./frameSettings.js";
import { CpuComputePipeline, floatTexture } from "./compute.js";
import type { CpuEnvironment } from "./environment.js";
import { CpuRasterizer } from "./rasterizer.js";

/** A completed bake survives camera/quality changes. An interrupted bake never enters the cache. */
export class CpuScenePipeline {
  private computed = new Map<string, Record<string, SerializedTexture>>();

  constructor(readonly source: SerializedScene) {}

  *prepare(camera: CameraState, settings: FrameSettings, width: number, height: number,
    environment: CpuEnvironment): Generator<void, SerializedScene> {
    const materials = new Map<SerializedMaterial, SerializedMaterial>();

    const scene: SerializedScene = {
      ...this.source, textures: [...this.source.textures], meshes: this.source.meshes.map(mesh => ({
        ...mesh, materials: mesh.materials.map(original => {
          let material = materials.get(original);

          if (!material) {
            material = { ...original,
              shader: original.shader ? { ...original.shader, uniforms: { ...original.shader.uniforms } } : undefined,
            };
            materials.set(original, material);
          }

          return material;
        }),
      })) };

    const textureIds = new Set(scene.textures.map(t => t.id));

    for (const material of materials.values()) {
      if (!material.shader) continue;
      const { compute } = material;

      if (compute) {
        let outputs = this.computed.get(compute.id);

        if (!outputs) {
          const pipeline = new CpuComputePipeline(compute.pipeline, compute.id, compute.data ?? {});
          if (compute.snapshot) pipeline.restore(compute.snapshot);
          else yield* pipeline.tick(material.time ?? 0, 1 / 60);
          outputs = pipeline.outputs();
          this.computed.set(compute.id, outputs);
        }

        for (const [uniform, texture] of Object.entries(outputs)) {
          material.shader.uniforms[uniform] = { texture: texture.id };

          if (!textureIds.has(texture.id)) {
            scene.textures.push(texture); textureIds.add(texture.id);
          }
        }
      }

      if (material.time !== undefined)
        material.shader.uniforms.time = material.time;
    }

    // Planar surfaces are excluded from secondary captures to avoid recursive dependencies.
    const captureScene = { ...scene, meshes: scene.meshes.filter(mesh =>
      !mesh.materials.some(m => m.planarCapture)) };

    for (const mesh of scene.meshes) {
      if (mesh.primitive) continue;

      for (const material of mesh.materials) {
        if (!material.planarCapture || !material.shader) continue;
        const id = `capture:${material.planarCapture.id}`;
        const model = new Matrix4().fromArray(mesh.matrixWorld);
        const normal = new Vector3(0, 1, 0).applyMatrix3(new Matrix3().getNormalMatrix(model)).normalize();
        const point = new Vector3().setFromMatrixPosition(model);
        const plane = new Plane().setFromNormalAndCoplanarPoint(normal, point);
        const { x, y, z } = normal, d = plane.constant;

        const mirror = new Matrix4().set(1 - 2 * x * x, -2 * x * y, -2 * x * z, -2 * d * x,
          -2 * y * x, 1 - 2 * y * y, -2 * y * z, -2 * d * y,
          -2 * z * x, -2 * z * y, 1 - 2 * z * z, -2 * d * z, 0, 0, 0, 1);

        const reflectedCamera = { ...camera,
          matrixWorld: mirror.multiply(new Matrix4().fromArray(camera.matrixWorld)).toArray(),
        };

        const side = plane.distanceToPoint(new Vector3().fromArray(camera.matrixWorld, 12)) >= 0 ? 1 : -1;
        const projection = cameraProjection(camera, width / height);
        const bias = new Matrix4().set(.5, 0, 0, .5, 0, .5, 0, .5, 0, 0, .5, .5, 0, 0, 0, 1);

        const textureMatrix = (world: number[]) => bias.clone().multiply(projection)
          .multiply(new Matrix4().fromArray(world).invert()).multiply(model).toArray();

        Object.assign(material.shader.uniforms, {
          reflectionMatrix: textureMatrix(reflectedCamera.matrixWorld),
          refractionMatrix: textureMatrix(camera.matrixWorld),
          cameraNear: camera.near, cameraFar: camera.far,
        });
        const captureScale = Math.min(1, 1024 / Math.max(width, height));
        const captureWidth = Math.max(1, Math.round(width * captureScale));
        const captureHeight = Math.max(1, Math.round(height * captureScale));
        const captureSettings = { ...settings, dofEnabled: false, maxSamples: 1 };

        const captures = [
          { name: "tReflectionMap", camera: reflectedCamera, scene: captureScene, background: "transparent" as const,
            clip: [x * side, y * side, z * side, d * side + .01] },
          { name: "tEnvironmentReflectionMap", camera: reflectedCamera, scene: { ...captureScene, meshes: [] },
            background: "environment" as const },
          { name: "tRefractionMap", camera, scene: captureScene, background: "environment" as const,
            clip: [-x * side, -y * side, -z * side, -d * side + .01] },
        ];

        for (const capture of captures) {
          const renderer = new CpuRasterizer(capture.scene, capture.camera,
            { ...captureSettings, backgroundMode: capture.background },
            captureWidth, captureHeight, environment,
            { linear: true, clipPlane: capture.clip, aspect: width / height });

          const texture = floatTexture(`${id}:${capture.name}`, captureWidth, captureHeight);
          texture.flipY = true;
          texture.nearest = false;

          const depth = capture.name === "tRefractionMap"
            ? floatTexture(`${id}:tRefractionDepth`, captureWidth, captureHeight) : null;

          if (depth) {
            depth.flipY = true; depth.data.fill(1);
          }

          yield* renderer.prepare();

          for (const bucket of renderer.buckets) {
            const result = yield* renderer.renderBucket(bucket);

            for (let y = 0; y < bucket.height; y++) {
              texture.data.set(result.linear!.subarray(y * bucket.width * 4, (y + 1) * bucket.width * 4),
                ((bucket.y + y) * captureWidth + bucket.x) * 4);

              if (depth) for (let x = 0; x < bucket.width; x++) {
                const value = result.depth![y * bucket.width + x];
                depth.data.set([value, value, value, 1], ((bucket.y + y) * captureWidth + bucket.x + x) * 4);
              }
            }

            yield;
          }

          scene.textures.push(texture);
          material.shader.uniforms[capture.name] = { texture: texture.id };

          if (depth) {
            scene.textures.push(depth);
            material.shader.uniforms.tRefractionDepth = { texture: depth.id };
          }
        }
      }
    }

    return scene;
  }
}
