import * as THREE from "three";
import { cpuRendererData, type CpuRendererLightData, type CpuRendererMaterialData, type CpuRendererSceneData,
  type MaterialCompute, type ProceduralSky } from "./extensions.js";
import { sharedFloat32, sharedUint32, sharedUint8 } from "./sharedBuffers.js";
import { generateMipmaps, srgbToLinear } from "./texture.js";
import type { VolumetricMedium } from "./volumetricLight.js";

export type UniformValue = number | boolean | string | null | UniformValue[] | {
  [key: string]: UniformValue;
};

export interface SerializedAttribute {
  data: Float32Array;
  itemSize: number;
}

export interface SerializedMipLevel {
  width: number;
  height: number;
  data: Float32Array | Uint8Array;
}

export interface SerializedTexture {
  id: string;
  width: number;
  height: number;
  /** Byte textures keep their original encoding; float buffers contain linear RGBA. */
  data: Float32Array | Uint8Array;
  colorSpace: string;
  wrapS: number;
  wrapT: number;
  matrix: number[];
  flipY: boolean;
  channel: number;
  nearest: boolean;
  minFilter?: number;
  magFilter?: number;
  /** Levels 1..N; level zero remains in width/height/data. All levels use colorSpace. */
  mipmaps?: SerializedMipLevel[];
  /** Maximum taps along the footprint's major axis; 1 or undefined keeps isotropic filtering. */
  anisotropy?: number;
  /** Environment maps with a refraction mapping bend the view ray instead of mirroring it. */
  refraction?: boolean;
}

export interface SerializedShader {
  vertex: string;
  fragment: string;
  fragmentAfter?: string;
  uniforms: Record<string, UniformValue>;
  defines: Record<string, string | number | boolean>;
  patched: boolean;
}

export interface SerializedMaterial {
  /** Value of the shader's `time` uniform, from the material's `userData.cpuRenderer.time`. */
  time?: number;
  /** Planar captures for this surface; their textures are named after `id`, the material's uuid. */
  planarCapture?: { id: string };
  /** Simulation baked into the material's uniforms before rendering. */
  compute?: MaterialCompute;
  type: string;
  name: string;
  values: Record<string, UniformValue>;
  maps: Record<string, string>;
  shader?: SerializedShader;
}

export interface SerializedMesh {
  primitive?: "points" | "line" | "lineSegments" | "lineLoop" | "sprite";
  center?: number[];
  attributes: Record<string, SerializedAttribute>;
  index: Uint32Array | null;
  groups: {
    start: number;
    count: number;
    materialIndex: number;
  }[];
  materials: SerializedMaterial[];
  matrixWorld: number[];
  castShadow: boolean;
  receiveShadow: boolean;
  renderOrder: number;
}

const isMedium = (value: unknown): value is Partial<VolumetricMedium> => typeof value === "object" && value !== null;

/** Only the medium's finite numbers: anything else falls back to its default. */
function mediumOf(value: Partial<VolumetricMedium>): VolumetricMedium {
  return Object.fromEntries((["density", "anisotropy", "spread"] as const)
    .filter(key => Number.isFinite(value[key])).map(key => [key, value[key]]));
}

export interface SerializedLight {
  volumetric?: boolean;
  /** The air a volumetric light scatters in, from `userData.cpuRenderer.volumetric` when it is an object. */
  medium?: VolumetricMedium;
  type: string;
  color: number[];
  intensity: number;
  position: number[];
  target: number[];
  distance: number;
  decay: number;
  angle: number;
  penumbra: number;
  groundColor: number[];
  width: number;
  height: number;
  matrixWorld: number[];
  castShadow: boolean;
  /** Texture a spot light projects, with the matrix that takes a world position to its UV (Three's `spotLightMatrix`). */
  map?: string;
  mapMatrix?: number[];
}

/** `scene.fog`: linear between `near` and `far`, or exponential squared when `density` is set. */
export interface SerializedFog {
  color: number[];
  near: number;
  far: number;
  density?: number;
}

export interface SerializedScene {
  sky?: ProceduralSky;
  fog?: SerializedFog;
  meshes: SerializedMesh[];
  lights: SerializedLight[];
  textures: SerializedTexture[];
}

export interface SerializedSceneResult {
  scene: SerializedScene;
  transfer: Transferable[];
}

export { sharedFloat32, sharedUint32 };

const materialTypes = new Set([
  "PointsMaterial", "LineBasicMaterial", "LineDashedMaterial", "SpriteMaterial",
  "MeshBasicMaterial", "MeshLambertMaterial", "MeshPhongMaterial", "MeshToonMaterial",
  "MeshStandardMaterial", "MeshPhysicalMaterial", "MeshNormalMaterial", "MeshDepthMaterial",
  "MeshDistanceMaterial", "MeshMatcapMaterial", "ShadowMaterial", "ShaderMaterial", "RawShaderMaterial",
]);

const properties = [
  "size", "sizeAttenuation", "rotation", "linewidth", "scale", "dashSize", "gapSize",
  "color", "emissive", "emissiveIntensity", "roughness", "metalness", "opacity", "transparent", "alphaTest",
  "side", "vertexColors", "flatShading", "visible", "fog", "depthTest", "depthWrite", "depthFunc", "colorWrite",
  "blending", "premultipliedAlpha", "alphaHash", "polygonOffset", "polygonOffsetFactor", "polygonOffsetUnits",
  "ior", "transmission", "thickness", "dispersion", "attenuationColor", "attenuationDistance", "clearcoat", "clearcoatRoughness",
  "sheen", "sheenRoughness", "sheenColor", "iridescence", "iridescenceIOR", "iridescenceThicknessRange",
  "specularIntensity", "specularColor", "specular", "shininess", "normalScale", "normalMapType", "bumpScale",
  "displacementScale", "displacementBias", "aoMapIntensity", "lightMapIntensity", "envMapIntensity",
  "reflectivity", "refractionRatio", "combine", "toneMapped", "wireframe", "wireframeLinewidth", "depthPacking",
  "anisotropy", "anisotropyRotation", "clearcoatNormalScale", "clipIntersection",
];

export function serializeValue(value: unknown, textures: Map<string, THREE.Texture>): UniformValue {
  if (value == null)
    return null;
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "string")
    return value;

  if (value instanceof THREE.Texture) {
    textures.set(value.uuid, value);

    return { texture: value.uuid };
  }

  if (value instanceof THREE.Color)
    return value.toArray();
  if (Array.isArray(value))
    return value.map(v => serializeValue(v, textures));
  if (ArrayBuffer.isView(value))
    return Array.from(value as unknown as ArrayLike<number>);
  if (typeof value === "object" && "toArray" in value)
    return (value as THREE.Vector3).toArray();
  if (typeof value === "object")
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, serializeValue(v, textures)]));
  throw new Error(`CPU renderer: unsupported uniform value (${typeof value}).`);
}

function serializeMaterial(
  material: THREE.Material,
  textures: Map<string, THREE.Texture>): SerializedMaterial {
  if (!materialTypes.has(material.type))
    throw new Error(`CPU renderer: unsupported material ${material.type} (${material.name}).`);
  if (material.stencilWrite) throw new Error(`CPU renderer: stencil operations are unsupported (${material.name || material.type}).`);
  const record = material as unknown as Record<string, unknown>;
  const values: Record<string, UniformValue> = {};
  const maps: Record<string, string> = {};
  for (const name of properties)
    if (record[name] !== undefined)
      values[name] = serializeValue(record[name], textures);

  for (const [name, value] of Object.entries(record)) {
    if (value instanceof THREE.Texture) {
      textures.set(value.uuid, value);
      maps[name] = value.uuid;
    }
  }

  if (material.blending === THREE.CustomBlending)
    throw new Error(`CPU renderer: CustomBlending is unsupported (${material.name}).`);
  if (material.clippingPlanes?.length)
    values.clippingPlanes = material.clippingPlanes.map(p => [...p.normal.toArray(), p.constant]);

  const result: SerializedMaterial = {
    type: material.type,
    name: material.name || material.type,
    values,
    maps,
  };

  const data = cpuRendererData<CpuRendererMaterialData>(material);
  if (data.compute && (!data.compute.pipeline || !data.compute.id))
    throw new Error(`CPU renderer: ${result.name} has a compute simulation without a pipeline or an id.`);
  if (data.time !== undefined)
    result.time = data.time;
  if (data.planarCapture)
    result.planarCapture = { id: material.uuid };
  if (data.compute)
    result.compute = data.compute;
  const generated = new Set(Object.keys(data.compute?.pipeline.outputs ?? {}));
  if (data.planarCapture)
    for (const name of ["tReflectionMap", "tEnvironmentReflectionMap", "tRefractionMap", "tRefractionDepth"]) generated.add(name);
  const liveUniforms = (material as THREE.ShaderMaterial).uniforms ?? data.uniforms;
  // The CPU pipeline owns these buffers. Never export stale GPU render-target pixels.
  const saved = new Map<string, unknown>();

  for (const name of generated) if (liveUniforms?.[name]) {
    saved.set(name, liveUniforms[name].value);
    liveUniforms[name].value = null;
  }

  try {
    const custom = material as THREE.ShaderMaterial;

    if (custom.isShaderMaterial) {
      result.shader = {

        vertex: custom.vertexShader,
        fragment: custom.fragmentShader,

        uniforms: Object.fromEntries(Object.entries(custom.uniforms).map(([name, uniform]) => [name, serializeValue(
          uniform.value,
          textures,
        )])),

        defines: (custom.defines ?? {}) as Record<string, string | number | boolean>,
        patched: false,
      };
    } else if (material.onBeforeCompile !== THREE.Material.prototype.onBeforeCompile) {
      result.shader = captureMaterialShader(material, textures);
    }

    return result;
  } finally {
    for (const [name, value] of saved) liveUniforms[name].value = value;
  }
}

const vertexHooks = [
  "common",
  "uv_pars_vertex",
  "beginnormal_vertex",
  "defaultnormal_vertex",
  "begin_vertex",
  "project_vertex",
  "worldpos_vertex",
];

const fragmentHooks = [
  "common",
  "uv_pars_fragment",
  "clipping_planes_fragment",
  "color_fragment",
  "normal_fragment_begin",
  "normal_fragment_maps",
  "roughnessmap_fragment",
  "metalnessmap_fragment",
  "emissivemap_fragment",
  "opaque_fragment",
  "dithering_fragment",
];

function shaderTemplate(hooks: string[]) {
  return hooks.slice(
    0,
    2,
  ).map(key => `#include <${key}>`).join("\n") + "\nvoid main() {\n"
    + hooks.slice(2).map(key => `#include <${key}>`).join("\n")
    + "\n}";
}

/** Capture data-only onBeforeCompile hooks without instantiating a GPU renderer. */
function captureMaterialShader(
  material: THREE.Material,
  textures: Map<string, THREE.Texture>): SerializedShader {
  const shader = {
    vertexShader: shaderTemplate(vertexHooks),
    fragmentShader: shaderTemplate(fragmentHooks),
    uniforms: {} as Record<string, THREE.IUniform>,
  };

  const unavailableRenderer = new Proxy({}, {
    get(_target, property) {
      throw new Error(`CPU renderer: onBeforeCompile requires renderer.${String(property)} (${material.name || material.type}).`);
    },
  });

  material.onBeforeCompile(
    shader as THREE.WebGLProgramParametersWithUniforms,
    unavailableRenderer as THREE.WebGLRenderer,
  );

  const resolveHooks = (source: string, hooks: string[]) =>
    source.replace(
      /#include <(\w+)>/g,
      (include, name: string) => hooks.includes(name) ? "" : include,
    );

  const opaque = "#include <opaque_fragment>";
  if (!shader.fragmentShader.includes(opaque))
    throw new Error(`CPU renderer: replacing <opaque_fragment> is unsupported (${material.name || material.type}).`);
  const mainStart = shader.fragmentShader.indexOf("void main() {");
  if (mainStart < 0)
    throw new Error("CPU renderer: onBeforeCompile replaced the material entry point.");
  const common = shader.fragmentShader.slice(0, mainStart);
  const [before, after] = shader.fragmentShader.split(opaque);
  const end = shader.vertexShader.lastIndexOf("}");

  const vertex = shader.vertexShader.slice(
    0,
    end,
  ) + "\ncpuPosition = transformed; cpuNormal = objectNormal;\n"
    + shader.vertexShader.slice(end);

  return {
    vertex: resolveHooks(vertex, vertexHooks),
    fragment: resolveHooks(before + "\n}", fragmentHooks),
    fragmentAfter: resolveHooks(common + "\nvoid main() {\n" + after, fragmentHooks),
    uniforms: Object.fromEntries(Object.entries(shader.uniforms).map(([name, uniform]) => [name, serializeValue(
      uniform.value,
      textures,
    )])),
    defines: ((material as THREE.ShaderMaterial).defines ?? {}) as Record<string, string | number | boolean>,
    patched: true,
  };
}

type TexturePixels = Pick<SerializedTexture, "width" | "height" | "data" | "colorSpace" | "mipmaps">;

function textureMetadata(texture: THREE.Texture, pixels: TexturePixels): SerializedTexture {
  if (texture.matrixAutoUpdate) texture.updateMatrix();

  return { ...pixels, id: texture.uuid, wrapS: texture.wrapS, wrapT: texture.wrapT,
    matrix: texture.matrix.toArray(), flipY: texture.flipY, channel: texture.channel,
    nearest: texture.magFilter === THREE.NearestFilter, minFilter: texture.minFilter, magFilter: texture.magFilter,
    ...texture.anisotropy > 1 ? { anisotropy: texture.anisotropy } : {},
    refraction: texture.mapping === THREE.EquirectangularRefractionMapping
      || texture.mapping === THREE.CubeRefractionMapping };
}

interface ReadableImage {
  data?: ArrayLike<number>;
  width: number;
  height: number;
}

/** RGBA pixels of one image of `texture`: bytes stay bytes, half floats and floats become floats. */
function readPixels(texture: THREE.Texture, image: ReadableImage) {
  const { width, height } = image;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1)
    throw new Error("CPU renderer: invalid texture dimensions.");
  const byteTexture = !image.data || texture.type === THREE.UnsignedByteType;
  const data = byteTexture ? sharedUint8(width * height * 4) : sharedFloat32(width * height * 4);

  if (image.data) {
    const channels = image.data.length / (width * height);
    if (![1, 2, 3, 4].includes(channels))
      throw new Error("CPU renderer: invalid texture channel count.");

    for (let i = 0; i < width * height; i++)
      for (let c = 0; c < 4; c++) {
        let value = c < channels ? image.data[i * channels + c] : c === 3 ? byteTexture ? 255 : 1 : 0;

        if (c < channels) {
          if (texture.type === THREE.HalfFloatType)
            value = THREE.DataUtils.fromHalfFloat(value);
          else if (!byteTexture && texture.type !== THREE.FloatType)
            throw new Error(`CPU renderer: unsupported texture type ${texture.type}.`);
        }

        data[i * 4 + c] = value;
      }
  } else {
    // Read strips instead of keeping a second full-size canvas + ImageData beside the snapshot.
    const stripHeight = Math.min(height, 64);
    const canvas = new OffscreenCanvas(width, stripHeight);

    try {
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("CPU renderer: Canvas 2D is unavailable.");

      for (let y = 0; y < height; y += stripHeight) {
        const rows = Math.min(stripHeight, height - y);
        context.clearRect(0, 0, width, stripHeight);
        context.drawImage(image as unknown as CanvasImageSource, 0, y, width, rows, 0, 0, width, rows);
        data.set(context.getImageData(0, 0, width, rows).data, y * width * 4);
      }
    } finally {
      canvas.width = canvas.height = 0;
    }
  }

  return { width, height, data, byteTexture };
}

/**
 * Cube maps become the equirectangular image every environment lookup already samples. Faces follow the GL
 * layout (+x, -x, +y, -y, +z, -z), and x is mirrored as Three does with `flipEnvMap` for real cube textures.
 */
function cubeToEquirect(texture: THREE.Texture, images = texture.image) {
  const faces = (images as (ReadableImage | THREE.Texture)[]).map(face =>
    readPixels(texture, ((face as THREE.Texture).isTexture ? (face as THREE.Texture).image : face) as ReadableImage));

  const size = Math.min(faces[0].width, 1024), width = size * 4, height = size * 2;
  const byteTexture = faces[0].byteTexture;
  const data = byteTexture ? sharedUint8(width * height * 4) : sharedFloat32(width * height * 4);

  for (let j = 0; j < height; j++) {
    const latitude = ((j + .5) / height - .5) * Math.PI, y = Math.sin(latitude), ring = Math.cos(latitude);

    for (let i = 0; i < width; i++) {
      const longitude = ((i + .5) / width - .5) * 2 * Math.PI;
      const x = -Math.cos(longitude) * ring, z = Math.sin(longitude) * ring;
      const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
      let face: number, sc: number, tc: number, ma: number;

      if (ax >= ay && ax >= az) {
        face = x > 0 ? 0 : 1; sc = x > 0 ? -z : z; tc = -y; ma = ax;
      } else if (ay >= az) {
        face = y > 0 ? 2 : 3; sc = x; tc = y > 0 ? z : -z; ma = ay;
      } else {
        face = z > 0 ? 4 : 5; sc = z > 0 ? x : -x; tc = -y; ma = az;
      }

      const source = faces[face];
      const px = Math.min(source.width - 1, Math.floor((sc / ma + 1) / 2 * source.width));
      const py = Math.min(source.height - 1, Math.floor((tc / ma + 1) / 2 * source.height));
      const from = (py * source.width + px) * 4, to = (j * width + i) * 4;
      for (let c = 0; c < 4; c++) data[to + c] = source.data[from + c];
    }
  }

  return { width, height, data, byteTexture };
}

export async function serializeTexture(texture: THREE.Texture): Promise<SerializedTexture> {
  // KTX2Loader transcodes to plain RGBA bytes when no GPU block format is available: those levels are readable.
  if ((texture as THREE.CompressedTexture).isCompressedTexture && (texture.format !== THREE.RGBAFormat
    || (texture as THREE.CompressedArrayTexture).isCompressedArrayTexture
    || (texture as THREE.CompressedCubeTexture).isCompressedCubeTexture
    || !(texture.mipmaps[0] as ReadableImage | undefined)?.data))
    throw new Error("CPU renderer: compressed textures are unsupported unless transcoded to RGBA (KTX2Loader without a GPU format).");
  const cube = (texture as THREE.CubeTexture).isCubeTexture;
  const manual = texture.mipmaps;
  const image = (!cube && manual.length ? manual[0] : texture.image) as ReadableImage | unknown[] | undefined;
  if (texture.isRenderTargetTexture || (cube ? !Array.isArray(image) || image.length !== 6
    : !image || Array.isArray(image) || !image.width || !image.height))
    throw new Error(`CPU renderer: texture ${texture.name || texture.uuid} has no readable pixels.`);

  const convert = (input: ReadableImage | unknown[]): TexturePixels => {
    const { width, height, data, byteTexture } = cube ? cubeToEquirect(texture, input)
      : readPixels(texture, input as ReadableImage);

    let colorSpace: string = texture.colorSpace;

    if (!byteTexture && colorSpace === THREE.SRGBColorSpace) {
      for (let i = 0; i < data.length; i += 4) for (let c = 0; c < 3; c++) data[i + c] = srgbToLinear(data[i + c]);
      colorSpace = THREE.LinearSRGBColorSpace;
    }

    return { width, height, data, colorSpace };
  };

  const base = convert(image!);
  const mipmaps: SerializedMipLevel[] = [];
  if (manual.length) {
    let previous = base;

    for (const mip of cube ? manual : manual.slice(1)) {
      const pixels = convert(cube ? (mip as unknown as THREE.CubeTexture).image : mip as ReadableImage);
      // Cube faces are capped during conversion; skip levels larger than that proxy resolution.
      if (cube && pixels.width >= previous.width) continue;
      if (pixels.width !== Math.max(1, Math.floor(previous.width / 2))
        || pixels.height !== Math.max(1, Math.floor(previous.height / 2))
        || (previous.width === 1 && previous.height === 1))
        throw new Error("CPU renderer: invalid mipmap dimensions.");
      if (pixels.colorSpace !== base.colorSpace
        || (pixels.data instanceof Uint8Array) !== (base.data instanceof Uint8Array))
        throw new Error("CPU renderer: mipmap pixel encodings must match the base level.");
      mipmaps.push(pixels);
      previous = pixels;
    }

    // A converted cube's last 1x1 face is a 4x2 equirectangular image; finish its proxy chain.
    if (cube) mipmaps.push(...generateMipmaps({ ...previous, colorSpace: base.colorSpace }));
  } else if (texture.generateMipmaps) mipmaps.push(...generateMipmaps(base));
  const metadata = textureMetadata(texture, { ...base, ...(mipmaps.length ? { mipmaps } : {}) });

  return cube ? { ...metadata, flipY: false, wrapS: THREE.RepeatWrapping, wrapT: THREE.ClampToEdgeWrapping,
    matrix: new THREE.Matrix3().toArray() } : metadata;
}

/** Runs each material's `userData.cpuRenderer.prepare`, so the snapshot needs no onBeforeRender or GPU. */
export function prepareCpuMaterials(scene: THREE.Scene) {
  scene.updateMatrixWorld(true);
  traverseRendered(scene, object => {
    const mesh = object as THREE.Mesh;
    if (!mesh.material) return;
    for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material])
      cpuRendererData<CpuRendererMaterialData>(material).prepare?.(scene);
  });
}

/** Visits the visible objects the renderer draws: an excluded object hides its children too, like `visible`. */
export function traverseRendered(object: THREE.Object3D, visit: (object: THREE.Object3D) => void) {
  if (!object.visible || cpuRendererData(object).exclude)
    return;
  visit(object);
  for (const child of object.children)
    traverseRendered(child, visit);
}

export async function serializeScene(scene: THREE.Scene): Promise<SerializedSceneResult> {
  const sceneFog = scene.fog as (THREE.Fog & THREE.FogExp2) | null;
  prepareCpuMaterials(scene);
  const meshes: SerializedMesh[] = [], lights: SerializedLight[] = [];
  const textures = new Map<string, THREE.Texture>();

  const geometryCache = new Map<THREE.BufferGeometry, {
    attributes: Record<string, SerializedAttribute>; index: Uint32Array | null;
  }>();

  const materialCache = new Map<THREE.Material, SerializedMaterial>();
  traverseRendered(scene, object => {
    const mesh = object as THREE.Mesh;

    if ((mesh.isMesh || (object as THREE.Points).isPoints || (object as THREE.Line).isLine || (object as THREE.Sprite).isSprite) && mesh.geometry?.getAttribute("position")) {
      const geometry = mesh.geometry;
      const deformed = (mesh as THREE.SkinnedMesh).isSkinnedMesh || mesh.morphTargetInfluences?.some(v => v !== 0);
      const cached = deformed ? undefined : geometryCache.get(geometry);
      const attributes: Record<string, SerializedAttribute> = cached?.attributes ?? {};

      for (const [name, attr] of cached ? [] : Object.entries(geometry.attributes)) {
        const data = sharedFloat32(attr.count * attr.itemSize);
        for (let i = 0; i < attr.count; i++)
          for (let c = 0; c < attr.itemSize; c++)
            data[i * attr.itemSize + c] = attr.getComponent(i, c);
        attributes[name] = {
          data,
          itemSize: attr.itemSize,
        };
      }

      if (deformed) {
        (mesh as THREE.SkinnedMesh).skeleton?.update();
        const vertex = new THREE.Vector3();

        for (let i = 0; i < attributes.position.data.length / 3; i++) {
          mesh.getVertexPosition(i, vertex);
          vertex.toArray(attributes.position.data, i * 3);
        }

        // Recompute the normals of the deformed surface; no stale bind-pose normals.
        const baked = new THREE.BufferGeometry();
        baked.setAttribute("position", new THREE.BufferAttribute(attributes.position.data, 3));
        baked.setIndex(geometry.index);
        baked.computeVertexNormals();
        const normals = sharedFloat32(baked.getAttribute("normal").array.length);
        normals.set(baked.getAttribute("normal").array as ArrayLike<number>);
        attributes.normal = {
          data: normals,
          itemSize: 3,
        };
        baked.dispose();
      }

      let index: Uint32Array | null = cached?.index ?? null;

      if (!cached && geometry.index) {
        index = sharedUint32(geometry.index.count);
        index.set(geometry.index.array as ArrayLike<number>);
      }

      if (!deformed && !cached) geometryCache.set(geometry, { attributes, index });
      const count = index?.length ?? attributes.position.data.length / 3;
      const start = geometry.drawRange.start, end = Math.min(count, start + geometry.drawRange.count);
      const multi = Array.isArray(mesh.material);
      const sourceMaterials = multi ? mesh.material as THREE.Material[] : [mesh.material as THREE.Material];

      const materials = sourceMaterials.map(m => {
        let serialized = materialCache.get(m);

        if (!serialized) {
          serialized = serializeMaterial(m, textures);
          materialCache.set(m, serialized);
        }

        return serialized;
      });

      const groups = (multi ? geometry.groups : [{

        start: 0,
        count,
        materialIndex: 0,

      }]).map(g => {
        const first = Math.max(start, g.start), last = Math.min(end, g.start + g.count);

        return {

          start: first,
          count: Math.max(0, last - first),
          materialIndex: g.materialIndex ?? 0,

        };
      });

      const instanced = mesh as THREE.InstancedMesh;

      for (let i = 0; i < (instanced.isInstancedMesh ? instanced.count : 1); i++) {
        const matrix = mesh.matrixWorld.clone();

        if (instanced.isInstancedMesh) {
          const instance = new THREE.Matrix4();
          instanced.getMatrixAt(i, instance);
          matrix.multiply(instance);
        }

        let instanceMaterials = materials;

        if (instanced.isInstancedMesh && instanced.instanceColor) {
          const color = new THREE.Color();
          instanced.getColorAt(i, color);
          instanceMaterials
            = materials.map(m =>
              ({
                ...m,
                values: {
                  ...m.values,
                  color: ((m.values.color as number[]) ?? [1, 1, 1]).map((v, c) => v * color.toArray()[c]),
                },
              }));
        }

        meshes.push({
          primitive: (object as THREE.Points).isPoints ? "points"
            : (object as THREE.Sprite).isSprite ? "sprite"
              : (object as THREE.LineSegments).isLineSegments ? "lineSegments"
                : (object as THREE.LineLoop).isLineLoop ? "lineLoop"
                  : (object as THREE.Line).isLine ? "line" : undefined,
          center: (object as THREE.Sprite).isSprite ? (object as THREE.Sprite).center.toArray() : undefined,

          attributes,
          index,
          groups,
          materials: instanceMaterials,
          matrixWorld: matrix.toArray(),
          castShadow: mesh.castShadow,
          receiveShadow: mesh.receiveShadow,
          renderOrder: mesh.renderOrder,

        });
      }
    }

    const light = object as THREE.Light;
    if (!light.isLight)
      return;
    const l = light as THREE.SpotLight & THREE.HemisphereLight & THREE.RectAreaLight;
    const position = light.getWorldPosition(new THREE.Vector3()).toArray();
    const target = l.target?.getWorldPosition(new THREE.Vector3()).toArray() ?? [0, 0, 0];
    const map = (light as THREE.SpotLight).isSpotLight ? l.map : null;
    const { volumetric } = cpuRendererData<CpuRendererLightData>(light);

    if (map) {
      textures.set(map.uuid, map);
      l.shadow.updateMatrices(l);
    }

    lights.push({

      type: light.type,
      volumetric: volumetric === true || isMedium(volumetric),
      ...isMedium(volumetric) ? { medium: mediumOf(volumetric) } : {},
      color: light.color.toArray(),
      intensity: light.intensity,
      position,
      target,

      distance: l.distance ?? 0,
      decay: l.decay ?? 0,
      angle: l.angle ?? 0,
      penumbra: l.penumbra ?? 0,

      groundColor: l.groundColor?.toArray() ?? [0, 0, 0],
      width: l.width ?? 0,
      height: l.height ?? 0,

      matrixWorld: light.matrixWorld.toArray(),
      castShadow: light.castShadow,
      ...map ? { map: map.uuid, mapMatrix: l.shadow.matrix.toArray() } : {},
    });
  });
  const serializedTextures: SerializedTexture[] = [];
  const pixelCache = new Map<object, Map<string, TexturePixels>>();

  for (const texture of textures.values()) {
    const key = `${texture.type}:${texture.colorSpace}:${texture.generateMipmaps}`;
    const source = pixelCache.get(texture.source);

    const cached = texture.isRenderTargetTexture || texture.mipmaps.length
      || (texture as THREE.CubeTexture).isCubeTexture
      ? undefined : source?.get(key);

    const serialized = cached ? textureMetadata(texture, cached) : await serializeTexture(texture);
    serializedTextures.push(serialized);

    if (!cached && !texture.mipmaps.length) {
      const entries = source ?? new Map<string, TexturePixels>();
      entries.set(key, { width: serialized.width, height: serialized.height,
        data: serialized.data, colorSpace: serialized.colorSpace, mipmaps: serialized.mipmaps });
      pixelCache.set(texture.source, entries);
    }

    // Let the browser process input and reclaim temporary readback buffers between textures.
    await new Promise<void>(resolve => setTimeout(resolve, 0));
  }

  // Only plain ArrayBuffers go in the transfer list; shared memory is already visible to everyone.
  const buffers = new Set<ArrayBuffer>();

  const collect = (buffer: ArrayBufferLike) => {
    if (!(typeof SharedArrayBuffer !== "undefined" && buffer instanceof SharedArrayBuffer))
      buffers.add(buffer as ArrayBuffer);
  };

  for (const mesh of meshes) {
    for (const attr of Object.values(mesh.attributes))
      collect(attr.data.buffer);
    if (mesh.index)
      collect(mesh.index.buffer);
  }

  for (const texture of serializedTextures) {
    collect(texture.data.buffer);
    for (const level of texture.mipmaps ?? []) collect(level.data.buffer);
  }

  const { sky } = cpuRendererData<CpuRendererSceneData>(scene);

  return {
    scene: {
      ...sky ? { sky } : {},
      ...sceneFog ? { fog: { color: sceneFog.color.toArray(), near: sceneFog.near ?? 0, far: sceneFog.far ?? 0,
        ...sceneFog.isFogExp2 ? { density: sceneFog.density } : {} } } : {},
      meshes,
      lights,
      textures: serializedTextures,
    },
    transfer: [...buffers],
  };
}
