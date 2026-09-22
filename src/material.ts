import { ShaderChunk } from "three";
import type { SerializedLight, SerializedMaterial, SerializedTexture } from "./sceneSerialization.js";
import type { FrameSettings } from "./frameSettings.js";
import { CpuShader, evaluateDerivativeQuad, ShaderError, type DerivativeContext, type Value } from "./glsl.js";
import { add, clamp, cross, dot, length, mix, mul, normalize, reflect, refract, scale, sub, transform4, type Vec3 } from "./math.js";
import { sampleTexture, type TextureSampling } from "./texture.js";
import type { CpuEnvironment } from "./environment.js";

export interface Surface {
  /** Lazy same-primitive helper surfaces, in GL bottom-left to top-right order. */
  quad?: { lane: number; surfaces: () => Surface[] };
  derivatives?: DerivativeContext;
  helper?: boolean;
  uvGradients?: Record<string, TextureSampling>;
  position: number[];
  normal: number[];
  tangent: number[];
  bitangent: number[];
  view: number[];
  attributes: Record<string, number[]>;
  varyings: Record<string, Value>;
  frontFacing: boolean;
  depth: number;
  fragCoord: number[];
  uniforms: Record<string, Value>;
}

export interface ShadedFragment {
  color: number[];
  alpha: number;
  depth: number;
  transmission: number;
  attenuation: number[];
  /** Portion of `color` that transmission replaces with the background (the diffuse term, as in Three). */
  diffuse: number[];
  /** A custom shader with `premultipliedAlpha` already multiplied `color` by `alpha`; blending must not do it again. */
  premultiplied?: boolean;
  /** Opaque scene seen through the refracted exit point; without it the blend uses what lies behind the pixel. */
  transmitted?: number[];
  /** Raw indirect light of this fragment and the factor that multiplies it in `color`; only when GI was queried. */
  indirect?: number[];
  modulation?: number[];
}

export interface LightingContext {
  /** Irradiance / pi from diffuse path tracing, replacing unoccluded environment fill. */
  indirect?: (position: number[], normal: number[]) => number[];
  transport?: "diffuse" | "emission";
  emissionWeight?: number;
  lights: SerializedLight[];
  environment: CpuEnvironment;
  settings: FrameSettings;
  /** `ignoreSelf`: surfaces of the shaded mesh group do not block the light. */
  visibility(position: number[], normal: number[], direction: number[], distance: number, ignoreSelf?: boolean): number;
  /** Ray-traced ambient occlusion in [0, 1]; 1 when the effect is disabled. */
  occlusion: (position: number[], normal: number[]) => number;
  /**
   * Opaque scene seen by the camera through a world-space point, as Three's transmission framebuffer.
   * `blur` is the mip level fraction (roughness scaled by ior) that widens the footprint.
   */
  transmitted?: (exit: number[], blur: number) => number[] | undefined;
}

/**
 * Tone mapping and the output color space belong to the final image (see `resolveSample`): a shader that includes
 * Three's output chunks, as it must to match the other materials in WebGL, keeps writing linear light here.
 */
const shaderChunks: Record<string, string> = {
  ...ShaderChunk, tonemapping_fragment: "", colorspace_fragment: "", encodings_fragment: "",
};

const nativeShaderHelpers = `
vec3 inverseTransformDirection(vec3 dir, mat4 matrix) { return normalize((vec4(dir, 0.0) * matrix).xyz); }
float getDistanceAttenuation(float lightDistance, float cutoffDistance, float decayExponent) {
  float falloff = 1.0 / max(pow(lightDistance, decayExponent), 0.01);
  if (cutoffDistance > 0.0) falloff *= pow(clamp(1.0 - pow(lightDistance / cutoffDistance, 4.0), 0.0, 1.0), 2.0);
  return falloff;
}
float getSpotAttenuation(float coneCosine, float penumbraCosine, float angleCosine) {
  return smoothstep(coneCosine, penumbraCosine, angleCosine);
}
`;

const ONES = [1, 1, 1, 1], ONES3 = [1, 1, 1], ZERO3 = [0, 0, 0];
/** Direction and cone cosines of each light, computed once per light rather than per fragment. */
const lightCache = new WeakMap<SerializedLight, { direction: number[]; outer: number; inner: number }>();

function lightConstants(light: SerializedLight) {
  let c = lightCache.get(light);

  if (!c) {
    c = {
      direction: normalize(sub(light.position, light.target)),
      outer: Math.cos(light.angle), inner: Math.cos(light.angle * (1 - light.penumbra)),
    };
    lightCache.set(light, c);
  }

  return c;
}

const schlick = (f0: number, cosine: number) => f0 + (1 - f0) * (1 - cosine) ** 5;

/** Three's evalSensitivity: the XYZ response of a thin film, Fourier-fitted, in linear sRGB. */
function filmSensitivity(opd: number, shift: number[]): number[] {
  const phase = 2 * Math.PI * opd * 1e-9;
  const val = [5.4856e-13, 4.4201e-13, 5.2481e-13], pos = [1.6810e+06, 1.7953e+06, 2.2084e+06];
  const variance = [4.3278e+09, 9.3046e+09, 6.6121e+09];

  const xyz = [0, 1, 2].map(i => val[i] * Math.sqrt(2 * Math.PI * variance[i])
    * Math.cos(pos[i] * phase + shift[i]) * Math.exp(-phase * phase * variance[i]));

  xyz[0] += 9.7470e-14 * Math.sqrt(2 * Math.PI * 4.5282e+09) * Math.cos(2.2399e+06 * phase + shift[0])
    * Math.exp(-4.5282e+09 * phase * phase);

  const [x, y, z] = xyz.map(c => c / 1.0685e-7);

  return [3.2404542 * x - 1.5371385 * y - .4985314 * z, -.9692660 * x + 1.8760108 * y + .0415560 * z,
    .0556434 * x - .2040259 * y + 1.0572252 * z];
}

/** Three's evalIridescence (Belcour and Barla 2017): Fresnel of a thin film over a base of reflectance `baseF0`. */
function iridescentFresnel(filmIor: number, cosTheta1: number, thickness: number, baseF0: number[]): number[] {
  const ior = 1 + (filmIor - 1) * (thickness >= .03 ? 1 : (t => t * t * (3 - 2 * t))(thickness / .03));
  const cosTheta2Sq = 1 - (1 / ior) ** 2 * (1 - cosTheta1 * cosTheta1);
  if (cosTheta2Sq < 0)
    return [1, 1, 1];
  const cosTheta2 = Math.sqrt(cosTheta2Sq), r12 = schlick(((ior - 1) / (ior + 1)) ** 2, cosTheta1), t121 = 1 - r12;
  const phi21 = Math.PI - (ior < 1 ? Math.PI : 0);

  const baseIor = baseF0.map(f => {
    const root = Math.sqrt(clamp(f, 0, .9999));

    return (1 + root) / (1 - root);
  });

  const r23 = baseIor.map(b => schlick(((b - ior) / (b + ior)) ** 2, cosTheta2));
  const phi = baseIor.map(b => phi21 + (b < ior ? Math.PI : 0)), opd = 2 * ior * thickness * cosTheta2;
  const r123 = r23.map(r => clamp(r12 * r, 1e-5, .9999));
  const rs = r23.map((r, i) => t121 * t121 * r / (1 - r123[i]));
  const result = rs.map(r => r12 + r), cm = rs.map(r => r - t121);

  for (let m = 1; m <= 2; m++) {
    const sensitivity = filmSensitivity(m * opd, phi.map(x => m * x));

    for (let i = 0; i < 3; i++) {
      cm[i] *= Math.sqrt(r123[i]);
      result[i] += cm[i] * 2 * sensitivity[i];
    }
  }

  return result.map(x => Math.max(x, 0));
}

/** Three's IBLSheenBRDF: the Charlie sheen lobe integrated over the hemisphere. */
function sheenAlbedo(cosine: number, roughness: number) {
  const r2 = roughness * roughness, inverse = 1 / (roughness + .1);
  const a = -1.9362 + 1.0678 * roughness + .4573 * r2 - .8469 * inverse;
  const b = -.6014 + .5538 * roughness - .4670 * r2 - .1255 * inverse;

  return clamp(Math.exp(a * clamp(cosine) + b));
}

/** Three's D_Charlie times V_Neubelt. */
function sheenLobe(roughness: number, nh: number, nv: number, nl: number) {
  const inverse = 1 / (roughness * roughness), sin2h = Math.max(1 - nh * nh, .0078125);

  return (2 + inverse) * sin2h ** (inverse * .5) / (2 * Math.PI) * clamp(1 / (4 * (nl + nv - nl * nv)));
}

/**
 * Split-sum environment BRDF, Karis' analytical fit (Three's DFGApprox before its lookup table): the scale and bias
 * that turn F0 into the directional albedo of a GGX lobe. Unlike plain Schlick, it keeps rough surfaces seen at
 * grazing angles from turning into mirrors.
 */
function environmentBrdf(roughness: number, nv: number): [number, number] {
  const x = 1 - roughness, y = .0425 - .0275 * roughness, z = 1.04 - .572 * roughness, w = .022 * roughness - .04;
  const a004 = Math.min(x * x, 2 ** (-9.28 * nv)) * x + y;

  return [-1.04 * a004 + z, 1.04 * a004 + w];
}

/** Three's computeSpecularOcclusion (Lagarde): occlusion fades from glossy reflections. */
const specularOcclusion = (nv: number, ao: number, roughness: number) =>
  clamp((nv + ao) ** (2 ** (-16 * roughness - 1)) - 1 + ao);

const LIT_MATERIALS = new Set(["MeshLambertMaterial", "MeshPhongMaterial", "MeshToonMaterial",
  "MeshStandardMaterial", "MeshPhysicalMaterial"]);

export class CpuMaterial {
  readonly vertex?: CpuShader;
  readonly fragment?: CpuShader;
  readonly fragmentAfter?: CpuShader;
  readonly uniforms: Record<string, Value>;
  /** The tangent frame is only built for normal/bump maps and anisotropy. */
  readonly needsTangents: boolean;
  /** Per-fragment uniforms only matter to shaders; native materials skip the copy. */
  readonly needsUniforms: boolean;
  readonly needsTextureFootprint: boolean;
  /** A ray hitting this material always stops: no map, alpha, cutout or shader that could let it through. */
  readonly rayOpaque: boolean;
  /** Lit native material, with or without hooks: these are the ones that query indirect light from GI. */
  readonly diffuseGi: boolean;
  /** Without maps, vertex colors or shaders, a ray hit can skip interpolating attributes and varyings. */
  readonly needsAttributes: boolean;
  /** Scalar and vector material values, resolved once so shading does not look them up by name. */
  private readonly p: {
    transparent: boolean; vertexColors: boolean; alphaHash: boolean; alphaTest: number; opacity: number;
    color: number[]; roughness: number; metalness: number; emissive: number[]; emissiveIntensity: number;
    ior: number; dielectric: number; specularIntensity: number; specularColor: number[];
    iridescence: number; iridescenceIOR: number; iridescenceThicknessRange: number[];
    clearcoat: number; clearcoatRoughness: number; sheen: number; sheenColor: number[]; sheenRoughness: number;
    anisotropy: number; anisotropyRotation: number; transmission: number; thickness: number;
    attenuationDistance: number; attenuationColor: number[]; envMapIntensity: number; normalMapType: number;
    normalScale: number[]; aoMapIntensity: number; lightMapIntensity: number; shininess: number;
    reflectivity: number; refractionRatio: number; combine: number; clearcoatNormalScale: number[];
    dispersion: number;
    clippingPlanes: number[][] | undefined; clipIntersection: boolean; pbr: boolean;
  };

  constructor(readonly data: SerializedMaterial, readonly textures: Map<string, SerializedTexture>) {
    const pbr = data.type === "MeshStandardMaterial" || data.type === "MeshPhysicalMaterial";
    const ior = this.number("ior", 1.5);
    this.p = {
      transparent: this.flag("transparent"), vertexColors: this.flag("vertexColors"), alphaHash: this.flag("alphaHash"),
      alphaTest: this.number("alphaTest", 0), opacity: this.number("opacity", 1), color: this.vector("color", [1, 1, 1]),
      roughness: this.number("roughness", 1), metalness: this.number("metalness", 0),
      emissive: this.vector("emissive", [0, 0, 0]), emissiveIntensity: this.number("emissiveIntensity", 1),
      ior, dielectric: ((ior - 1) / (ior + 1)) ** 2, specularIntensity: this.number("specularIntensity", 1),
      specularColor: this.vector(pbr ? "specularColor" : "specular", pbr ? [1, 1, 1] : [.066, .066, .066]),
      iridescence: this.number("iridescence", 0), iridescenceIOR: this.number("iridescenceIOR", 1.3),
      iridescenceThicknessRange: this.vector("iridescenceThicknessRange", [100, 400]),
      clearcoat: this.number("clearcoat", 0), clearcoatRoughness: this.number("clearcoatRoughness", 0),
      sheen: this.number("sheen", 0), sheenColor: this.vector("sheenColor", [0, 0, 0]),
      sheenRoughness: this.number("sheenRoughness", 1), anisotropy: this.number("anisotropy", 0),
      anisotropyRotation: this.number("anisotropyRotation", 0), transmission: this.number("transmission", 0),
      thickness: this.number("thickness", 0), attenuationDistance: this.number("attenuationDistance", Infinity),
      attenuationColor: this.vector("attenuationColor", [1, 1, 1]), envMapIntensity: this.number("envMapIntensity", 1),
      normalMapType: this.number("normalMapType", 0), normalScale: this.vector("normalScale", [1, 1]),
      aoMapIntensity: this.number("aoMapIntensity", 1), lightMapIntensity: this.number("lightMapIntensity", 1),
      shininess: this.number("shininess", 30), reflectivity: this.number("reflectivity", 1),
      refractionRatio: this.number("refractionRatio", .98), combine: this.number("combine", 0),
      clearcoatNormalScale: this.vector("clearcoatNormalScale", [1, 1]), dispersion: this.number("dispersion", 0),
      clippingPlanes: data.values.clippingPlanes as number[][] | undefined, clipIntersection: this.flag("clipIntersection"),
      pbr,
    };
    this.uniforms = data.shader?.uniforms ?? {};
    this.rayOpaque = !data.shader && !this.flag("transparent") && !data.maps.map && !data.maps.alphaMap
      && !this.flag("vertexColors") && !this.flag("alphaHash") && this.number("alphaTest", 0) <= 0
      && this.number("transmission", 0) <= 0 && !(data.values.clippingPlanes as unknown[] | undefined)?.length;
    this.needsTangents = Boolean(data.maps.normalMap || data.maps.bumpMap || data.maps.clearcoatNormalMap || this.number("anisotropy", 0) > 0);
    this.diffuseGi = LIT_MATERIALS.has(data.type) && (!data.shader || data.shader.patched);
    this.needsAttributes = Boolean(data.shader) || this.flag("vertexColors")
      || Object.values(data.maps).some(Boolean);
    this.needsUniforms = Boolean(data.shader);
    const textureIds = new Set(Object.values(data.maps));

    const collectTextures = (value: Value) => {
      if (Array.isArray(value)) { for (const child of value) collectTextures(child); } else if (value && typeof value === "object") {
        if (typeof value.texture === "string") textureIds.add(value.texture);
        else for (const child of Object.values(value)) collectTextures(child);
      }
    };

    collectTextures(this.uniforms);
    this.needsTextureFootprint = [...textureIds].some(id => {
      const t = textures.get(id);

      return t && t.minFilter !== undefined && t.minFilter !== t.magFilter;
    });

    if (data.shader) {
      try {
        this.vertex = new CpuShader(data.shader.vertex, data.shader.defines, shaderChunks);
        this.fragment = new CpuShader((data.shader.patched ? nativeShaderHelpers : "") + data.shader.fragment, data.shader.defines, shaderChunks);
        if (data.shader.fragmentAfter)
          this.fragmentAfter = new CpuShader(data.shader.fragmentAfter, data.shader.defines, shaderChunks);
      } catch(error) {
        throw new Error(`${data.name}: ${(error as Error).message}`, { cause: error });
      }
    }
  }

  get usesDerivatives(): boolean {
    return Boolean(this.fragment?.usesDerivatives || this.fragmentAfter?.usesDerivatives);
  }

  get requiresShaderQuad(): boolean {
    return this.usesDerivatives || (this.needsTextureFootprint
      && Boolean(this.fragment?.usesImplicitTextureSampling || this.fragmentAfter?.usesImplicitTextureSampling));
  }

  get needsFragmentQuad(): boolean { return this.requiresShaderQuad || this.needsTextureFootprint; }

  number(name: string, fallback: number) {
    const value = this.data.values[name];

    return typeof value === "number" ? value : fallback;
  }

  flag(name: string, fallback = false) {
    const value = this.data.values[name];

    return typeof value === "boolean" ? value : fallback;
  }

  vector(name: string, fallback: number[]): number[] {
    const value = this.data.values[name];

    return Array.isArray(value) ? value as number[] : fallback;
  }

  texture = (sampler: Value, uv: number[], sampling?: TextureSampling): number[] => {
    if (!sampler || typeof sampler !== "object" || Array.isArray(sampler) || typeof sampler.texture !== "string")
      throw new Error(`${this.data.name}: unbound sampler.`);
    const texture = this.textures.get(sampler.texture);
    if (!texture)
      throw new Error(`${this.data.name}: missing texture ${sampler.texture}.`);

    return sampleTexture(texture, uv, false, sampling);
  };

  textureNeedsFootprint = (sampler: Value): boolean => {
    const id = sampler && typeof sampler === "object" && !Array.isArray(sampler) ? sampler.texture : null;
    const texture = typeof id === "string" ? this.textures.get(id) : undefined;

    return Boolean(texture && texture.minFilter !== undefined && texture.minFilter !== texture.magFilter);
  };

  map(
    name: string,
    attributes: Record<string, number[]>,
    fallback: number[] = ONES,
    offset?: number[], gradients?: Record<string, TextureSampling>): number[] {
    const id = this.data.maps[name];
    if (!id)
      return fallback;
    const texture = this.textures.get(id);
    if (!texture)
      return fallback;
    const uv = attributes[texture.channel ? `uv${texture.channel}` : "uv"] ?? [0, 0];

    return sampleTexture(texture, offset ? [uv[0] + offset[0], uv[1] + offset[1]] : uv, true,
      gradients?.[texture.channel ? `uv${texture.channel}` : "uv"]);
  }

  alpha(attributes: Record<string, number[]>, gradients?: Record<string, TextureSampling>) {
    let alpha = this.p.opacity;
    if (this.data.maps.map) alpha *= this.map("map", attributes, ONES, undefined, gradients)[3];
    if (this.data.maps.alphaMap) alpha *= this.map("alphaMap", attributes, ONES, undefined, gradients)[1];

    return alpha;
  }

  /**
   * UV footprints of `s` from its fragment quad, computed once for all four lanes so that alpha tests and shading
   * read the same mip level. Undefined without a quad or when no texture needs a footprint.
   */
  uvGradients(s: Surface): Record<string, TextureSampling> | undefined {
    if (s.uvGradients || !this.needsTextureFootprint || !s.quad) return s.uvGradients;
    const helpers = s.quad.surfaces();
    const keys = Object.keys(s.attributes).filter(name => /^uv\d*$/.test(name));
    if (!s.attributes.uv && s.varyings.gl_PointCoord) keys.push("uv");

    for (let lane = 0; lane < 4; lane++) {
      const gradients: Record<string, TextureSampling> = {};

      for (const key of keys) {
        const uv = (i: number) => helpers[i].attributes[key]
          ?? [Number((helpers[i].varyings.gl_PointCoord as number[])[0]),
            1 - Number((helpers[i].varyings.gl_PointCoord as number[])[1])];

        const left = uv(lane & 2), right = uv((lane & 2) + 1), bottom = uv(lane & 1), top = uv((lane & 1) + 2);
        gradients[key] = { dx: [right[0] - left[0], right[1] - left[1]],
          dy: [top[0] - bottom[0], top[1] - bottom[1]] };
      }

      helpers[lane].uvGradients = gradients;
    }

    return s.uvGradients;
  }

  shade(s: Surface, ctx: LightingContext): ShadedFragment | null {
    const planes = this.p.clippingPlanes;

    if (planes?.length && !s.helper) {
      const tests = planes.map(p => dot(p, s.position) + p[3] < 0);
      if (this.p.clipIntersection ? tests.every(Boolean) : tests.some(Boolean))
        return null;
    }

    // Arbitrary display shaders have no recoverable BSDF or emission closure.
    if (ctx.transport && this.fragment && !this.data.shader?.patched)
      return { color: [0, 0, 0], alpha: 1, depth: s.depth,
        transmission: 0, attenuation: [1, 1, 1], diffuse: [0, 0, 0] };

    this.uvGradients(s);

    if (this.requiresShaderQuad && !s.derivatives) {
      if (!s.quad) throw new ShaderError("material derivatives require rasterized fragment helpers");
      const quad = s.quad, surfaces = quad.surfaces();

      return evaluateDerivativeQuad((lane, derivatives) => this.shade(
        { ...surfaces[lane], derivatives, helper: lane !== quad.lane }, ctx), quad.lane)[quad.lane];
    }

    // Only shaders read this; native materials do not pay for the per-fragment copy.
    const shaderInputs: Record<string, Value> = this.fragment || this.fragmentAfter ? {
      ...s.uniforms,
      ...this.uniforms,
      ...s.varyings,
      gl_FragCoord: s.fragCoord,
      gl_FrontFacing: s.frontFacing,
      gl_FragDepth: s.depth,
      gl_FragColor: [0, 0, 0, 1],
    } : {};

    if (this.fragment && !this.data.shader?.patched) {
      const result = this.fragment.run(shaderInputs, { texture: this.texture,
        textureNeedsFootprint: this.textureNeedsFootprint, derivatives: s.derivatives });

      if (!result)
        return null;
      const color = (this.fragment.outputs.length ? result[this.fragment.outputs[0]] : result.gl_FragColor) as number[];
      if (!Array.isArray(color) || color.length !== 4)
        throw new Error(`${this.data.name}: fragment shader must write vec4 output.`);

      return {
        color: color.slice(0, 3),
        alpha: clamp(color[3]),
        depth: Number(result.gl_FragDepth ?? s.depth),
        transmission: 0,
        attenuation: [1, 1, 1],
        diffuse: color.slice(0, 3),
        premultiplied: this.flag("premultipliedAlpha"),
      };
    }

    const sampleMap = (name: string, attributes: Record<string, number[]>,
      fallback: number[] = ONES, offset?: number[]) =>
      this.map(name, attributes, fallback, offset, s.uvGradients);

    if (this.data.type === "LineDashedMaterial") {
      const distance = (s.attributes.lineDistance?.[0] ?? 0) * this.number("scale", 1);
      const dash = this.number("dashSize", 3), period = dash + this.number("gapSize", 1);
      if (period > 0 && ((distance % period) + period) % period > dash) return null;
    }

    const coord = s.varyings.gl_PointCoord as number[] | undefined;

    const attributes = this.data.type === "PointsMaterial" && coord && !s.attributes.uv
      ? { ...s.attributes, uv: [coord[0], 1 - coord[1]] } : s.attributes;

    const p = this.p;
    let color = this.data.maps.map ? mul(p.color, sampleMap("map", attributes)) : p.color, alpha = this.alpha(attributes, s.uvGradients);

    if (p.vertexColors && attributes.color) {
      color = mul(color, attributes.color);
      alpha *= attributes.color[3] ?? 1;
    }

    if (alpha < p.alphaTest && !s.helper)
      return null;

    if (p.alphaHash) {
      const noise = Math.sin(dot(s.position, [12.9898, 78.233, 37.719])) * 43758.5453;
      if (alpha < noise - Math.floor(noise) && !s.helper)
        return null;
      alpha = 1;
    }

    let n = normalize(s.normal);
    if (!s.frontFacing)
      n = scale(n, -1);
    const geometricNormal = n;

    if (this.data.maps.normalMap) {
      const tex = sampleMap("normalMap", attributes), ns = p.normalScale;
      const mapped = [(tex[0] * 2 - 1) * ns[0], (tex[1] * 2 - 1) * ns[1], tex[2] * 2 - 1];
      n
        = p.normalMapType === 1 ? normalize(mapped) : normalize(add(
          add(
            scale(
              s.tangent,
              mapped[0],
            ),
            scale(
              s.bitangent,
              mapped[1],
            ),
          ),
          scale(
            n,
            mapped[2],
          ),
        ));
    } else if (this.data.maps.bumpMap) {
      const tex = this.textures.get(this.data.maps.bumpMap)!, du = 1 / tex.width, dv = 1 / tex.height, h = sampleMap(
          "bumpMap",
          attributes,
        )[0], amount = this.number(
          "bumpScale",
          1,
        );

      const dx = (sampleMap(
          "bumpMap",
          attributes,
          [1, 1, 1, 1],
          [du, 0],
        )[0] - h) * amount, dy = (sampleMap(
          "bumpMap",
          attributes,
          [1, 1, 1, 1],
          [0, dv],
        )[0] - h) * amount;

      n = normalize(sub(n, add(scale(s.tangent, dx), scale(s.bitangent, dy))));
    }

    let roughness = clamp(this.data.maps.roughnessMap ? p.roughness * sampleMap("roughnessMap", attributes)[1] : p.roughness, .0525, 1);
    let metalness = clamp(this.data.maps.metalnessMap ? p.metalness * sampleMap("metalnessMap", attributes)[2] : p.metalness);

    let emissive = p.emissiveIntensity === 1 && !this.data.maps.emissiveMap ? p.emissive
      : scale(this.data.maps.emissiveMap ? mul(p.emissive, sampleMap("emissiveMap", attributes)) : p.emissive, p.emissiveIntensity);

    // Globals computed by the chunks before <opaque_fragment> remain visible after it.
    let stage: Record<string, Value> = {};

    if (this.fragment) {
      const result = this.fragment.run({

        ...shaderInputs,
        diffuseColor: [...color, alpha],
        normal: n,
        roughnessFactor: roughness,
        metalnessFactor: metalness,

        totalEmissiveRadiance: emissive,
        vViewPosition: s.uniforms.vViewPosition ?? scale(s.view, -1),
        vUv: attributes.uv ?? [0, 0],
        outgoingLight: color,

      }, { texture: this.texture, textureNeedsFootprint: this.textureNeedsFootprint, derivatives: s.derivatives });

      if (!result)
        return null;
      stage = result;
      const diffuse = result.diffuseColor as number[];
      color = diffuse.slice(0, 3) as Vec3;
      alpha = diffuse[3];
      n = normalize(result.normal as number[]);
      roughness = clamp(Number(result.roughnessFactor), .0525, 1);
      metalness = clamp(Number(result.metalnessFactor));
      emissive = result.totalEmissiveRadiance as Vec3;
    }

    const type = this.data.type;
    // Raw indirect light and its multiplier: the denoiser filters the former and reapplies it through the latter.
    let indirectRaw: number[] | undefined, modulation: number[] | undefined;

    const finish = (rgb: number[], transmission = 0, attenuation = [1, 1, 1], diffuse = rgb,
      transmitted?: number[]): ShadedFragment | null => {
      const result = this.fragmentAfter?.run({
        ...stage,
        ...shaderInputs,
        diffuseColor: [...color, alpha],
        normal: n,
        outgoingLight: rgb,
        gl_FragColor: [...rgb, p.transparent ? clamp(alpha) : 1],
      }, { texture: this.texture, textureNeedsFootprint: this.textureNeedsFootprint, derivatives: s.derivatives });

      if (this.fragmentAfter && !result)
        return null;
      const output = result?.gl_FragColor as number[] | undefined;
      const final = output?.slice(0, 3) ?? rgb;

      return {
        color: final,
        alpha: output ? output[3] : p.transparent ? clamp(alpha) : 1,
        depth: s.depth,
        transmission,
        attenuation,
        // If the hook rewrote the final color, the diffuse/specular split no longer exists.
        diffuse: output ? final : diffuse,
        transmitted,
        indirect: output ? undefined : indirectRaw,
        modulation: output ? undefined : modulation,
      };
    };

    if (ctx.transport && !LIT_MATERIALS.has(type))
      return finish([0, 0, 0]);
    if (ctx.transport === "emission")
      return finish(emissive);
    if (ctx.emissionWeight !== undefined)
      emissive = scale(emissive, ctx.emissionWeight);
    if (type === "MeshNormalMaterial")
      return finish(n.map(v => v * .5 + .5));
    if (type === "MeshDepthMaterial" || type === "MeshDistanceMaterial")
      return finish([1 - s.depth, 1 - s.depth, 1 - s.depth]);

    if (type === "MeshMatcapMaterial") {
      const x = normalize([s.view[2], 0, -s.view[0]]), y = cross(s.view, x);
      const texture = this.textures.get(this.data.maps.matcap);

      const matcap = texture ? sampleTexture(
        texture,
        [dot(
          x,
          n,
        ) * .495 + .5, dot(
          y,
          n,
        ) * .495 + .5],
      ) : [dot(
        n,
        s.view,
      ) * .5 + .5, dot(
        n,
        s.view,
      ) * .5 + .5, dot(
        n,
        s.view,
      ) * .5 + .5];

      return finish(mul(color, matcap));
    }

    const v = normalize(s.view), nv = Math.max(.0001, dot(n, v)), reflection = reflect(scale(v, -1), n);
    const ownEnv = this.textures.get(this.data.maps.envMap);

    // Three's equirectUv; the texture matrix does not apply to environment lookups.
    const equirect = (texture: SerializedTexture, direction: number[]) => sampleTexture(texture, [
      Math.atan2(direction[2], direction[0]) / (2 * Math.PI) + .5,
      Math.asin(clamp(direction[1], -1, 1)) / Math.PI + .5,
    ], false)
      .slice(0, 3);

    // Three's envmap_fragment: Basic, Lambert and Phong blend their own envMap over the lit color through `combine`.
    const legacyEnv = ownEnv && !ctx.transport
      && (type === "MeshBasicMaterial" || type === "MeshLambertMaterial" || type === "MeshPhongMaterial")
      ? (rgb: number[]) => {
        const direction = ownEnv.refraction ? refract(scale(v, -1), n, p.refractionRatio) : reflection;
        const envColor = equirect(ownEnv, direction);
        const amount = p.reflectivity * (this.data.maps.specularMap ? sampleMap("specularMap", attributes)[0] : 1);

        return p.combine === 1 ? mix(rgb, envColor, amount)
          : p.combine === 2 ? add(rgb, scale(envColor, amount)) : mix(rgb, mul(rgb, envColor), amount);
      } : null;

    if (["MeshBasicMaterial", "PointsMaterial", "LineBasicMaterial", "LineDashedMaterial", "SpriteMaterial"].includes(type))
      return finish(legacyEnv ? legacyEnv(color) : color);
    const envIntensity = ctx.settings.environmentIntensity * p.envMapIntensity;

    const env = (direction: number[], r = 0, diffuse = false) => {
      const own = p.pbr ? ownEnv : undefined;

      return scale(own ? equirect(own, direction)
        : ctx.environment.sample(direction, ctx.settings.environmentRotation, r, diffuse), envIntensity);
    };

    const pbr = p.pbr, dielectric = p.dielectric;
    const specularMapName = pbr ? "specularColorMap" : "specularMap";

    const specularBase = this.data.maps[specularMapName]
      ? mul(p.specularColor, sampleMap(specularMapName, attributes))
      : p.specularColor;

    const specularFactor = pbr ? dielectric * p.specularIntensity
      * (this.data.maps.specularIntensityMap ? sampleMap("specularIntensityMap", attributes)[3] : 1) : 1;

    const specular = specularFactor === 1 ? specularBase : scale(specularBase, specularFactor);
    const f0 = mix(specular, color, metalness);
    const range = p.iridescenceThicknessRange;

    const filmThickness = this.data.maps.iridescenceThicknessMap
      ? range[0] + (range[1] - range[0]) * sampleMap("iridescenceThicknessMap", attributes)[1] : range[1];

    const iri = p.iridescence && filmThickness > 0
      ? clamp(p.iridescence * sampleMap("iridescenceMap", attributes)[0]) : 0;

    // As in Three, the film's Fresnel is evaluated once at the view angle and blended over every specular lobe.
    const film = iri > 0 ? mix(iridescentFresnel(p.iridescenceIOR, nv, filmThickness, specular),
      iridescentFresnel(p.iridescenceIOR, nv, filmThickness, color), metalness) : null;

    const fresnel = (vh: number) => {
      const t = 1 - vh, k = t * t * t * t * t;
      const f = [f0[0] + (1 - f0[0]) * k, f0[1] + (1 - f0[1]) * k, f0[2] + (1 - f0[2]) * k];

      return film ? mix(f, film, iri) : f;
    };

    // The material's AO map and ray-traced occlusion combine; both only affect indirect light, as in Three.
    const ao = (this.data.maps.aoMap ? 1 + (sampleMap("aoMap", attributes)[0] - 1) * p.aoMapIntensity : 1)
      * ctx.occlusion(s.position, geometricNormal);

    // Diffuse and specular accumulate separately: Three's transmission replaces only the diffuse term.
    const diffuseAlbedo = pbr && ctx.indirect ? mul(color, f0.map(f => 1 - clamp(f))) : color;
    if (ctx.indirect && !ctx.transport && metalness < 1 && color.some(c => c > 0))
      indirectRaw = ctx.indirect(s.position, n);

    // As in Three (r16x on), the scene environment adds diffuse light to Standard, Physical, Lambert and Phong, but
    // not to Toon; only Standard and Physical reflect it (below).
    let indirect = mul(diffuseAlbedo, ctx.indirect
      ? (indirectRaw ?? (metalness < 1 && color.some(c => c > 0) ? ctx.indirect(s.position, n) : [0, 0, 0]))
      : type === "MeshToonMaterial" ? ZERO3 : env(n, 1, true));

    if (this.data.maps.lightMap)
      indirect = add(indirect, scale(mul(color, sampleMap("lightMap", attributes)), p.lightMapIntensity));

    for (const light of ctx.indirect ? [] : ctx.lights) {
      if (light.type !== "AmbientLight" && light.type !== "HemisphereLight")
        continue;

      const ambient = light.type === "AmbientLight" ? light.color
        : mix(light.groundColor, light.color, dot(n, normalize(light.position)) * .5 + .5);

      // As in Three, ambient irradiance goes through the Lambert BRDF (1/pi).
      indirect = add(indirect, scale(mul(color, ambient), light.intensity / Math.PI));
    }

    // Sheen, as in Three: its lobe sits on top and takes back the energy it reflects from what lies below.
    const sheen = p.sheen;
    const sheenColor = sheen ? scale(mul(p.sheenColor, sampleMap("sheenColorMap", attributes)), sheen) : ZERO3;

    const sheenRoughness = clamp(p.sheenRoughness, .0001, 1)
      * (sheen ? sampleMap("sheenRoughnessMap", attributes)[3] : 1);

    const sheenPeak = Math.max(...sheenColor), sheenView = sheenPeak ? sheenAlbedo(nv, sheenRoughness) : 0;
    const sheenIndirect = 1 - sheenPeak * sheenView;
    let diffuse = scale(indirect, (1 - metalness) * ao * sheenIndirect);
    if (indirectRaw)
      modulation = scale(diffuseAlbedo, (1 - metalness) * ao * sheenIndirect);
    let specularOut: number[] = emissive;

    if (!ctx.transport && pbr) {
      // The lobe's albedo scales the reflection (the film's Fresnel blends in as in Three); aoMap occlusion fades it.
      const [scaleF0, bias] = environmentBrdf(roughness, nv);
      const albedo = f0.map(f => f * scaleF0 + bias), reflectance = film ? mix(albedo, film, iri) : albedo;
      specularOut = add(specularOut, scale(mul(env(reflection, roughness), reflectance),
        specularOcclusion(nv, ao, roughness) * sheenIndirect));
      // What the lobe reflects the diffuse layer below does not get, as in Three's multiscattering split.
      if (!ctx.indirect)
        diffuse = mul(diffuse, albedo.map(a => 1 - clamp(a)));
    }

    if (sheenPeak && !ctx.transport)
      specularOut = add(specularOut, scale(mul(env(n, 1, true), sheenColor), sheenView * ao));
    const coat = p.clearcoat ? p.clearcoat * sampleMap("clearcoatMap", attributes)[0] : 0;

    const coatRoughness = Math.max(.0525,
      coat ? p.clearcoatRoughness * sampleMap("clearcoatRoughnessMap", attributes)[1] : p.clearcoatRoughness);

    // Three's clearcoat ignores the base normal map: it has its own, over the interpolated normal.
    let coatN = geometricNormal;

    if (coat && this.data.maps.clearcoatNormalMap) {
      const tex = sampleMap("clearcoatNormalMap", attributes), ns = p.clearcoatNormalScale;

      coatN = normalize(add(
        add(scale(s.tangent, (tex[0] * 2 - 1) * ns[0]), scale(s.bitangent, (tex[1] * 2 - 1) * ns[1])),
        scale(geometricNormal, tex[2] * 2 - 1)));
    }

    const coatNv = Math.max(.0001, dot(coatN, v));

    if (coat) {
      const coatFresnel = coat * (.04 + .96 * (1 - coatNv) ** 5);
      diffuse = scale(diffuse, 1 - coatFresnel);
      if (modulation)
        modulation = scale(modulation, 1 - coatFresnel);
      // The coat's own reflection, like Three's EnvironmentBRDF over F0 = 0.04; the layers below lose its Fresnel.
      const [coatScale, coatBias] = environmentBrdf(coatRoughness, coatNv);
      if (!ctx.transport)
        specularOut = add(scale(specularOut, 1 - coatFresnel), scale(env(reflect(scale(v, -1), coatN), coatRoughness),
          coat * (.04 * coatScale + coatBias) * ao));
    }

    const toon = type === "MeshToonMaterial", gradient = toon ? this.textures.get(this.data.maps.gradientMap) : undefined;
    let shadow = 0;

    for (const light of ctx.lights) {
      if (light.type === "AmbientLight" || light.type === "HemisphereLight")
        continue;
      const constant = lightConstants(light);
      let l: number[], distance: number, power = light.intensity, tint: number[] | null = null;

      if (light.type === "DirectionalLight") {
        l = constant.direction;
        distance = Infinity;
      } else {
        l = sub(light.position, s.position); distance = length(l);
        power /= Math.max(distance ** light.decay, .01);
        if (light.distance > 0)
          power *= Math.max(0, 1 - (distance / light.distance) ** 4) ** 2;

        if (light.type === "SpotLight") {
          const cosine = dot(constant.direction, l) / distance;
          const outer = constant.outer, inner = constant.inner;
          const t = inner === outer ? Number(cosine >= outer) : clamp((cosine - outer) / (inner - outer));
          power *= t * t * (3 - 2 * t);
          const projected = light.map && light.mapMatrix ? this.textures.get(light.map) : undefined;

          if (projected && power > 0) {
            // Three's spotLightMap: inside the projection frustum the texture tints the light.
            const c = transform4(light.mapMatrix!, s.position), x = c[0] / c[3], y = c[1] / c[3], z = c[2] / c[3];
            if (c[3] > 0 && x > 0 && x < 1 && y > 0 && y < 1 && z > 0 && z < 1)
              tint = sampleTexture(projected, [x, y]);
          }
        }

        if (light.type === "RectAreaLight") {
          const m = light.matrixWorld;
          power *= light.width * light.height / Math.max(distance * distance, .01)
            * Math.max(0, dot(normalize(l), [m[8], m[9], m[10]]));
        }
      }

      if (light.type !== "DirectionalLight") l = normalize(l);
      let nl = Math.max(0, dot(n, l));
      // Toon lighting reads its ramp across the whole sphere of directions, the side facing away included.
      if ((nl <= 0 && !toon) || power <= 0)
        continue;
      // Toon lights the side facing away too. Three's shadow map (drawn from back faces) leaves a mesh's own far side
      // unshadowed there; tracing through the mesh instead would stair-step the terminator along its facets.
      const visibility = light.castShadow ? ctx.visibility(s.position, n, l, distance, toon && dot(n, l) < .1) : 1;
      shadow = Math.max(shadow, 1 - visibility);
      power *= visibility;

      if (toon) {
        // Three's getGradientIrradiance: the ramp spans dot(n, l) from -1 to 1; without one, a single step.
        const coord = dot(n, l) * .5 + .5;
        nl = gradient ? sampleTexture(gradient, [coord, .5], false)[0] : coord >= .7 ? 1 : .7;
      }

      let diffuseBrdf = scale(color, (1 - metalness) / Math.PI), specularBrdf: number[] = [0, 0, 0];
      const h = normalize(add(l, v)), nh = Math.max(0, dot(n, h)), vh = Math.max(0, dot(v, h));

      if (pbr && ctx.transport) {
        // Diffuse transport: only the diffuse part counts, the bounce's specular term is left out of the estimate.
        diffuseBrdf = mul(diffuseBrdf, fresnel(vh).map(x => 1 - x));
      } else if (pbr) {
        const a = roughness * roughness, a2 = a * a, denominator = nh * nh * (a2 - 1) + 1;
        let distribution = a2 / (Math.PI * denominator * denominator);
        const anisotropy = p.anisotropy ? p.anisotropy * sampleMap("anisotropyMap", attributes)[2] : 0;

        if (anisotropy > 0) {
          const angle = p.anisotropyRotation;
          const tangent = add(scale(s.tangent, Math.cos(angle)), scale(s.bitangent, Math.sin(angle)));
          const bitangent = cross(n, tangent);
          const at = Math.max(a, .001) + anisotropy * anisotropy * (1 - a), ab = Math.max(a, .001);
          const d = (dot(h, tangent) / at) ** 2 + (dot(h, bitangent) / ab) ** 2 + nh * nh;
          distribution = 1 / (Math.PI * at * ab * d * d);
        }

        const visibilityTerm = .5 / (
          nl * Math.sqrt(nv * nv * (1 - a2) + a2) + nv * Math.sqrt(nl * nl * (1 - a2) + a2) + 1e-7
        );

        const f = fresnel(vh);
        diffuseBrdf = mul(diffuseBrdf, f.map(x => 1 - x));
        specularBrdf = scale(f, distribution * visibilityTerm);
      } else if (type === "MeshPhongMaterial" && !ctx.transport) {
        // Three's D_BlinnPhong: (shininess/2 + 1)/pi, with Schlick Fresnel over the specular color.
        const shininess = p.shininess;
        specularBrdf = scale(fresnel(vh), (shininess * .5 + 1) / Math.PI * nh ** shininess);
      }

      if (sheenPeak && !ctx.transport) {
        const direct = 1 - sheenPeak * Math.max(sheenView, sheenAlbedo(nl, sheenRoughness));
        diffuseBrdf = scale(diffuseBrdf, direct);
        specularBrdf = add(scale(specularBrdf, direct), scale(sheenColor, sheenLobe(sheenRoughness, nh, nv, nl)));
      }

      if (coat > 0) {
        const coatNl = Math.max(0, dot(coatN, l)), coatNh = Math.max(0, dot(coatN, h));
        const a2 = coatRoughness ** 4, d = coatNh * coatNh * (a2 - 1) + 1, f = .04 + .96 * (1 - vh) ** 5;
        diffuseBrdf = scale(diffuseBrdf, 1 - coat * f);
        // The lobe is lit over the coat's own normal; `nl` is divided out because the sum below multiplies by it.
        if (!ctx.transport)
          specularBrdf = add(scale(specularBrdf, 1 - coat * f), scale([1, 1, 1],
            coat * f * a2 / (Math.PI * d * d * Math.max(4 * coatNv * coatNl, .001)) * coatNl / nl));
      }

      const lightColor = tint ? mul(light.color, tint) : light.color;
      diffuse = add(diffuse, scale(mul(diffuseBrdf, lightColor), power * nl));
      if (!ctx.transport)
        specularOut = add(specularOut, scale(mul(specularBrdf, lightColor), power * nl));
    }

    if (type === "ShadowMaterial")
      return {
        color,
        alpha: shadow * p.opacity,
        depth: s.depth,
        transmission: 0,
        attenuation: [1, 1, 1],
        diffuse: color,
      };

    const transmission = p.transmission
      ? clamp(p.transmission * sampleMap("transmissionMap", attributes)[0]) * (1 - metalness) : 0;

    if (ctx.transport === "diffuse")
      return finish(add(scale(diffuse, 1 - transmission), emissive));

    if (legacyEnv) {
      // The blend rewrites the lit color as a whole: the denoiser's indirect split no longer applies.
      indirectRaw = modulation = undefined;

      return finish(legacyEnv(add(diffuse, specularOut)));
    }

    if (!transmission)
      return finish(add(diffuse, specularOut), 0, ONES3, diffuse);
    const thickness = p.thickness * sampleMap("thicknessMap", attributes)[1];
    const attenuationDistance = p.attenuationDistance;
    // Like getIBLVolumeRefraction: the background passes through tinted by base color, volume absorption and (1 - F).
    const surfaceFresnel = fresnel(nv);

    const attenuation = p.attenuationColor.map((x, i) =>
      color[i] * (1 - surfaceFresnel[i]) * (attenuationDistance === Infinity ? 1
        : Math.exp(Math.log(Math.max(x, 1e-6)) * thickness / Math.max(attenuationDistance, 1e-6))));

    // Like getVolumeTransmissionRay: the view ray bends at the surface and leaves the volume `thickness` further on.
    const lookup = (ior: number) => ctx.transmitted?.(
      add(s.position, scale(normalize(refract(scale(v, -1), n, 1 / ior)), thickness)), roughness * clamp(ior * 2 - 2));

    let transmitted = p.dispersion > 0 ? undefined : lookup(p.ior);

    if (p.dispersion > 0) {
      // Three's dispersion: each channel refracts with its own ior, spread around the material's.
      const halfSpread = (p.ior - 1) * .025 * p.dispersion;
      const channels = [p.ior - halfSpread, p.ior, p.ior + halfSpread].map(lookup);
      if (channels.every(Boolean))
        transmitted = channels.map((rgb, i) => rgb![i]);
    }

    return finish(add(diffuse, specularOut), transmission, attenuation, diffuse, transmitted);
  }
}
