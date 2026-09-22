import * as THREE from "three";
import type { RenderSettingsInput } from "@tinypoly/cpu-renderer";
import type { SceneId } from "./catalog";

export interface ExampleScene {
  scene: THREE.Scene;
  cameraPosition: [number, number, number];
  target: [number, number, number];
  /** Closed off from the sky: the WebGL preview skips environment lighting. */
  enclosed?: boolean;
  /** Render settings the scene is best seen with; leaving the scene puts the viewer's own back. */
  settings?: RenderSettingsInput;
}

/** A custom GLSL material: the renderer interprets it on the CPU like any Three shader. */
const stripes = new THREE.ShaderMaterial({
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      float stripe = step(0.5, fract(vUv.y * 12.0));
      gl_FragColor = vec4(mix(vec3(0.9, 0.5, 0.05), vec3(0.08), stripe), 1.0);
      // The color above is linear light: tone map and encode it like the built-in materials around it.
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }
  `,
});

function shadowed<T extends THREE.Mesh>(mesh: T): T {
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return mesh;
}

/** Procedural textures keep the example free of image files; `pixel` returns RGBA bytes for each texel. */
function dataTexture(width: number, height: number, pixel: (x: number, y: number) => number[]): THREE.DataTexture {
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) data.set(pixel(x, y), (y * width + x) * 4);
  const texture = new THREE.DataTexture(data, width, height);
  texture.needsUpdate = true;

  return texture;
}

const srgb = (hex: number) => [hex >> 16 & 255, hex >> 8 & 255, hex & 255, 255];

/** Two-color checkerboard: one repeat covers 2 × 2 squares. */
function checker(a: number, b: number, repeat: number, repeatY = repeat) {
  const texture = dataTexture(2, 2, (x, y) => srgb((x + y) % 2 ? b : a));
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeat, repeatY);

  return texture;
}

/**
 * A photo studio sweep: the floor runs back from the camera and curves up into the wall, so no horizon or floor
 * edge shows behind the subject. `back` is where the wall stands; the curve starts `radius` in front of it.
 */
function cyclorama(material: THREE.Material, back = -3.5, { width = 30, radius = 2.5, height = 9, front = 12 } = {}) {
  const profile: [number, number][] = [[0, front], [0, back + radius]];

  for (let i = 1; i <= 24; i++) {
    const angle = i / 24 * Math.PI / 2;
    profile.push([radius - Math.cos(angle) * radius, back + radius - Math.sin(angle) * radius]);
  }

  profile.push([height, back]);
  const positions: number[] = [], uvs: number[] = [], indices: number[] = [];
  let run = 0;

  profile.forEach(([y, z], i) => {
    if (i) run += Math.hypot(y - profile[i - 1][0], z - profile[i - 1][1]);
    positions.push(-width / 2, y, z, width / 2, y, z);
    uvs.push(0, run, width, run);
    // Counterclockwise seen from above the floor and in front of the wall.
    if (i) indices.push(2 * i - 2, 2 * i - 1, 2 * i, 2 * i - 1, 2 * i + 1, 2 * i);
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = true;

  return mesh;
}

/** The studio sweep most scenes stand on: a neutral gray, matte enough to show soft shadows. */
const sweep = (back?: number) => cyclorama(new THREE.MeshStandardMaterial({ color: 0x8e9196, roughness: .92 }), back);

/**
 * Fractal value noise that tiles every `width` × `height` texels, so textures built from it repeat without seams.
 * `cells` counts lattice cells along the height; the width gets as many more as its aspect asks. Returns roughly [0, 1].
 */
function tilingNoise(seed: number, width: number, height = width) {
  const next = random(seed), lattice = Array.from({ length: 256 * 256 }, () => next());
  const wrap = (i: number, cells: number) => ((i % cells) + cells) % cells;

  const octave = (x: number, y: number, cells: number) => {
    const cellsX = Math.min(256, Math.round(cells * width / height));
    const fx = x / width * cellsX, fy = y / height * cells, ix = Math.floor(fx), iy = Math.floor(fy);
    const tx = fx - ix, ty = fy - iy, sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
    const at = (i: number, j: number) => lattice[wrap(j, cells) * 256 + wrap(i, cellsX)];
    const top = at(ix, iy) + (at(ix + 1, iy) - at(ix, iy)) * sx;
    const bottom = at(ix, iy + 1) + (at(ix + 1, iy + 1) - at(ix, iy + 1)) * sx;

    return top + (bottom - top) * sy;
  };

  return (x: number, y: number, cells = 4, octaves = 4) => {
    let value = 0, amplitude = .5, total = 0;

    for (let i = 0; i < octaves; i++, cells *= 2, amplitude *= .5) {
      value += octave(x, y, Math.min(cells, 256)) * amplitude;
      total += amplitude;
    }

    return value / total;
  };
}

function floor(material: THREE.Material, size = 12) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;

  return mesh;
}

function sunlight(scene: THREE.Scene, position: [number, number, number], intensity = 2.2) {
  const sun = new THREE.DirectionalLight(0xffeedd, intensity);
  sun.position.set(...position);
  sun.castShadow = true;
  scene.add(sun);
}

/** Physical, metallic and shader materials on a floor, lit by the sun and the environment. */
export function studioScene(): ExampleScene {
  const scene = new THREE.Scene();

  const metal = new THREE.MeshPhysicalMaterial({ color: 0xe4573d, roughness: .25, metalness: .7, clearcoat: 1 });
  const plastic = new THREE.MeshPhysicalMaterial({ color: 0x3a8ee6, roughness: .5, clearcoat: 1 });
  const glass = new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: .05, transmission: 1, thickness: .5 });

  const spheres: [THREE.Material, number, number][] = [[metal, -1.35, 1.8], [plastic, 0, 0], [stripes, 1.35, -1.8]];

  for (const [material, x, z] of spheres) {
    const sphere = shadowed(new THREE.Mesh(new THREE.SphereGeometry(.55, 48, 24), material));
    sphere.position.set(x, .55, z);
    scene.add(sphere);
  }

  const knot = shadowed(new THREE.Mesh(new THREE.TorusKnotGeometry(.32, .11, 160, 20), glass));
  knot.position.set(1.4, .55, 1.2);
  scene.add(knot);

  scene.add(sweep(-4.5));
  sunlight(scene, [-3, 6, 4]);

  return { scene, cameraPosition: [3.4, 2.2, 5.2], target: [0, .55, 0] };
}

/** A closed room lit only from the ceiling, with global illumination on: the red and green walls bleed color. */
export function roomScene(): ExampleScene {
  const scene = new THREE.Scene();
  const matte = (color: number) => new THREE.MeshLambertMaterial({ color });

  const walls: [[number, number, number], [number, number, number], number][] = [
    [[0, 0, 0], [-Math.PI / 2, 0, 0], 0xc8c8c8],
    [[0, 4, 0], [Math.PI / 2, 0, 0], 0xc8c8c8],
    [[0, 2, -2], [0, 0, 0], 0xc8c8c8],
    [[-2, 2, 0], [0, Math.PI / 2, 0], 0xc52520],
    [[2, 2, 0], [0, -Math.PI / 2, 0], 0x249b48],
  ];

  for (const [position, rotation, color] of walls) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), matte(color));
    wall.position.set(...position);
    wall.rotation.set(...rotation);
    wall.receiveShadow = true;
    scene.add(wall);
  }

  const emitter = new THREE.Mesh(
    new THREE.PlaneGeometry(1.5, 1.5),
    new THREE.MeshStandardMaterial({ color: 0, emissive: 0xffffff, emissiveIntensity: 12 }),
  );

  emitter.position.set(0, 3.98, -.3);
  emitter.rotation.x = Math.PI / 2;
  scene.add(emitter);

  for (const [x, height, z, angle] of [[-.85, .65, .2, .3], [.8, 1.1, -.6, -.3]]) {
    const box = shadowed(new THREE.Mesh(new THREE.BoxGeometry(1.1, height * 2, 1.1), matte(0xc8c8c8)));
    box.position.set(x, height, z);
    box.rotation.y = angle;
    scene.add(box);
  }

  const lamp = new THREE.PointLight(0xffffff, 7, 0, 2);
  lamp.position.set(0, 3.7, -.3);
  lamp.castShadow = true;
  scene.add(lamp);

  return { scene, cameraPosition: [0, 2, 6.6], target: [0, 1.9, 0], enclosed: true,
    settings: { globalIllumination: { enabled: true, denoise: true } } };
}

/** Refraction through several kinds of glass, over a checkerboard and colored columns that show the bending. */
export function glassScene(): ExampleScene {
  const scene = new THREE.Scene();

  // A checkered mat on the sweep: something with edges for the glass to bend.
  const mat = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 4), new THREE.MeshStandardMaterial({
    map: checker(0xece6da, 0x33404c, 8, 5), roughness: .75 }));

  mat.rotation.x = -Math.PI / 2;
  mat.position.set(0, .002, -.4);
  mat.receiveShadow = true;
  scene.add(sweep(-5), mat);

  const columns: [number, number][] = [[0xd8452e, -2.1], [0xe8b83a, -.7], [0x2f7fd6, .7], [0x3aa55c, 2.1]];

  for (const [color, x] of columns) {
    const column = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(.22, .22, 2.4, 32),
      new THREE.MeshStandardMaterial({ color, roughness: .45 })));

    column.position.set(x, 1.2, -1.6);
    scene.add(column);
  }

  const glass = (parameters: THREE.MeshPhysicalMaterialParameters) =>
    new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0, transmission: 1, ior: 1.5, ...parameters });

  const pieces: [THREE.BufferGeometry, THREE.Material, [number, number, number], number][] = [
    // Clear glass: a thick ball inverts what lies behind it.
    [new THREE.SphereGeometry(.5, 64, 32), glass({ thickness: 1 }), [-2.1, .5, .3], 0],
    // A gem: high ior and dispersion split the columns into colored fringes.
    [new THREE.IcosahedronGeometry(.55, 0), glass({ ior: 2.3, thickness: 1, dispersion: 6, flatShading: true }),
      [-.7, .55, .3], .4],
    // Frosted: roughness blurs the scene behind.
    [new THREE.BoxGeometry(.8, .8, .8), glass({ roughness: .35, thickness: .8 }), [.7, .4, .3], .6],
    // Tinted: volume absorption deepens the amber where the glass is thicker.
    [new THREE.TorusGeometry(.38, .16, 32, 96), glass({ thickness: .6, attenuationColor: 0xff8a2a,
      attenuationDistance: .35 }), [2.1, .56, .3], 0],
  ];

  for (const [geometry, material, position, angle] of pieces) {
    const piece = shadowed(new THREE.Mesh(geometry, material));
    piece.position.set(...position);
    piece.rotation.set(angle * .5, angle, 0);
    scene.add(piece);
  }

  sunlight(scene, [-3, 6, 4]);

  return { scene, cameraPosition: [0, 1.7, 5.6], target: [0, .55, 0] };
}

/** One sphere per shading model and Physical lobe, classic materials in front and Physical ones behind. */
export function materialsScene(): ExampleScene {
  const scene = new THREE.Scene();
  scene.add(sweep(-3.5));

  // An equirectangular studio for the classic envMap: a gradient with two bright softboxes.
  const studio = dataTexture(64, 32, (x, y) => {
    const t = y / 31, softbox = y > 18 && y < 26 && (x % 32 > 6 && x % 32 < 14);

    return softbox ? [255, 250, 240, 255] : [40 + 120 * t, 44 + 130 * t, 52 + 150 * t, 255].map(Math.round);
  });

  studio.colorSpace = THREE.SRGBColorSpace;
  studio.mapping = THREE.EquirectangularReflectionMapping;
  studio.magFilter = THREE.LinearFilter;
  const toonRamp = dataTexture(3, 1, x => [[70, 160, 255][x], [70, 160, 255][x], [70, 160, 255][x], 255]);

  // A tangent-space normal map of small ripples: orange peel on the clear coat only.
  const ripples = dataTexture(64, 64, (x, y) => {
    const u = x / 64 * 2 * Math.PI, v = y / 64 * 2 * Math.PI;
    const dx = .35 * Math.cos(u * 3) * Math.cos(v * 2), dy = -.35 * Math.sin(u * 3) * Math.sin(v * 2) * 1.5;
    const length = Math.hypot(dx, dy, 1);

    return [(-dx / length * .5 + .5) * 255, (-dy / length * .5 + .5) * 255, (1 / length * .5 + .5) * 255, 255]
      .map(Math.round);
  });

  ripples.wrapS = ripples.wrapT = THREE.RepeatWrapping;
  ripples.repeat.set(4, 2);
  ripples.magFilter = THREE.LinearFilter;

  const front: THREE.Material[] = [
    new THREE.MeshLambertMaterial({ color: 0xc0623b }),
    new THREE.MeshPhongMaterial({ color: 0x3f6fc4, shininess: 90, specular: 0x555555 }),
    new THREE.MeshToonMaterial({ color: 0x52b36b, gradientMap: toonRamp }),
    // Classic chrome: the envMap replaces most of the color (combine = MixOperation).
    new THREE.MeshPhongMaterial({ color: 0x222222, envMap: studio, combine: THREE.MixOperation, reflectivity: .85 }),
    new THREE.MeshStandardMaterial({ color: 0xe0a93c, metalness: 1, roughness: .35 }),
  ];

  const back: THREE.Material[] = [
    new THREE.MeshPhysicalMaterial({ color: 0xa31621, roughness: .5, clearcoat: 1, clearcoatRoughness: .05,
      clearcoatNormalMap: ripples, clearcoatNormalScale: new THREE.Vector2(.6, .6) }),
    new THREE.MeshPhysicalMaterial({ color: 0x2a1840, roughness: .9, sheen: 1, sheenColor: 0xd28bff,
      sheenRoughness: .4 }),
    new THREE.MeshPhysicalMaterial({ color: 0x111111, metalness: 1, roughness: .15, iridescence: 1,
      iridescenceIOR: 1.6, iridescenceThicknessRange: [200, 600] }),
    new THREE.MeshPhysicalMaterial({ color: 0xc8ccd2, metalness: 1, roughness: .3, anisotropy: .9 }),
    new THREE.MeshPhysicalMaterial({ color: 0xf2f2f2, metalness: 1, roughness: .04 }),
  ];

  for (const [row, z] of [[front, .9], [back, -.9]] as const)
    row.forEach((material, i) => {
      const sphere = shadowed(new THREE.Mesh(new THREE.SphereGeometry(.5, 64, 32), material));
      sphere.position.set((i - 2) * 1.3, .5, z);
      scene.add(sphere);
    });

  sunlight(scene, [-4, 7, 5], 2.6);

  return { scene, cameraPosition: [0, 2.8, 6.8], target: [0, .45, 0] };
}

/**
 * A closed hall in fog, lit by two spot lights: one projects a stained-glass pattern through its `map`.
 * Nothing reaches it from the sky, so its materials ignore the environment, like the WebGL preview does here.
 */
export function hallScene(): ExampleScene {
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x1e2430, .07);

  const surface = (color: number, roughness = .9, metalness = 0) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness, envMapIntensity: 0 });

  const room: [[number, number, number], [number, number, number], number][] = [
    [[0, 0, 0], [-Math.PI / 2, 0, 0], 0x4a4a4c],
    [[0, 3.6, 0], [Math.PI / 2, 0, 0], 0x2a2e36],
    [[0, 1.8, -3.5], [0, 0, 0], 0x5b6170],
    [[0, 1.8, 3.5], [0, Math.PI, 0], 0x2a2e36],
    [[-3.5, 1.8, 0], [0, Math.PI / 2, 0], 0x3a404c],
    [[3.5, 1.8, 0], [0, -Math.PI / 2, 0], 0x3a404c],
  ];

  for (const [position, rotation, color] of room) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(7, 7), surface(color));
    wall.position.set(...position);
    wall.rotation.set(...rotation);
    wall.receiveShadow = true;
    scene.add(wall);
  }

  for (const x of [-1.3, 1.3]) {
    const pedestal = shadowed(new THREE.Mesh(new THREE.BoxGeometry(.8, 1, .8), surface(0xd9d4c8, .6)));
    pedestal.position.set(x, .5, -1.4);
    scene.add(pedestal);
  }

  const knot = shadowed(new THREE.Mesh(new THREE.TorusKnotGeometry(.3, .1, 160, 20), surface(0xd6a24a, .25, 1)));
  knot.position.set(-1.3, 1.5, -1.4);
  scene.add(knot);

  const bust = shadowed(new THREE.Mesh(new THREE.SphereGeometry(.4, 64, 32), new THREE.MeshPhysicalMaterial({
    color: 0xf3efe6, roughness: .5, clearcoat: .6, clearcoatRoughness: .2, envMapIntensity: 0 })));

  bust.position.set(1.3, 1.4, -1.4);
  scene.add(bust);

  // Leaded glass: colored panes in a dark grid, read with nearest filtering so the leads stay sharp.
  const panes = [0xc23b3b, 0xe0a33a, 0x3a78c2, 0x3f9e5a, 0x8a4fb8, 0xe8e2c8];

  const window = dataTexture(16, 16, (x, y) =>
    x % 4 === 0 || y % 4 === 0 ? [18, 18, 18, 255] : srgb(panes[(Math.floor(x / 4) * 7 + Math.floor(y / 4) * 3) % 6]));

  window.colorSpace = THREE.SRGBColorSpace;

  const spot = (color: number, intensity: number, position: [number, number, number],
    target: [number, number, number], angle: number, map?: THREE.Texture) => {
    const light = new THREE.SpotLight(color, intensity, 0, angle, .3, 2);
    light.position.set(...position);
    light.target.position.set(...target);
    light.castShadow = true;
    if (map) light.map = map;
    scene.add(light, light.target);
  };

  spot(0xffffff, 170, [2.2, 3.4, 2.4], [-.4, 1.2, -3.5], .42, window);
  spot(0xffc58a, 60, [-1.3, 3.5, .4], [-1.3, 1.4, -1.4], .3);
  scene.add(new THREE.HemisphereLight(0x6a7890, 0x1a1a1a, 1.2));

  return { scene, cameraPosition: [0, 1.6, 3.2], target: [0, 1.2, -1.4], enclosed: true };
}

/** Deterministic random numbers (mulberry32), so every run builds the same scene. */
function random(seed: number) {
  return () => {
    seed = seed + 0x6d2b79f5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;

    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

interface SurfaceTexel { color: number[]; height: number; roughness: number; metalness?: number }

/**
 * A color, normal and roughness set from one height field: `sample` returns the texel's sRGB color, its height, its
 * roughness and, for metals, its metalness. Normals come from the height's slope, in tangent space. Roughness sits in
 * the green channel and metalness in the blue one, as Three reads them, so the same map can serve as both.
 */
function surfaceMaps(width: number, height: number, repeat: [number, number], strength: number,
  sample: (x: number, y: number) => SurfaceTexel) {
  const texels = Array.from({ length: width * height }, (_, i) => sample(i % width, Math.floor(i / width)));
  const at = (x: number, y: number) => texels[((y + height) % height) * width + (x + width) % width];
  const map = dataTexture(width, height, (x, y) => [...at(x, y).color.map(Math.round), 255]);

  const normalMap = dataTexture(width, height, (x, y) => {
    const dx = (at(x + 1, y).height - at(x - 1, y).height) * strength;
    const dy = (at(x, y + 1).height - at(x, y - 1).height) * strength, length = Math.hypot(dx, dy, 1);

    return [(-dx / length * .5 + .5) * 255, (-dy / length * .5 + .5) * 255, (1 / length * .5 + .5) * 255, 255]
      .map(Math.round);
  });

  const roughnessMap = dataTexture(width, height, (x, y) => {
    const { roughness, metalness = 0 } = at(x, y);

    return [255, Math.round(roughness * 255), Math.round(metalness * 255), 255];
  });

  map.colorSpace = THREE.SRGBColorSpace;

  for (const texture of [map, normalMap, roughnessMap]) {
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(...repeat);
    texture.magFilter = texture.minFilter = THREE.LinearFilter;
  }

  return { map, normalMap, roughnessMap };
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

/**
 * Bricks at their real size for a wall `width` × `height` units (meters) across: 22.5 × 7.5 cm with the mortar, in
 * running bond. One repeat holds sixteen courses of eight, each with its own tone, chipped edges and a pitted face; a
 * few came out of the kiln darker. The mortar sits back and stays rough.
 */
function brickMaps(width: number, height: number) {
  const shades = Array.from({ length: 256 }, random(7)), stone = tilingNoise(3, 768, 512);

  return surfaceMaps(768, 512, [width / 1.8, height / 1.2], 4, (x, y) => {
    const course = Math.floor(y / 32), shifted = (x + (course % 2) * 48) % 768;
    const along = shifted % 96, index = course * 8 + Math.floor(shifted / 96);
    const edge = Math.min(along, 96 - along, y % 32, 32 - y % 32) + (stone(x, y, 16, 3) - .5) * 3;
    const face = clamp01((edge - 2.2) / 2), pits = stone(x, y, 64, 2), grime = stone(x, y, 3, 3);
    const tone = shades[index], burnt = shades[(index * 7 + 3) % 256] > .86 ? .74 : 1, dirt = .82 + grime * .3;

    if (face === 0) {
      const sand = 150 + pits * 34;

      return { color: [sand, sand * .96, sand * .9].map(c => c * dirt), height: 0, roughness: 1 };
    }

    const speck = pits < .32 ? .78 : 1, fire = burnt * speck * dirt;

    return {
      color: [(138 + tone * 48) * fire, (60 + tone * 24) * fire, (44 + tone * 14) * fire],
      height: face * (.85 + pits * .15) - (pits < .32 ? .1 : 0),
      roughness: .78 + pits * .2,
    };
  });
}

/**
 * Flat-sawn oak: growth rings run along the board, bent by `noise`, with fine dark pores. `x` crosses the grain and
 * `y` follows it; `board` picks the board's tone.
 */
function oak(noise: ReturnType<typeof tilingNoise>, shades: number[], x: number, y: number, board: number) {
  const tone = shades[board % shades.length];
  const figure = (x / 64 * 4 + noise(x, y, 2, 2) * 1.6 + tone * 5) % 1;
  const late = Math.abs(figure - .5) * 2, pores = noise(x * 8, y, 64, 2), streaks = noise(x * 4, y, 4, 2);
  const dark = late ** 6 * .45 + pores * .25 + streaks * .2;

  return [128 + tone * 34 - dark * 62, 84 + tone * 22 - dark * 44, 50 + tone * 12 - dark * 28];
}

/** Floorboards 14 cm wide and 2.24 m long for a floor `width` × `depth` units across, with a groove between boards. */
function plankMaps(width: number, depth: number) {
  const shades = Array.from({ length: 64 }, random(11)), grain = tilingNoise(11, 512, 1024);

  return surfaceMaps(512, 1024, [width / 1.12, depth / 2.24], 3, (x, y) => {
    const plank = Math.floor(x / 64), across = x % 64, joint = Math.floor(shades[plank + 32] * 1024);
    const end = (y - joint + 1024) % 1024, groove = across < 2 || across > 61 || end < 2 || end > 1021;
    if (groove)
      return { color: [40, 28, 18], height: 0, roughness: .9 };

    return { color: oak(grain, shades, x, y, plank), height: 1, roughness: .42 + grain(x * 3, y, 32, 2) * .3 };
  });
}

/** A shipping crate's face: horizontal slats inside a proud frame, braced corner to corner. */
function crateMaps() {
  const shades = Array.from({ length: 64 }, random(23)), grain = tilingNoise(23, 256);

  return surfaceMaps(256, 256, [1, 1], 4, (x, y) => {
    const frame = Math.min(x, 255 - x, y, 255 - y) < 26, brace = Math.abs(x - y) < 15;
    const slat = Math.floor(y / 51), gap = !frame && !brace && y % 51 < 3;
    if (gap)
      return { color: [30, 22, 15], height: 0, roughness: 1 };
    const proud = frame || brace, stile = frame && (x < 26 || x > 229);
    // The slats and rails run across the face, the stiles up it.
    const color = stile ? oak(grain, shades, x, y, 3) : oak(grain, shades, y, x, proud ? 5 : slat);

    return { color: color.map(c => c * (proud ? 1.08 : .9)), height: proud ? 1 : .55, roughness: .72 };
  });
}

const crateSurface = (() => {
  let maps: ReturnType<typeof crateMaps> | undefined;

  return () => new THREE.MeshStandardMaterial({ ...(maps ??= crateMaps()) });
})();

function crate(size: number, position: [number, number, number], angle: number) {
  const mesh = shadowed(new THREE.Mesh(new THREE.BoxGeometry(size, size, size), crateSurface()));
  mesh.position.set(...position);
  mesh.rotation.y = angle;

  return mesh;
}

/**
 * A painted 200-liter steel drum: two rolling hoops and rolled rims in the normal map, and paint chipped away where
 * it takes knocks, showing bright steel or rust. Metalness rides in the roughness map's blue channel.
 */
function steelDrum(paint: number[], position: [number, number, number], angle = 0) {
  const wear = tilingNoise(31, 512, 256), stain = tilingNoise(37, 512, 256);

  const maps = surfaceMaps(512, 256, [1, 1], 2.5, (x, y) => {
    const v = y / 255, hoop = (c: number) => Math.exp(-(((v - c) / .014) ** 2));
    const hoops = hoop(1 / 3) + hoop(2 / 3), rim = Math.max(0, 1 - Math.min(v, 1 - v) / .025);
    const knocks = hoops * .3 + rim * .35 + (1 - Math.min(1, v / .2)) * .15;
    const chip = wear(x, y, 6, 5) + knocks > .74, rusty = stain(x, y, 8, 4) > .48;
    const relief = hoops + rim;

    if (!chip) {
      const dust = .85 + stain(x, y, 3, 3) * .25;

      return { color: paint.map(c => c * dust), height: relief, roughness: .42 + wear(x, y, 32, 2) * .2 };
    }

    return rusty
      ? { color: [118, 60, 32], height: relief - .12, roughness: .92 }
      : { color: [182, 182, 186], height: relief - .08, roughness: .32, metalness: 1 };
  });

  const side = new THREE.MeshStandardMaterial({ ...maps, metalnessMap: maps.roughnessMap, metalness: 1 });

  const lid = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...paint.map(c => c / 255) as
    [number, number, number], THREE.SRGBColorSpace), roughness: .5 });

  const drum = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(.29, .29, .88, 64), [side, lid, lid]));
  drum.position.set(...position);
  drum.rotation.y = angle;

  return drum;
}

/**
 * A workshop corner in the afternoon sun: brick walls, oak floorboards, crates and a painted steel drum, every one
 * of them color, normal and roughness maps built here, at real scale. A perforated sheet leans on the wall, its
 * holes cut by an alphaMap; the sun comes in low across the bricks so their relief shows.
 */
export function texturesScene(): ExampleScene {
  const scene = new THREE.Scene();

  const back = new THREE.Mesh(new THREE.PlaneGeometry(9, 3.2), new THREE.MeshStandardMaterial({
    ...brickMaps(9, 3.2) }));

  back.position.set(.5, 1.6, -2);

  const side = new THREE.Mesh(new THREE.PlaneGeometry(7, 3.2), new THREE.MeshStandardMaterial({
    ...brickMaps(7, 3.2) }));

  side.position.set(-3.5, 1.6, 1.5);
  side.rotation.y = Math.PI / 2;
  back.receiveShadow = side.receiveShadow = true;
  const boards = floor(new THREE.MeshStandardMaterial({ ...plankMaps(10, 10) }), 10);
  boards.position.set(1, 0, 3);
  scene.add(back, side, boards);

  scene.add(
    crate(.72, [-2.95, .36, -1.45], .12),
    crate(.46, [-2.95, .95, -1.5], -.32),
    crate(.5, [.95, .25, -1.55], -.18),
    steelDrum([34, 82, 138], [-1.95, .44, -1.55], .4),
  );

  // Perforated steel with staggered 3 cm holes on a 4.5 cm pitch: the alphaMap's green channel cuts them, and the
  // sun draws them again in the shadow on the bricks.
  const centers = [[16, 16], [48, 16], [0, 48], [32, 48], [64, 48], [16, 80], [48, 80]];

  const holes = dataTexture(64, 64, (x, y) => {
    const open = centers.some(([cx, cy]) => Math.hypot(x + .5 - cx, y + .5 - cy) < 10.5) ? 0 : 255;

    return [open, open, open, 255];
  });

  holes.wrapS = holes.wrapT = THREE.RepeatWrapping;
  holes.repeat.set(10, 14);
  holes.magFilter = holes.minFilter = THREE.LinearFilter;

  const sheet = shadowed(new THREE.Mesh(new THREE.PlaneGeometry(.9, 1.3), new THREE.MeshStandardMaterial({
    color: 0xb4b9bf, metalness: .85, roughness: .38, alphaMap: holes, alphaTest: .5, side: THREE.DoubleSide })));

  sheet.position.set(-.55, .64, -1.82);
  sheet.rotation.set(-.2, .08, 0);
  scene.add(sheet);
  sunlight(scene, [5, 3.1, .6], 2.8);

  return { scene, cameraPosition: [1.5, 1.5, 3.2], target: [-1, .75, -1.3] };
}

/** Transparent panes over striped paint: normal, additive and multiply blending, and hashed alpha. */
export function blendingScene(): ExampleScene {
  const scene = new THREE.Scene();
  const bars = [0xe8e2d4, 0x2b2f38, 0xd8452e, 0xe8b83a, 0x2f7fd6, 0x3aa55c, 0xe8e2d4, 0x2b2f38];
  const stripes = dataTexture(8, 1, x => srgb(bars[x]));
  stripes.colorSpace = THREE.SRGBColorSpace;
  stripes.wrapS = THREE.RepeatWrapping;
  stripes.repeat.set(2, 1);

  // A striped test board standing on the studio sweep: every pane covers light, dark and saturated paint.
  const board = shadowed(new THREE.Mesh(new THREE.BoxGeometry(7.2, 2.3, .06), new THREE.MeshStandardMaterial({
    map: stripes, roughness: .7 })));

  board.position.set(0, 1.15, -1.2);
  scene.add(board, sweep(-3.2));

  // Transparent materials would cast solid shadow-map shadows in WebGL but none here: the panes cast none at all.
  const panes: [THREE.Material, number][] = [
    [new THREE.MeshStandardMaterial({ color: 0x3a7bd5, roughness: .15, transparent: true, opacity: .45,
      side: THREE.DoubleSide }), -2.25],
    [new THREE.MeshBasicMaterial({ color: 0x2a4a9a, blending: THREE.AdditiveBlending, transparent: true,
      depthWrite: false, side: THREE.DoubleSide }), -.75],
    // Multiply blending needs premultiplied alpha in WebGL; opaque color makes both conventions agree.
    [new THREE.MeshBasicMaterial({ color: 0xffc640, blending: THREE.MultiplyBlending, premultipliedAlpha: true,
      transparent: true, depthWrite: false, side: THREE.DoubleSide }), .75],
    [new THREE.MeshStandardMaterial({ color: 0xe0463a, roughness: .4, alphaHash: true, opacity: .5,
      side: THREE.DoubleSide }), 2.25],
  ];

  for (const [material, x] of panes) {
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.7), material);
    pane.position.set(x, .95, .1);
    pane.receiveShadow = true;
    scene.add(pane);

    const stand = shadowed(new THREE.Mesh(new THREE.BoxGeometry(1.3, .1, .3),
      new THREE.MeshStandardMaterial({ color: 0x3a3d44, roughness: .5, metalness: .6 })));

    stand.position.set(x, .05, .1);
    scene.add(stand);
  }

  sunlight(scene, [-3, 6, 4]);

  return { scene, cameraPosition: [0, 1.5, 6.4], target: [0, .9, 0] };
}

/** Linear output: the renderer and the preview's output pass tone map and encode the final image. */
const OUTPUT = "\n#include <tonemapping_fragment>\n#include <colorspace_fragment>\n";

/** World position and normal for custom shaders that light themselves. */
const WORLD_VERTEX = /* glsl */ `
  varying vec3 vObject;
  varying vec3 vWorld;
  varying vec3 vNormal;
  void main() {
    vObject = position;
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    vNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

/** Noise, Voronoi and a small lighting model shared by the procedural materials below. */
const SHADER_LIBRARY = /* glsl */ `
  uniform vec3 uLight;
  varying vec3 vObject;
  varying vec3 vWorld;
  varying vec3 vNormal;

  float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1) * 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }

  float noise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
      mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
      f.z);
  }

  float fbm(vec3 p) {
    float value = 0.0, amplitude = 0.5;
    for (int octave = 0; octave < 5; octave++) {
      value += amplitude * noise(p);
      p = p * 2.03 + vec3(1.7, 9.2, 4.1);
      amplitude *= 0.5;
    }
    return value;
  }

  // Distance to the nearest cell border of a 3D Voronoi pattern.
  float cracks(vec3 p) {
    vec3 cell = floor(p), f = fract(p);
    float nearest = 8.0, second = 8.0;
    for (int z = -1; z <= 1; z++)
      for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++) {
          vec3 g = vec3(float(x), float(y), float(z));
          vec3 r = g + vec3(hash(cell + g), hash(cell + g + 31.0), hash(cell + g + 67.0)) - f;
          float d = dot(r, r);
          if (d < nearest) { second = nearest; nearest = d; }
          else if (d < second) second = d;
        }
    return sqrt(second) - sqrt(nearest);
  }

  // Key light, sky fill and a Blinn-Phong highlight, in linear light.
  vec3 shade(vec3 albedo, float shininess, float gloss) {
    vec3 n = normalize(vNormal), v = normalize(cameraPosition - vWorld), l = normalize(uLight);
    float diffuse = max(dot(n, l), 0.0), fill = 0.5 + 0.5 * n.y;
    float highlight = pow(max(dot(n, normalize(l + v)), 0.0), shininess) * gloss;
    float rim = pow(1.0 - max(dot(n, v), 0.0), 4.0) * 0.25;
    return albedo * (diffuse * 2.4 + fill * 0.35) + vec3(highlight * 3.0 + rim);
  }
`;

/** Four procedural GLSL materials on the studio sweep, interpreted on the CPU: marble, lava, a gas giant, pearl. */
export function shadersScene(): ExampleScene {
  const scene = new THREE.Scene();
  const light = new THREE.Vector3(-3, 6, 4).normalize();

  const material = (body: string) => new THREE.ShaderMaterial({
    uniforms: { uLight: { value: light } },
    vertexShader: WORLD_VERTEX,
    fragmentShader: `${SHADER_LIBRARY}\nvoid main() {\n${body}\n${OUTPUT}}`,
  });

  const materials = [
    // Marble: turbulence folds a sine into veins.
    material(/* glsl */ `
      vec3 p = vObject * 2.2;
      float turbulence = fbm(p * 1.5);
      float vein = pow(1.0 - abs(sin(p.x * 2.0 + p.y * 1.2 + turbulence * 7.0)), 10.0);
      float fine = pow(1.0 - abs(sin(p.z * 5.0 + fbm(p * 3.0 + 5.0) * 9.0)), 24.0) * 0.5;
      vec3 stone = mix(vec3(0.82, 0.8, 0.76), vec3(0.95, 0.94, 0.92), fbm(p * 4.0));
      vec3 albedo = mix(stone, vec3(0.18, 0.22, 0.24), clamp(vein + fine, 0.0, 1.0));
      gl_FragColor = vec4(shade(albedo * 0.85, 90.0, 0.6), 1.0);
    `),
    // Lava: a dark crust broken by glowing cracks.
    material(/* glsl */ `
      vec3 p = vObject * 3.2;
      float edge = cracks(p + fbm(p) * 0.6);
      float heat = smoothstep(0.14, 0.0, edge);
      vec3 crust = vec3(0.05, 0.04, 0.04) * (0.6 + fbm(p * 3.0));
      vec3 glow = mix(vec3(1.0, 0.18, 0.02), vec3(1.0, 0.75, 0.2), heat * heat) * heat * 6.0;
      gl_FragColor = vec4(shade(crust, 20.0, 0.15) + glow, 1.0);
    `),
    // A gas giant: latitude bands, stirred by noise, with one oval storm.
    material(/* glsl */ `
      vec3 p = normalize(vObject);
      float stir = fbm(p * vec3(3.0, 14.0, 3.0)) * 1.6;
      float band = sin(p.y * 16.0 + stir);
      vec3 albedo = mix(vec3(0.85, 0.72, 0.55), vec3(0.62, 0.38, 0.22), band * 0.5 + 0.5);
      albedo = mix(albedo, vec3(0.95, 0.9, 0.82), smoothstep(0.6, 1.0, fbm(p * 8.0)) * 0.5);
      vec2 storm = vec2((atan(p.z, p.x) - 0.6) * 1.6, (p.y + 0.28) * 4.0);
      albedo = mix(albedo, vec3(0.7, 0.28, 0.14), smoothstep(0.35, 0.1, length(storm)));
      gl_FragColor = vec4(shade(albedo, 12.0, 0.08), 1.0);
    `),
    // Pearl: a cosine palette over the facing ratio, like a thin film.
    material(/* glsl */ `
      vec3 n = normalize(vNormal), v = normalize(cameraPosition - vWorld);
      float facing = max(dot(n, v), 0.0);
      vec3 film = 0.5 + 0.5 * cos(6.2831853 * (facing * 1.3 + fbm(vObject * 3.0) * 0.4 + vec3(0.0, 0.33, 0.67)));
      vec3 albedo = mix(vec3(0.9, 0.88, 0.86), film, 0.55);
      gl_FragColor = vec4(shade(albedo * 0.8, 160.0, 1.0), 1.0);
    `),
  ];

  materials.forEach((shader, i) => {
    const ball = shadowed(new THREE.Mesh(new THREE.SphereGeometry(.55, 96, 48), shader));
    ball.position.set((i - 1.5) * 1.35, .55, 0);
    ball.rotation.y = i * .7;
    scene.add(ball);
  });

  scene.add(sweep(-3));
  sunlight(scene, [-3, 6, 4], 2);

  return { scene, cameraPosition: [0, 1.6, 5.2], target: [0, .55, 0] };
}

/** Standard normal deviates (Box-Muller) from a uniform generator. */
function gaussian(next: () => number) {
  return () => Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());
}

/**
 * A barred-less spiral galaxy in about 190 000 shader points: a warm bulge, two logarithmic arms of blue young
 * stars with pink nebulae, dust lanes that multiply the light behind them, a faint halo and distant stars.
 */
export function galaxyScene(): ExampleScene {
  const scene = new THREE.Scene();
  const next = random(42), normal = gaussian(next);
  const stars: number[][] = [], dust: number[][] = [];
  // Pitch of the arms: a logarithmic spiral winds 1 / tan(pitch) radians per e-fold of radius.
  const wind = 1 / Math.tan(13 * Math.PI / 180);
  const armAngle = (radius: number, arm: number) => arm * Math.PI + wind * Math.log(radius / .35);

  const add = (list: number[][], x: number, y: number, z: number, size: number, color: number[]) =>
    list.push([x, y, z, size, ...color]);

  // Bulge: a squashed ball of old, warm stars.
  for (let i = 0; i < 38000; i++) {
    const r = Math.abs(normal()) * .38, theta = next() * 2 * Math.PI, u = next() * 2 - 1, ring = Math.sqrt(1 - u * u);
    const warm = .75 + next() * .25;
    add(stars, r * ring * Math.cos(theta), r * u * .55, r * ring * Math.sin(theta), .6 + next() * .8,
      [1, .78 * warm, .52 * warm]);
  }

  // Disk and arms: most stars crowd around the arms; young blue ones sit right on them.
  for (let i = 0; i < 120000; i++) {
    const radius = .3 + -Math.log(1 - next() * .985) * .95, arm = next() < .5 ? 0 : 1;
    const onArm = next() < .78, spread = onArm ? normal() * .32 : (next() - .5) * Math.PI;
    const angle = armAngle(radius, arm) + spread, height = normal() * (.035 + radius * .012);
    const young = onArm && Math.abs(spread) < .15 && next() < .45, fade = Math.min(1, radius / 3.2);

    const color = young ? [.62, .76, 1] : [1 - fade * .3, .86 - fade * .1, .7 + fade * .22];
    const size = young ? 1 + next() : .5 + next() * .7;
    add(stars, radius * Math.cos(angle), height, radius * Math.sin(angle), size, color);
  }

  // Nebulae: clumps of pink along the arms.
  for (let cluster = 0; cluster < 90; cluster++) {
    const radius = .8 + next() * 2.6, arm = cluster % 2, angle = armAngle(radius, arm) + normal() * .08;
    const cx = radius * Math.cos(angle), cz = radius * Math.sin(angle);

    for (let i = 0; i < 40; i++)
      add(stars, cx + normal() * .05, normal() * .02, cz + normal() * .05, 1.4 + next() * 1.6, [1, .36, .56]);
  }

  // Halo: a thin spherical scatter.
  for (let i = 0; i < 6000; i++) {
    const r = 1 + next() * 3.5, theta = next() * 2 * Math.PI, u = next() * 2 - 1, ring = Math.sqrt(1 - u * u);
    add(stars, r * ring * Math.cos(theta), r * u * .6, r * ring * Math.sin(theta), .4 + next() * .4, [.8, .82, .9]);
  }

  // Dust lanes along the inner edge of each arm.
  for (let i = 0; i < 26000; i++) {
    const radius = .45 + -Math.log(1 - next() * .95) * .9, arm = next() < .5 ? 0 : 1;
    const angle = armAngle(radius, arm) - .2 + normal() * .07;
    add(dust, radius * Math.cos(angle), normal() * .02, radius * Math.sin(angle), 2.2 + next() * 2.4, [.35, .27, .22]);
  }

  // Distant stars behind everything.
  const sky: number[][] = [];

  for (let i = 0; i < 4000; i++) {
    const theta = next() * 2 * Math.PI, u = next() * 2 - 1, ring = Math.sqrt(1 - u * u), tint = .7 + next() * .3;
    add(sky, 24 * ring * Math.cos(theta), 24 * u, 24 * ring * Math.sin(theta), .5 + next() * next() * 2,
      [tint, tint, .8 + tint * .2]);
  }

  /** Points from [x, y, z, size, r, g, b] rows; scalar attributes carry size and color. */
  const cloud = (rows: number[][], fragment: string, blending: THREE.Blending, size: number) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(rows.flatMap(r => r.slice(0, 3)), 3));
    ["aSize", "aRed", "aGreen", "aBlue"].forEach((name, k) =>
      geometry.setAttribute(name, new THREE.Float32BufferAttribute(rows.map(r => r[3 + k]), 1)));

    return new THREE.Points(geometry, new THREE.ShaderMaterial({
      uniforms: { uSize: { value: size } },
      transparent: true,
      depthWrite: false,
      blending,
      // Multiply blending needs premultiplied alpha in WebGL; with opaque output both conventions agree.
      premultipliedAlpha: blending === THREE.MultiplyBlending,
      vertexShader: /* glsl */ `
        uniform float uSize;
        attribute float aSize;
        attribute float aRed;
        attribute float aGreen;
        attribute float aBlue;
        varying vec3 vColor;
        void main() {
          vec4 view = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * view;
          gl_PointSize = uSize * aSize / -view.z;
          vColor = vec3(aRed, aGreen, aBlue);
        }
      `,
      fragmentShader: `varying vec3 vColor;\nvoid main() {\n  float d = distance(gl_PointCoord, vec2(0.5));\n${fragment}\n${OUTPUT}}`,
    }));
  };

  const glow = "  gl_FragColor = vec4(vColor * exp(-d * d * 22.0) * 0.55, 1.0);";
  const galaxy = new THREE.Group();
  galaxy.add(cloud(stars, glow, THREE.AdditiveBlending, 34));

  const lanes = cloud(dust, "  gl_FragColor = vec4(mix(vec3(1.0), vColor, exp(-d * d * 10.0) * 0.55), 1.0);",
    THREE.MultiplyBlending, 34);

  lanes.renderOrder = 1;
  galaxy.add(lanes);
  galaxy.rotation.set(.62, 0, .18);
  scene.add(galaxy, cloud(sky, glow, THREE.AdditiveBlending, 90));

  return { scene, cameraPosition: [0, .6, 6.2], target: [0, 0, 0], enclosed: true };
}

/**
 * Wet asphalt `width` × `depth` units across, lying flat: coarse grain everywhere, and puddles where low noise dips,
 * near mirrors that catch the lights.
 */
function wetAsphalt(width: number, depth: number) {
  const grit = tilingNoise(41, 512), pools = tilingNoise(43, 512);

  const asphalt = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), new THREE.MeshStandardMaterial({
    envMapIntensity: 0,
    ...surfaceMaps(512, 512, [width / 5.3, depth / 5], 1.5, (x, y) => {
      const stone = grit(x, y, 64, 3), wet = clamp01((pools(x, y, 3, 4) - .5) * 14);
      const gray = (34 + stone * 30) * (1 - wet * .2);

      return { color: [gray, gray, gray * 1.06], height: stone * (1 - wet), roughness: .5 - stone * .16 - wet * .4 };
    }),
  }));

  asphalt.rotation.x = -Math.PI / 2;
  asphalt.receiveShadow = true;

  return asphalt;
}

/**
 * A back street after rain: neon signs bent from glass tubes on a brick wall, a steel door under a warm lamp and wet
 * asphalt. Each tube has a near-white core and a soft additive glow; colored point lights just in front of the signs
 * paint the bricks, the crates and the drum, and glint in the puddles.
 */
export function neonScene(): ExampleScene {
  const scene = new THREE.Scene();

  const wall = new THREE.Mesh(new THREE.PlaneGeometry(16, 6), new THREE.MeshStandardMaterial({
    ...brickMaps(16, 6), color: 0x8a8480, envMapIntensity: 0 }));

  wall.position.set(0, 3, -1.2);
  wall.receiveShadow = true;
  scene.add(wall);

  const asphalt = wetAsphalt(16, 10);
  asphalt.position.set(0, 0, 3.8);
  scene.add(asphalt);

  const surface = (color: number, roughness: number, metalness = 0) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness, envMapIntensity: 0 });

  // A painted steel door with two pressed panels, worn where hands and boots hit it, in a proud steel frame over a
  // concrete step.
  const wear = tilingNoise(51, 256, 512);

  const doorMaps = surfaceMaps(256, 512, [1, 1], 3, (x, y) => {
    const panel = (x0: number, y0: number, x1: number, y1: number) =>
      clamp01(Math.min(x - x0, x1 - x, y - y0, y1 - y) / 7);

    const pressed = Math.max(panel(38, 40, 218, 250), panel(38, 286, 218, 472));
    const grime = clamp01(1 - y / 120) * .45 + clamp01(1 - Math.hypot(x - 225, y - 250) / 60) * .3;
    const chipped = wear(x, y, 20, 4) + grime * .35 > .83, dust = .85 + wear(x, y, 3, 3) * .25;

    if (chipped)
      return { color: [150, 150, 152], height: 1 - pressed * .5 - .05, roughness: .35, metalness: 1 };

    return {
      color: [52, 84, 90].map(c => c * dust * (1 - grime * .6)),
      height: 1 - pressed * .5,
      roughness: .5 + grime * .35,
    };
  });

  const steel = surface(0x9a9ea3, .32, .8), frameSteel = surface(0x26292d, .55, .3);

  const doorway: [THREE.BufferGeometry, THREE.Material, [number, number, number]][] = [
    [new THREE.BoxGeometry(1, 2.1, .05), new THREE.MeshStandardMaterial({ ...doorMaps,
      metalnessMap: doorMaps.roughnessMap, metalness: 1, envMapIntensity: 0 }), [0, 1.17, -1.17]],
    [new THREE.BoxGeometry(.09, 2.24, .12), frameSteel, [-.545, 1.12, -1.14]],
    [new THREE.BoxGeometry(.09, 2.24, .12), frameSteel, [.545, 1.12, -1.14]],
    [new THREE.BoxGeometry(1.18, .09, .12), frameSteel, [0, 2.265, -1.14]],
    [new THREE.BoxGeometry(.9, .22, .006), steel, [0, .24, -1.142]],
    [new THREE.CylinderGeometry(.03, .03, .02, 24), steel, [.38, 1.07, -1.135]],
    [new THREE.BoxGeometry(.13, .026, .026), steel, [.33, 1.07, -1.115]],
    [new THREE.BoxGeometry(1.5, .12, .42), surface(0x6f6c68, .9), [0, .06, -.99]],
  ];

  const door = new THREE.Group();

  for (const [geometry, material, position] of doorway) {
    const part = shadowed(new THREE.Mesh(geometry, material));
    part.position.set(...position);
    if (geometry instanceof THREE.CylinderGeometry) part.rotation.x = Math.PI / 2;
    // The handle is too small for the shadow map: its shadow would come out as a dark smudge.
    part.castShadow = material !== steel;
    door.add(part);
  }

  // A barn lamp over it: an arm curving out of the wall, an enameled dome and a warm bulb that lights the door.
  const arm = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 2.52, -1.2), new THREE.Vector3(0, 2.62, -1.05),
    new THREE.Vector3(0, 2.68, -.9), new THREE.Vector3(0, 2.64, -.82)]);

  const enamel = new THREE.MeshStandardMaterial({ color: 0x1f3b30, roughness: .35, metalness: .2,
    envMapIntensity: 0, side: THREE.DoubleSide });

  const dome = new THREE.LatheGeometry(Array.from({ length: 13 }, (_, i) => {
    const t = i / 12;

    return new THREE.Vector2(.03 + .19 * Math.sin(t * Math.PI / 2) ** 1.5, -.13 * t);
  }), 48);

  const shade = new THREE.Mesh(dome, enamel);
  shade.position.set(0, 2.64, -.82);
  shade.castShadow = true;

  const bulb = new THREE.Mesh(new THREE.SphereGeometry(.045, 24, 12),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0xffc27a).multiplyScalar(8) }));

  bulb.position.set(0, 2.55, -.82);
  const lamp = new THREE.SpotLight(0xffb066, 30, 0, .95, .7, 2);
  lamp.position.set(0, 2.52, -.82);
  lamp.target.position.set(0, 0, -.95);
  lamp.castShadow = true;
  // The dome's hot rim throws a little light back up the bricks.
  const spill = new THREE.PointLight(0xffb066, 1, 0, 2);
  spill.position.set(0, 2.5, -.7);

  door.add(shadowed(new THREE.Mesh(new THREE.TubeGeometry(arm, 24, .014, 8), frameSteel)), shade, bulb, lamp,
    lamp.target, spill);

  door.position.x = -3.1;
  scene.add(door);

  scene.add(
    steelDrum([150, 38, 34], [2.35, .44, -.75], .8),
    crate(.7, [3.35, .35, -.8], .2),
    crate(.44, [3.3, .92, -.85], -.35),
    crate(.5, [-1.7, .25, -.85], .5),
  );

  const v = (x: number, y: number) => new THREE.Vector3(x, y, 0);

  const arc = (cx: number, cy: number, r: number, from: number, to: number, steps = 16) =>
    Array.from({ length: steps + 1 }, (_, i) => {
      const a = (from + (to - from) * i / steps) * Math.PI / 180;

      return v(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
    });

  // Letters 0.84 units tall around y = 0, spaced evenly about x = 0 and drawn as the path a glass bender would follow.
  const letters = [
    arc(-.87, 0, .42, 48, 312, 28),
    [v(-.25, -.42), v(-.25, .42), v(.13, .42), ...arc(.13, .2, .22, 90, -90, 14), v(-.25, -.02)],
    [v(.67, .42), v(.67, -.11), ...arc(.98, -.11, .31, 180, 360, 16), v(1.29, .42)],
  ];

  const corner = .26, w = 1.78, h = .8;

  const frame = [
    ...arc(w - corner, h - corner, corner, 0, 90, 6), ...arc(-w + corner, h - corner, corner, 90, 180, 6),
    ...arc(-w + corner, -h + corner, corner, 180, 270, 6), ...arc(w - corner, -h + corner, corner, 270, 360, 6),
  ];

  const glowShader = (color: THREE.Color, strength: number) => new THREE.ShaderMaterial({
    uniforms: { uColor: { value: color }, uStrength: { value: strength } },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: WORLD_VERTEX,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uStrength;
      varying vec3 vWorld;
      varying vec3 vNormal;
      void main() {
        // Brightest where the shell faces the camera, fading to nothing at its silhouette.
        float facing = abs(dot(normalize(vNormal), normalize(cameraPosition - vWorld)));
        gl_FragColor = vec4(uColor * pow(facing, 3.0) * uStrength, 1.0);
      ${OUTPUT}}
    `,
  });

  const sign = new THREE.Group();

  const tube = (points: THREE.Vector3[], hex: number, closed = false) => {
    const color = new THREE.Color(hex), path = new THREE.CatmullRomCurve3(points, closed, "centripetal", .2);
    const core = color.clone().lerp(new THREE.Color(1, 1, 1), .4).multiplyScalar(4);
    const segments = points.length * 6;
    sign.add(new THREE.Mesh(new THREE.TubeGeometry(path, segments, .045, 12, closed),
      new THREE.MeshBasicMaterial({ color: core })));
    sign.add(new THREE.Mesh(new THREE.TubeGeometry(path, segments, .1, 16, closed), glowShader(color, 1.3)));
    sign.add(new THREE.Mesh(new THREE.TubeGeometry(path, segments, .2, 16, closed), glowShader(color, .28)));
  };

  for (const letter of letters) tube(letter, 0xff2a8a);
  tube(frame, 0x18d6ff, true);
  sign.position.set(.4, 3.05, -.95);
  scene.add(sign);

  // The light the tubes throw: a few colored points in front of each sign, far enough out to spread over the wall.
  const lights: [number, number, [number, number, number], boolean][] = [
    [0xff2a8a, 10, [-.3, 3.05, -.4], true], [0xff2a8a, 10, [1.2, 3.05, -.4], false],
    [0x18d6ff, 8, [-1.35, 2.6, -.45], false], [0x18d6ff, 8, [2.15, 2.6, -.45], true],
  ];

  for (const [color, intensity, position, shadows] of lights) {
    const light = new THREE.PointLight(color, intensity, 0, 2);
    light.position.set(...position);
    light.castShadow = shadows;
    scene.add(light);
  }

  scene.add(new THREE.HemisphereLight(0x3a4468, 0x0a0a0e, .6));

  return { scene, cameraPosition: [-.6, 1.45, 7.4], target: [.2, 1.85, -1.2], enclosed: true };
}

/**
 * A country road on a foggy night, far from town: a lamp on a wooden pole pours a cold cone of light through the
 * air, dead trees crowd the fields and someone stands at the edge of the light. The lamps' spot lights are marked
 * `userData.cpuRenderer.volumetric`, so the renderer marches the air along every view ray, and the scene fog dims what glows
 * far off.
 */
export function roadScene(): ExampleScene {
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x131922, .045);
  const next = random(66);

  // The night sky is a dome far enough out that the fog paints it: dark blue-gray, with the trees against it. The
  // moon ignores the fog, so it shows through as a dim disc.
  const sky = new THREE.Mesh(new THREE.SphereGeometry(80, 32, 16), new THREE.MeshBasicMaterial({
    color: 0x131922, side: THREE.BackSide }));

  const moon = new THREE.Mesh(new THREE.CircleGeometry(1.6, 48), new THREE.MeshBasicMaterial({
    color: 0x5c6573, fog: false }));

  moon.position.set(19, 27, -60);
  moon.lookAt(0, 0, 0);
  scene.add(sky, moon);

  const surface = (color: number, roughness: number, metalness = 0) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness, envMapIntensity: 0 });

  // A two-lane road, 6 m of old asphalt: cracked, patched with puddles, a worn dashed center line, and edges that
  // crumble into gravel shoulders. One repeat covers the full 7 m width and 6 m of length.
  const grit = tilingNoise(41, 512), pools = tilingNoise(43, 512);

  const roadMaps = surfaceMaps(512, 512, [1, 10], 2, (x, y) => {
    const across = (x / 512 - .5) * 7, along = y / 512 * 6, stone = grit(x, y, 64, 3), wear = grit(x, y, 6, 4);
    const edge = 3 - Math.abs(across) + (wear - .5) * .7;

    if (edge < 0) {
      const pebble = grit(x, y, 128, 2);

      return { color: [62 + pebble * 50, 57 + pebble * 44, 50 + pebble * 36], height: pebble, roughness: .95 };
    }

    const wet = clamp01((pools(x, y, 3, 4) - .52) * 12), crack = Math.abs(pools(x, y, 16, 3) - .5) < .01;
    const paint = Math.abs(across) < .06 && along % 6 < 3 && wear > .38;
    const gray = (36 + stone * 28) * (1 - wet * .25) * (crack ? .45 : 1);

    return {
      color: paint ? [150 * (.7 + stone * .3), 126 * (.7 + stone * .3), 58] : [gray, gray, gray * 1.05],
      height: crack ? 0 : stone * (1 - wet) * .6 + .4,
      roughness: (paint ? .6 : .55 - stone * .15) - wet * .45,
    };
  });

  const road = floor(new THREE.MeshStandardMaterial({ ...roadMaps, envMapIntensity: 0 }), 1);
  road.scale.set(7, 60, 1);
  road.position.set(0, .01, -20);

  // Mud and dead grass on both sides of the road.
  const soil = tilingNoise(71, 256);

  const field = floor(new THREE.MeshStandardMaterial({ envMapIntensity: 0, ...surfaceMaps(256, 256, [16, 16], 3,
    (x, y) => {
      const clump = soil(x, y, 8, 4), blade = soil(x * 3, y, 64, 2), grass = clamp01((clump - .45) * 5);

      return {
        color: [52 + blade * 26 - grass * 14, 46 + blade * 22 + grass * 6, 34 + blade * 12 - grass * 10],
        height: clump * .5 + blade * grass * .5, roughness: .95,
      };
    }) }), 64);

  field.position.z = -20;
  scene.add(road, field);

  const wood = surface(0x3b342d, .95), iron = surface(0x2a2c2e, .5, .5), cable = surface(0x111214, .6);

  // Wooden utility poles down the left shoulder, a crossarm on each. Two carry a cobra-head lamp on a bent arm.
  const poleX = -4.3, poles = [13, -5, -21, -37];

  const utilityPole = (z: number, intensity: number) => {
    const pole = new THREE.Group();
    const post = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(.09, .13, 7, 16), wood));
    post.position.y = 3.5;
    const crossarm = shadowed(new THREE.Mesh(new THREE.BoxGeometry(1.7, .1, .1), wood));
    crossarm.position.set(0, 6.6, 0);
    pole.add(post, crossarm);

    for (const x of [-.72, .72]) {
      const insulator = new THREE.Mesh(new THREE.CylinderGeometry(.03, .04, .12, 12), surface(0x6d746e, .3));
      insulator.position.set(x, 6.71, 0);
      pole.add(insulator);
    }

    pole.position.set(poleX, 0, z);
    scene.add(pole);
    if (!intensity)
      return;

    const arm = shadowed(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 5.4, 0), new THREE.Vector3(.6, 5.8, 0), new THREE.Vector3(1.8, 5.92, 0),
      new THREE.Vector3(2.5, 5.86, 0)]), 32, .03, 8), iron));

    // The head casts no shadow: its own light starts inside it.
    const head = new THREE.Mesh(new THREE.SphereGeometry(.28, 32, 12, 0, Math.PI * 2, 0, Math.PI / 2), iron);
    head.scale.set(1.4, .5, .8);
    head.position.set(2.65, 5.78, 0);

    const lens = new THREE.Mesh(new THREE.CircleGeometry(.2, 32), new THREE.MeshBasicMaterial({
      color: new THREE.Color(0xd8f5e0).multiplyScalar(6), side: THREE.DoubleSide }));

    lens.rotation.x = Math.PI / 2;
    lens.scale.set(1.4, .8, 1);
    lens.position.set(2.65, 5.76, 0);
    pole.add(arm, head, lens);

    // Two spots share the head: a strong one lights the ground, and a faint volumetric one lights the air. The
    // glow grows with its light's intensity, so one light bright enough for the road would fill the air.
    for (const [strength, volumetric] of [[intensity, false], [intensity * .075, true]] as const) {
      const light = new THREE.SpotLight(0xc8f0d4, strength, 16, .55, .55, 2);
      light.position.set(2.65, 5.7, 0);
      light.target.position.set(2.65, 0, .2);
      light.castShadow = true;
      // Fog scatters mostly forward, and bounces enough light between droplets to blur the cone into the air.
      light.userData.cpuRenderer = { volumetric: volumetric && { density: .022, anisotropy: .5, spread: .5 } };
      pole.add(light, light.target);
    }
  };

  poles.forEach((z, i) => utilityPole(z, [0, 160, 130, 0][i]));

  // Power lines sag between the insulators, pole to pole.
  for (let i = 1; i < poles.length; i++)
    for (const x of [-.72, .72]) {
      const span = Array.from({ length: 17 }, (_, j) => {
        const t = j / 16;

        return new THREE.Vector3(poleX + x, 6.76 - Math.sin(t * Math.PI) * .6,
          poles[i - 1] + (poles[i] - poles[i - 1]) * t);
      });

      scene.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(span), 32, .022, 5), cable));
    }

  // Dead trees: every limb forks into two or three thinner ones, bent at random. The nearest leans out over the
  // road, into the lamp's cone; the rest crowd the fields and fade into the fog.
  const bark = surface(0x241f1b, .95);

  const limb = (tree: THREE.Group, start: THREE.Vector3, direction: THREE.Vector3, length: number,
    radius: number, depth: number) => {
    const end = start.clone().addScaledVector(direction, length);
    const segment = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(radius * .7, radius, length, 8), bark));
    segment.position.copy(start).lerp(end, .5);
    segment.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    tree.add(segment);
    if (depth === 0)
      return;

    for (let i = 0, forks = 2 + Math.floor(next() * 2); i < forks; i++) {
      const bend = new THREE.Vector3(next() - .5, next() * .5 - .1, next() - .5).multiplyScalar(1.5);
      limb(tree, end, direction.clone().add(bend).normalize(), length * (.62 + next() * .2), radius * .62,
        depth - 1);
    }
  };

  const deadTree = (x: number, z: number, height: number, lean: [number, number]) => {
    const tree = new THREE.Group();
    limb(tree, new THREE.Vector3(x, 0, z), new THREE.Vector3(lean[0], 1, lean[1]).normalize(), height * .4,
      height * .035, 4);
    scene.add(tree);
  };

  deadTree(-6.3, -3.6, 8, [.5, 0]);
  deadTree(6.6, -7.5, 8.5, [-.25, .1]);

  for (let i = 0; i < 18; i++) {
    const side = i % 2 ? 1 : -1;
    deadTree(side * (6.5 + next() * 8), 7 - next() * 46, 6 + next() * 3.5, [(next() - .5) * .4, (next() - .5) * .3]);
  }

  // A sagging fence along the right-hand shoulder, a rail missing here and there, a post leaning.
  for (let z = 9; z > -40; z -= 2.4) {
    const post = shadowed(new THREE.Mesh(new THREE.BoxGeometry(.1, 1.25, .1), wood));
    post.position.set(4.1, .58, z);
    post.rotation.set((next() - .5) * .12, 0, (next() - .5) * .16);
    scene.add(post);

    for (const height of [.45, .92]) {
      if (next() < .18) continue;
      const rail = shadowed(new THREE.Mesh(new THREE.BoxGeometry(.05, .09, 2.45), wood));
      rail.position.set(4.1, height + (next() - .5) * .08, z - 1.2);
      rail.rotation.x = (next() - .5) * .08;
      scene.add(rail);
    }
  }

  // Someone standing at the edge of the light: tall and thin in a long coat, shoulders sloping, arms hanging a little
  // too long, hunched, the head tipped to one side. Too still.
  const figure = new THREE.Group(), upper = new THREE.Group(), cloth = surface(0x0e0e10, .95);
  const up = new THREE.Vector3(0, 1, 0);

  const part = (parent: THREE.Object3D, geometry: THREE.BufferGeometry, position: [number, number, number]) => {
    const mesh = shadowed(new THREE.Mesh(geometry, cloth));
    mesh.position.set(...position);
    parent.add(mesh);

    return mesh;
  };

  // A capsule `radius` thick from joint `a` to joint `b`.
  const bone = (parent: THREE.Object3D, a: [number, number, number], b: [number, number, number],
    radius: number) => {
    const from = new THREE.Vector3(...a), to = new THREE.Vector3(...b), length = from.distanceTo(to);
    const mesh = part(parent, new THREE.CapsuleGeometry(radius, Math.max(0, length - radius), 4, 12), a);
    mesh.position.lerp(to, .5);
    mesh.quaternion.setFromUnitVectors(up, to.sub(from).normalize());

    return mesh;
  };

  for (const side of [-1, 1]) {
    bone(figure, [side * .1, .92, 0], [side * .105, .46, .01], .068);
    bone(figure, [side * .105, .46, .01], [side * .11, .1, 0], .052);
    part(figure, new THREE.CapsuleGeometry(.048, .16, 4, 12), [side * .11, .045, .06]).rotation.x = Math.PI / 2;
  }

  // The coat: flared at the hem, in at the waist, a narrow chest and shoulders sloping into the neck.
  const coat: [number, number][] = [[0, .6], [.25, .6], [.22, .8], [.19, 1.02], [.2, 1.2], [.215, 1.34],
    [.21, 1.44], [.18, 1.5], [.12, 1.55], [.06, 1.575], [0, 1.58]];

  part(upper, new THREE.LatheGeometry(coat.map(([r, y]) => new THREE.Vector2(r, y - .92)), 40), [0, 0, 0])
    .scale.z = .62;

  for (const side of [-1, 1]) {
    part(upper, new THREE.SphereGeometry(.07, 16, 12), [side * .185, .5, 0]).scale.z = .85;
    bone(upper, [side * .2, .5, 0], [side * .25, .2, .02], .05);
    bone(upper, [side * .25, .2, .02], [side * .245, -.08, .09], .043);
    // Long, thin hands, fingers together.
    bone(upper, [side * .245, -.08, .08], [side * .25, -.26, .1], .03).scale.x = .75;
  }

  part(upper, new THREE.CylinderGeometry(.052, .06, .1, 12), [0, .66, .01]);
  const neck = new THREE.Group();
  neck.position.set(0, .69, .02);
  neck.rotation.set(.2, 0, -.42);
  part(neck, new THREE.SphereGeometry(.096, 24, 16), [0, .085, .015]).scale.set(.92, 1.1, 1);
  upper.add(neck);
  // Hunched: the whole upper body leans forward from the hips.
  upper.position.y = .92;
  upper.rotation.x = .08;
  figure.add(upper);
  figure.scale.setScalar(1.06);
  figure.position.set(1.3, .01, -5.6);
  figure.rotation.y = -.12;
  scene.add(figure);

  // Moonlight from behind the trees, throwing their shadows long down the road toward the camera, and a little
  // sky light to find the shapes outside the lamp.
  const moonlight = new THREE.DirectionalLight(0x8fa2c8, .5);
  moonlight.position.set(19, 27, -60);
  moonlight.target.position.set(0, 0, -5);
  moonlight.castShadow = true;
  scene.add(moonlight, moonlight.target, new THREE.HemisphereLight(0x3a4a6a, 0x050506, .6));

  return { scene, cameraPosition: [.9, 1.6, 9], target: [-.7, 2.6, -8], enclosed: true };
}

/**
 * A profile for LatheGeometry from [radius, height] points and arcs [center radius, center height, arc radius,
 * from°, to°], so beads and domes come out round.
 */
function turned(parts: ([number, number] | [number, number, number, number, number])[]) {
  return parts.flatMap(part => {
    if (part.length === 2)
      return [new THREE.Vector2(...part)];
    const [cx, cy, r, from, to] = part;

    return Array.from({ length: 11 }, (_, i) => {
      const a = (from + (to - from) * i / 10) * Math.PI / 180;

      return new THREE.Vector2(Math.max(0, cx + Math.cos(a) * r), cy + Math.sin(a) * r);
    });
  });
}

/**
 * A chess game in progress on a lacquered board, seen from low beside the white queen with depth of field: every
 * piece is turned on a lathe, bead by bead.
 */
export function chessScene(): ExampleScene {
  const scene = new THREE.Scene();
  const square = .3, shades = Array.from({ length: 64 }, random(5)), grain = tilingNoise(5, 512);

  // Maple and walnut squares, the grain turning a quarter from each square to the next.
  const board = surfaceMaps(512, 512, [1, 1], 2, (x, y) => {
    const file = Math.floor(x / 64), rank = Math.floor(y / 64), dark = (file + rank) % 2 === 0;
    const turn = (file + rank * 3) % 2 === 0, seam = x % 64 < 1 || y % 64 < 1;
    const wood = oak(grain, shades, turn ? x : y, turn ? y : x, file * 8 + rank);
    const color = dark ? wood.map(c => c * .42) : wood.map(c => Math.min(255, c * 1.35 + 30));

    return { color: seam ? color.map(c => c * .6) : color, height: seam ? 0 : 1, roughness: .3 };
  });

  const squares = new THREE.Mesh(new THREE.PlaneGeometry(square * 8, square * 8), new THREE.MeshPhysicalMaterial({
    ...board, clearcoat: 1, clearcoatRoughness: .08 }));

  squares.rotation.x = -Math.PI / 2;
  squares.position.y = .001;
  squares.receiveShadow = true;

  const frame = shadowed(new THREE.Mesh(new THREE.BoxGeometry(square * 8 + .3, .1, square * 8 + .3),
    new THREE.MeshPhysicalMaterial({ color: 0x3a2317, roughness: .35, clearcoat: 1, clearcoatRoughness: .1 })));

  frame.position.y = -.05;
  const table = sweep(-4);
  table.position.y = -.1;
  (table.material as THREE.MeshStandardMaterial).color.set(0x2e2c2b);
  scene.add(squares, frame, table);

  const ivory = new THREE.MeshPhysicalMaterial({ color: 0xeee4d0, roughness: .3, clearcoat: .6,
    clearcoatRoughness: .15 });

  const ebony = new THREE.MeshPhysicalMaterial({ color: 0x1b1614, roughness: .25, clearcoat: 1,
    clearcoatRoughness: .05 });

  type Profile = Parameters<typeof turned>[0];

  // Every base: a felt-edged foot, a rounded plinth bead, a cove and a second, finer bead, then an ogee sweeping
  // in toward the stem at `stem`.
  const base = (r: number, stem: number): Profile => [
    [0, 0], [r * .97, 0], [r * .97, .004, .004, -90, 0], [r, .012], [r, .022],
    [r * .97, .033, .013, -70, 90], [r * .9, .046], [r * .84, .05], [r * .82, .058, .008, -90, 90], [r * .76, .066],
    [r * .72, .078, .012, -90, 60], [r * .66, .092], [stem + (r * .6 - stem) * .45, .11], [stem, .135],
  ];

  // A collar under the head: a thin disc between two beads.
  const collar = (y: number, r: number, stem: number): Profile => [
    [stem * 1.1, y - .018, .008, -90, 90], [r * .8, y - .006], [r, y - .004, .006, -90, 90], [r * .8, y + .01],
    [stem * 1.15, y + .014, .006, -90, 90],
  ];

  const profiles = {
    pawn: turned([...base(.115, .046), [.042, .17], ...collar(.2, .08, .04), [.048, .222],
      [0, .27, .058, -58, 90]]),
    rook: turned([...base(.13, .07), [.064, .25], ...collar(.285, .098, .064), [.086, .3], [.094, .305],
      [.094, .37], [.07, .37], [.07, .345], [0, .345]]),
    bishop: turned([...base(.12, .052), [.04, .28], ...collar(.31, .08, .04), [.048, .33], [.064, .37],
      [.06, .41], [.045, .45], [.02, .48], [0, .5, .022, -60, 90]]),
    queen: turned([...base(.135, .062), [.045, .35], ...collar(.385, .092, .044), [.05, .41], [.07, .47],
      [.092, .51], [.092, .52], [.066, .52], [.04, .535], [0, .568, .026, -70, 90]]),
    king: turned([...base(.14, .066), [.048, .37], ...collar(.405, .096, .048), [.056, .43], [.078, .51],
      [.086, .535], [.08, .545], [0, .55]]),
  };

  const piece = (kind: keyof typeof profiles, white: boolean, file: number, rank: number) => {
    const material = white ? ivory : ebony, group = new THREE.Group();

    group.add(shadowed(new THREE.Mesh(new THREE.LatheGeometry(profiles[kind], 64), material)));

    if (kind === "rook")
      for (let i = 0; i < 6; i++) {
        const merlon = shadowed(new THREE.Mesh(new THREE.BoxGeometry(.034, .04, .03), material));
        const a = i / 6 * Math.PI * 2;
        merlon.position.set(Math.cos(a) * .078, .39, Math.sin(a) * .078);
        merlon.rotation.y = -a;
        group.add(merlon);
      }

    if (kind === "king")
      for (const [w, h, y] of [[.026, .13, .6], [.085, .026, .615]]) {
        const bar = shadowed(new THREE.Mesh(new THREE.BoxGeometry(w, h, .026), material));
        bar.position.y = y;
        group.add(bar);
      }

    group.position.set((file - 3.5) * square, 0, (3.5 - rank) * square);
    scene.add(group);
  };

  // Midgame: both sides castled, the queens out, the knights already traded off.
  const white: [keyof typeof profiles, number, number][] = [
    ["king", 6, 0], ["rook", 5, 0], ["rook", 0, 0], ["pawn", 5, 1], ["pawn", 6, 1], ["pawn", 7, 1], ["pawn", 0, 1],
    ["pawn", 1, 1], ["queen", 3, 2], ["bishop", 2, 3], ["pawn", 4, 3], ["pawn", 2, 2]];

  const black: [keyof typeof profiles, number, number][] = [
    ["king", 6, 7], ["rook", 5, 7], ["rook", 0, 7], ["pawn", 5, 6], ["pawn", 6, 6], ["pawn", 7, 5], ["pawn", 0, 6],
    ["pawn", 1, 5], ["queen", 4, 6], ["bishop", 1, 6], ["pawn", 4, 4], ["pawn", 3, 5]];

  for (const [kind, file, rank] of white) piece(kind, true, file, rank);
  for (const [kind, file, rank] of black) piece(kind, false, file, rank);
  sunlight(scene, [-3.5, 3.2, 1.5], 2.4);

  return { scene, cameraPosition: [-.95, .5, 1.45], target: [-.15, .3, .45],
    settings: { depthOfField: { enabled: true, aperture: .9 }, samples: 16 } };
}

/**
 * An attic in the afternoon: a shaft of sun comes through the round window in the gable and crosses the dusty air
 * to the floorboards. The sun's spot light is volumetric with a forward-scattering medium, so the beam glows
 * brightest looking toward the window, and the window's cross cuts the beam in four.
 */
export function atticScene(): ExampleScene {
  const scene = new THREE.Scene();
  const width = 6, depth = 8, wall = 1.2, ridge = 3.6;

  const surface = (color: number, roughness: number, metalness = 0) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness, envMapIntensity: 0 });

  const boards = floor(new THREE.MeshStandardMaterial({ ...plankMaps(depth, depth), envMapIntensity: 0 }), depth);
  scene.add(boards);

  // Plaster for the gables and the roof slopes: lime wash over uneven render.
  const render = tilingNoise(81, 256);

  const plaster = () => new THREE.MeshStandardMaterial({ envMapIntensity: 0, side: THREE.DoubleSide,
    ...surfaceMaps(256, 256, [4, 4], 1.5, (x, y) => {
      const lump = render(x, y, 6, 4), stain = render(x, y, 2, 3), tone = 206 + lump * 30 - stain * 40;

      return { color: [tone, tone * .95, tone * .86], height: lump, roughness: .95 };
    }) });

  // Each gable: a pentagon with a round window cut in the far one.
  const gableShape = () => new THREE.Shape([[-width / 2, 0], [width / 2, 0], [width / 2, wall], [0, ridge],
    [-width / 2, wall]].map(([x, y]) => new THREE.Vector2(x, y)));

  const far = gableShape(), windowY = 2.1, windowR = .55;
  far.holes.push(new THREE.Path().absarc(0, windowY, windowR, 0, Math.PI * 2, true));

  const farWall = shadowed(new THREE.Mesh(new THREE.ShapeGeometry(far, 48), plaster()));
  farWall.position.z = -depth / 2;
  const nearWall = shadowed(new THREE.Mesh(new THREE.ShapeGeometry(gableShape()), plaster()));
  nearWall.position.z = depth / 2;
  scene.add(farWall, nearWall);

  // The roof slopes, from the knee walls up to the ridge.
  const slope = Math.hypot(width / 2, ridge - wall), pitch = Math.atan2(ridge - wall, width / 2);

  for (const side of [-1, 1]) {
    const roof = shadowed(new THREE.Mesh(new THREE.PlaneGeometry(slope, depth), plaster()));
    roof.position.set(side * width / 4, (wall + ridge) / 2, 0);
    // Tilt the plane's width down the slope, then lay it along the attic's depth.
    roof.rotation.z = -side * pitch;
    roof.rotateX(-Math.PI / 2);
    const knee = shadowed(new THREE.Mesh(new THREE.PlaneGeometry(depth, wall), plaster()));
    knee.position.set(side * width / 2, wall / 2, 0);
    knee.rotation.y = -side * Math.PI / 2;
    scene.add(roof, knee);
  }

  // Rafters under the slopes and a ridge beam, in old dark timber.
  const timber = surface(0x4a3526, .85);

  for (let z = -depth / 2 + .6; z < depth / 2; z += 1.1)
    for (const side of [-1, 1]) {
      const rafter = shadowed(new THREE.Mesh(new THREE.BoxGeometry(slope, .16, .1), timber));
      rafter.position.set(side * width / 4 * .96, (wall + ridge) / 2 - .1, z);
      rafter.rotation.z = -side * pitch;
      scene.add(rafter);
    }

  const beam = shadowed(new THREE.Mesh(new THREE.BoxGeometry(.18, .2, depth), timber));
  beam.position.set(0, ridge - .16, 0);
  const tie = shadowed(new THREE.Mesh(new THREE.BoxGeometry(width - .4, .16, .14), timber));
  tie.position.set(0, 2.55, -1.2);
  scene.add(beam, tie);

  // The window's cross and its frame, set in the far gable.
  const frameRing = shadowed(new THREE.Mesh(new THREE.TorusGeometry(windowR, .05, 12, 48), timber));
  frameRing.position.set(0, windowY, -depth / 2);

  for (const angle of [0, Math.PI / 2]) {
    const bar = shadowed(new THREE.Mesh(new THREE.BoxGeometry(windowR * 2, .06, .06), timber));
    bar.position.set(0, windowY, -depth / 2);
    bar.rotation.z = angle;
    scene.add(bar);
  }

  scene.add(frameRing);

  // What the attic keeps: a trunk, a chair under a dust sheet, a stack of books, a lamp without a shade.
  const trunk = shadowed(new THREE.Mesh(new THREE.BoxGeometry(1, .55, .55), surface(0x5b3a22, .7)));
  trunk.position.set(-1.6, .275, -1.4);
  trunk.rotation.y = .2;

  const lid = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(.275, .275, 1, 24, 1, false, 0, Math.PI),
    surface(0x5b3a22, .7)));

  lid.rotation.set(0, .2, Math.PI / 2);
  lid.position.set(-1.6, .55, -1.4);
  lid.rotateX(-Math.PI / 2);

  for (const x of [-.35, .35]) {
    const strap = shadowed(new THREE.Mesh(new THREE.BoxGeometry(.05, .56, .57), surface(0x2c2622, .4, .7)));
    strap.position.set(x, 0, 0);
    trunk.add(strap);
  }

  const sheet = shadowed(new THREE.Mesh(new THREE.LatheGeometry(turned([[0, 0], [.5, 0], [.46, .3], [.42, .5],
    [.36, .56], [.4, .7], [.36, 1], [.24, 1.12], [0, 1.15]]), 32), surface(0xd9d2c4, .95)));

  sheet.scale.set(1, 1, .85);
  sheet.position.set(1.5, 0, -2.2);
  scene.add(trunk, lid, sheet);
  const bookColors = [0x7a2b22, 0x2f4a3a, 0x8a6a3a, 0x263a5a, 0x5a2a42, 0x6a6a5a];

  for (let i = 0, y = 0; i < 6; i++) {
    const h = .05 + (i % 3) * .015;

    const book = shadowed(new THREE.Mesh(new THREE.BoxGeometry(.34 - i * .02, h, .25 - (i % 2) * .03),
      surface(bookColors[i], .7)));

    book.position.set(.4, y + h / 2, -.6);
    book.rotation.y = (i * 1.7) % 1 - .5;
    y += h;
    scene.add(book);
  }

  // The sun: low and behind the gable, aimed through the window. One spot lights the floor; a second, fainter and
  // volumetric, lights the dust in the air.
  const sun = new THREE.Vector3(1.2, 6.5, -depth / 2 - 7), aim = new THREE.Vector3(-.4, 0, .6);

  for (const [intensity, volumetric] of [[900, false], [60, true]] as const) {
    const light = new THREE.SpotLight(0xffe2b0, intensity, 0, .13, .2, 2);
    light.position.copy(sun);
    light.target.position.copy(aim);
    light.castShadow = true;
    light.userData.cpuRenderer = { volumetric: volumetric && { density: .05, anisotropy: .6, spread: .12 } };
    scene.add(light, light.target);
  }

  // Bright sky through the window, beyond the sun so it throws no shadow on it.
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(30, 20), new THREE.MeshBasicMaterial({
    color: new THREE.Color(0xcfe0f0).multiplyScalar(1.6) }));

  sky.position.set(0, 4, -depth / 2 - 9);
  // The sun patch on the floor lights the room back up: a warm glow low over it stands in for the bounce.
  const bounce = new THREE.PointLight(0xffc690, 2.2, 0, 2);
  bounce.position.set(-.5, .5, .4);
  scene.add(sky, bounce, new THREE.HemisphereLight(0x9a8a78, 0x3a2c22, .7));

  return { scene, cameraPosition: [2.2, 1.55, 3.4], target: [-.3, 1.4, -2.2], enclosed: true };
}

/**
 * A city at the blue hour, seen from above, all of it instanced. Two districts lay their streets out at different
 * angles, split by a diagonal avenue; blocks divide into lots of their own sizes. Houses with pitched roofs and
 * mid-rise blocks, some L-shaped or with a penthouse, climb toward downtown and a second cluster, where towers go up
 * square, octagonal, round or crowned with a pyramid, in glass, stone and brick. Every lit window has its own brightness and
 * warmth, lamps line the curbs and cars' lights run down the streets.
 */
export function cityScene(): ExampleScene {
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x2a3452, .012);
  const next = random(91), noise = tilingNoise(93, 64), grit = tilingNoise(95, 256);

  // Each style is a facade and a window map, 16 floors by 16 bays per repeat. `light` gets two random numbers per
  // window and one per floor, and returns the window's color, or null when it is dark.
  const style = (seed: number, [x0, x1, y0, y1]: number[], wall: (x: number, y: number) => number[],
    glass: number[], light: (a: number, b: number, floor: number) => number[] | null, glow: number) => {
    const pick = random(seed), floors = Array.from({ length: 16 }, pick);
    const first = Array.from({ length: 256 }, pick), second = Array.from({ length: 256 }, pick);
    const inside = (x: number, y: number) => x % 16 >= x0 && x % 16 <= x1 && y % 16 >= y0 && y % 16 <= y1;

    // The first 2 × 2 texels are tar-dark: roofs sample them.
    const facade = dataTexture(256, 256, (x, y) => [...(x < 2 && y < 2 ? [46, 46, 50] : inside(x, y) ? glass
      : wall(x, y)).map(Math.round), 255]);

    const windows = dataTexture(256, 256, (x, y) => {
      const row = Math.floor(y / 16), cell = row * 16 + Math.floor(x / 16);
      const color = inside(x, y) ? light(first[cell], second[cell], floors[row]) : null;

      return [...(color ?? [0, 0, 0]).map(Math.round), 255];
    });

    for (const texture of [facade, windows]) {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      texture.magFilter = texture.minFilter = THREE.LinearFilter;
    }

    return new THREE.MeshStandardMaterial({ map: facade, emissiveMap: windows, emissive: 0xffffff,
      emissiveIntensity: glow, roughness: .55, metalness: .1, envMapIntensity: 0 });
  };

  // A home's window: most dark, the rest behind curtains or blinds at every brightness, from candle-warm to the
  // odd cool one, and now and then the blue of a television.
  const home = (threshold: number) => (a: number, b: number, floor: number) => {
    if (a < threshold - (floor < .15 ? .2 : 0)) return null;
    if (b > .97) return [90, 120, 210];
    const bright = .18 + b ** 1.6 * .82, cool = b * 7 % 1 < .12;

    return (cool ? [220, 222, 230] : [255, 170 + b * 40, 90 + b * 50]).map(c => c * bright);
  };

  const piers = (x: number) => x % 64 < 4 ? .82 : 1;

  const styles = {
    // Curtain wall: narrow mullions and a dark spandrel at each floor. Offices glow dim and warm, a few rooms
    // brighter.
    glass: style(101, [1, 15, 3, 15], (x, y) => (y % 16 < 3 ? [62, 68, 80] : [96, 104, 116]).map(c => c * piers(x)),
      [38, 46, 62], (a, b, floor) => floor > .94 && a < .7 ? [236, 212, 166].map(c => c * (.28 + b * .2))
        : a < .5 ? [214, 200, 172].map(c => c * (.1 + b * .26)) : null, 1),
    stone: style(102, [4, 11, 4, 12], (x, y) => [170, 158, 136].map(c => c * piers(x) * (.9 + grit(x, y, 32, 2) * .18)),
      [22, 24, 30], home(.6), 2),
    brick: style(103, [5, 10, 5, 12], (x, y) => [122, 68, 50].map(c => c * piers(x) * (.82 + grit(x, y, 64, 2) * .3)),
      [20, 20, 24], home(.64), 2.1),
  };

  const matte = (color: number, roughness = .85) =>
    new THREE.MeshStandardMaterial({ color, roughness, envMapIntensity: 0 });

  const glowing = (color: number, strength: number) =>
    new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(strength) });

  // One InstancedMesh per list of [x, y, z, sx, sy, sz, yaw] placements.
  const instanced = (geometry: THREE.BufferGeometry, material: THREE.Material, placements: number[][],
    shadows = true, tint?: () => THREE.Color) => {
    const mesh = new THREE.InstancedMesh(geometry, material, placements.length), matrix = new THREE.Matrix4();
    const turn = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);

    placements.forEach(([x, y, z, sx, sy, sz, yaw = 0], i) => {
      matrix.compose(new THREE.Vector3(x, y, z), turn.setFromAxisAngle(up, yaw), new THREE.Vector3(sx, sy, sz));
      mesh.setMatrixAt(i, matrix);
      if (tint) mesh.setColorAt(i, tint());
    });

    mesh.castShadow = mesh.receiveShadow = shadows;
    scene.add(mesh);
  };

  const floorHeight = .35, radius = 50;

  type Shape = "box" | "octagon" | "round";

  const boxes: { type: keyof typeof styles; shape: Shape; x: number; y: number; z: number; w: number; d: number;
    yaw: number; floors: number }[] = [];

  const roofs: number[][] = [], crowns: number[][] = [];

  const slabs: number[][] = [], tanks: number[][] = [], units: number[][] = [], antennas: number[][] = [];
  const beacons: number[][] = [], trees: number[][] = [], cornices: number[][] = [], lamps: number[][] = [];
  const headlights: number[][] = [], taillights: number[][] = [];

  // Two clusters rise out of a low city: downtown and a smaller one to the south-west, with noise between.
  const density = (x: number, z: number) => Math.max(Math.exp(-((x - 6) ** 2 + (z + 4) ** 2) / 260),
    .55 * Math.exp(-((x + 30) ** 2 + (z + 24) ** 2) / 120));

  // A diagonal avenue divides the districts; each keeps to its own side, and the city thins out at its edge.
  const boulevard = (x: number, z: number) => (x * .6 - z * .8 + 6);
  const inCity = (x: number, z: number) => Math.hypot(x, z) < radius + (noise(x + 64, z + 64, 4, 3) - .5) * 20;

  const districts = [{ yaw: 0, side: -1 }, { yaw: .52, side: 1 }];

  for (const { yaw, side } of districts) {
    const cos = Math.cos(yaw), sin = Math.sin(yaw);
    const world = (u: number, v: number) => [u * cos + v * sin, -u * sin + v * cos];
    const mine = (x: number, z: number) => side * boulevard(x, z) > 1.3 && inCity(x, z);

    // Streets at uneven spacing out from the middle, every so often an avenue.
    const lines = () => {
      const out = [{ c: 0, w: 3 }];

      for (const direction of [-1, 1])
        for (let c = 0; Math.abs(c) < radius + 14;) {
          c += direction * (5.5 + next() * 5);
          out.push({ c, w: next() < .2 ? 3 : 2 });
        }

      return out.sort((l, r) => l.c - r.c);
    };

    const us = lines(), vs = lines();

    // Lots come from splitting each block along its longer side until they are small, with alleys now and then.
    const split = (u0: number, v0: number, u1: number, v1: number): number[][] => {
      const w = u1 - u0, d = v1 - v0;
      if (Math.max(w, d) < 2.2 || (Math.max(w, d) < 4 && next() < .35))
        return [[u0, v0, u1, v1]];
      const at = .3 + next() * .4, gap = next() < .3 ? .2 : .04;

      return w > d ? [...split(u0, v0, u0 + w * at - gap, v1), ...split(u0 + w * at + gap, v0, u1, v1)]
        : [...split(u0, v0, u1, v0 + d * at - gap), ...split(u0, v0 + d * at + gap, u1, v1)];
    };

    for (let i = 1; i < us.length; i++)
      for (let j = 1; j < vs.length; j++) {
        const u0 = us[i - 1].c + us[i - 1].w / 2, u1 = us[i].c - us[i].w / 2;
        const v0 = vs[j - 1].c + vs[j - 1].w / 2, v1 = vs[j].c - vs[j].w / 2;
        const [cx, cz] = world((u0 + u1) / 2, (v0 + v1) / 2);
        if (!mine(cx, cz)) continue;
        const park = density(cx, cz) < .15 && next() < .07;

        // Lamps down the curbs, and cars in the lane beside them: headlights on two sides of the block, tail
        // lights on the other two, so each street carries both.
        const edges: [number, number, number, number, number][] = [[u0, v0, u1, v0, 0], [u1, v0, u1, v1, 1],
          [u1, v1, u0, v1, 2], [u0, v1, u0, v0, 3]];

        for (const [ua, va, ub, vb, edge] of edges) {
          const length = Math.hypot(ub - ua, vb - va), nu = (vb - va) / length, nv = -(ub - ua) / length;

          for (let t = .5; t < length; t += 1.7) {
            const u = ua + (ub - ua) * t / length, v = va + (vb - va) * t / length, [lx, lz] = world(u + nu * .1,
              v + nv * .1);

            if (mine(lx, lz)) lamps.push([lx, .45, lz, 1, 1, 1]);
          }

          for (let n = 0; n < length * .5; n++) {
            const t = next() * length, u = ua + (ub - ua) * t / length, v = va + (vb - va) * t / length;
            const [x, z] = world(u + nu * .55, v + nv * .55);
            if (mine(x, z))
              (edge < 2 ? headlights : taillights).push([x, .05, z, .16, .05, .08, yaw + (edge % 2 ? Math.PI / 2 : 0)]);
          }
        }

        for (const [lu0, lv0, lu1, lv1] of split(u0 + .3, v0 + .3, u1 - .3, v1 - .3)) {
          const lw = lu1 - lu0, ld = lv1 - lv0, [x0, z0] = world((lu0 + lu1) / 2, (lv0 + lv1) / 2);
          if (!mine(x0, z0)) continue;
          // The lot's sidewalk and yard: streets are what is left between them.
          slabs.push([x0, .02, z0, lw + .6, .04, ld + .6, yaw]);

          if (park) {
            for (let t = 0, n = Math.ceil(lw * ld); t < n; t++) {
              const r = .28 + next() * .25, [tx, tz] = world(lu0 + next() * lw, lv0 + next() * ld);
              trees.push([tx, r * .9 + .04, tz, r, r * 1.1, r]);
            }

            continue;
          }

          if (next() < .04) continue;
          // Each building takes most of its lot, set back by its own amount.
          const w = lw * (.66 + next() * .3), d = ld * (.66 + next() * .3);
          const [x, z] = world((lu0 + lu1) / 2 + (next() - .5) * (lw - w), (lv0 + lv1) / 2 + (next() - .5) * (ld - d));
          const center = density(x, z), rise = noise(x + 64, z + 64, 8, 3);
          // Mostly two to six floors, climbing gradually toward the centers.
          const floors = Math.max(2, Math.round(1 + rise * 5 + next() * 2 + center ** 1.5 * (6 + next() * 46)));

          const kind = next(), type: keyof typeof styles = floors > 24 ? (kind < .8 ? "glass" : "stone")
            : floors > 9 ? (kind < .3 ? "glass" : kind < .75 ? "stone" : "brick") : kind < .6 ? "brick" : "stone";

          const base = .04, add = (shape: Shape, bx: number, bz: number, bw: number, bd: number, y: number,
            count: number) => boxes.push({ type, shape, x: bx, y, z: bz, w: bw, d: bd, yaw, floors: count });

          let top = base + floors * floorHeight;

          if (floors <= 4 && center < .15 && next() < .8) {
            // A house: a pitched roof whose ridge runs along the longer side.
            add("box", x, z, w, d, base, floors);
            const span = Math.min(w, d), long = Math.max(w, d);
            roofs.push([x, top, z, span + .14, span * (.3 + next() * .2), long + .14, yaw + (w > d ? Math.PI / 2 : 0)]);
          } else if (floors > 16) {
            // Towers: octagonal, round, crowned with a pyramid, or square and stepping back as they rise.
            const form = next(), size = Math.min(w, d);

            if (form < .2 || form > .92) {
              add(form < .2 ? "octagon" : "round", x, z, size, size, base, floors);
            } else {
              add("box", x, z, w, d, base, floors);
              if (type !== "glass") cornices.push([x, top, z, w + .08, .12, d + .08, yaw]);

              if (form < .4) {
                crowns.push([x, top, z, w, size * (.5 + next() * .7), d, yaw]);
                top = -1;
              } else if (floors > 26) {
                const upper = Math.round(floors * (.15 + next() * .15));
                add("box", x, z, w * .74, d * .74, top, upper);
                top += upper * floorHeight;
              }
            }

            // The tallest carry a mast with a warning light.
            if (floors > 44 && top > 0) {
              const mast = 1.2 + next() * 2.5;
              antennas.push([x, top + mast / 2, z, 1, mast, 1]);
              beacons.push([x, top + mast, z, 1, 1, 1]);
            }
          } else {
            const form = next();

            if (form < .28) {
              // An L: a full-width wing along one side and a shorter one across the other.
              const [mx, mz] = world(0, -d * .25), side = next() < .5 ? -1 : 1;
              const [wx, wz] = world(side * w * .25, d * .25);
              add("box", x + mx, z + mz, w, d * .5, base, floors);
              add("box", x + wx, z + wz, w * .5, d * .5, base, Math.max(2, floors - Math.floor(next() * 3)));
            } else {
              add("box", x, z, w, d, base, floors);

              // Now and then a penthouse, set back on the roof.
              if (form > .82) {
                add("box", x, z, w * .5, d * .5, top, 1 + Math.floor(next() * 2));
              }
            }

            if (type !== "glass") cornices.push([x, top, z, w + .08, .12, d + .08, yaw]);
            // Low roofs keep their machinery: a water tank on its legs, a couple of air conditioners.
            const [ox, oz] = world((next() - .5) * w * .5, (next() - .5) * d * .5);
            if (floors < 14 && next() < .45) tanks.push([x + ox, top + .22, z + oz, 1, 1, 1]);

            for (let n = 0, count = Math.floor(next() * 3); n < count; n++) {
              const [ux, uz] = world((next() - .5) * w * .7, (next() - .5) * d * .7);
              units.push([x + ux, top + .06, z + uz, .2 + next() * .2, .12, .15 + next() * .12, yaw]);
            }
          }
        }
      }
  }

  // Buildings group by style, shape, floor count, width and depth: each group's geometry has UVs that fit its floors
  // and bays about 0.2 units wide all the way round, so no window stretches. A random offset into the map keeps
  // neighbors from repeating one another's lit windows.
  const groups = new Map<string, { type: keyof typeof styles; shape: Shape; floors: number; wide: number;
    deep: number; shift: number; placements: number[][] }>();

  for (const { type, shape, x, y, z, w, d, yaw, floors } of boxes) {
    const wide = Math.max(1, Math.round(w / .8)), deep = Math.max(1, Math.round(d / .8));
    const shift = Math.floor(next() * 4), key = `${type}:${shape}:${floors}:${wide}:${deep}:${shift}`;
    const group = groups.get(key) ?? { type, shape, floors, wide, deep, shift, placements: [] };
    group.placements.push([x, y, z, w, 1, d, yaw]);
    groups.set(key, group);
  }

  for (const { type, shape, floors, wide, deep, shift, placements } of groups.values()) {
    const tall = floors * floorHeight;

    const geometry = shape === "box" ? new THREE.BoxGeometry(1, tall, 1)
      : new THREE.CylinderGeometry(.5, .5, tall, shape === "round" ? 28 : 8);

    if (shape === "octagon") geometry.rotateY(Math.PI / 8);
    geometry.translate(0, tall / 2, 0);
    const uv = geometry.attributes.uv, normal = geometry.attributes.normal;
    // Four bays per 0.8 units, sixteen to a repeat; a cylinder's u runs once round its circumference.
    const around = Math.max(4, Math.round(Math.PI * wide * 4)) / 16;

    for (let i = 0; i < uv.count; i++) {
      const bays = shape !== "box" ? around : (Math.abs(normal.getX(i)) > .5 ? deep : wide) / 4;
      if (Math.abs(normal.getY(i)) > .5)
        uv.setXY(i, 1 / 256, 1 / 256);
      else
        uv.setXY(i, uv.getX(i) * bays + shift * 5 / 16, uv.getY(i) * floors / 16 + shift * 3 / 16);
    }

    const tint = new THREE.Color();
    instanced(geometry, styles[type], placements, true, () => tint.setHSL(0, 0, .8 + next() * .3));
  }

  // Pitched roofs in terracotta or slate, and pyramid crowns in dark copper.
  const prism = new THREE.ExtrudeGeometry(new THREE.Shape([new THREE.Vector2(-.5, 0), new THREE.Vector2(.5, 0),
    new THREE.Vector2(0, 1)]), { depth: 1, bevelEnabled: false });

  prism.translate(0, 0, -.5);
  const pyramid = new THREE.ConeGeometry(Math.SQRT1_2, 1, 4).rotateY(Math.PI / 4).translate(0, .5, 0);
  const shade = new THREE.Color();
  instanced(prism, matte(0xffffff, .8), roofs, true, () => next() < .55 ? shade.setHSL(.03, .5, .3 + next() * .12)
    : shade.setHSL(.6, .1, .26 + next() * .1));
  instanced(pyramid, new THREE.MeshStandardMaterial({ color: 0x3a4a44, roughness: .45, metalness: .6,
    envMapIntensity: 0 }), crowns);

  const unit = new THREE.BoxGeometry(1, 1, 1);
  instanced(unit, matte(0x34353a, .9), slabs);
  instanced(unit, matte(0x4a4744), cornices);
  instanced(new THREE.CylinderGeometry(.16, .16, .3, 12), matte(0x4a3a2c), tanks);
  instanced(unit, matte(0x55585c, .6), units);
  instanced(new THREE.CylinderGeometry(.03, .06, 1, 6), matte(0x3a3c40, .5), antennas);
  instanced(new THREE.SphereGeometry(.09, 8, 6), glowing(0xff2a1a, 6), beacons, false);
  instanced(new THREE.IcosahedronGeometry(1, 1), matte(0x1c3020, 1), trees);
  instanced(new THREE.BoxGeometry(.06, .06, .06), glowing(0xffb870, 5), lamps, false);
  instanced(unit, glowing(0xfff2dc, 4), headlights, false);
  instanced(unit, glowing(0xff2a20, 3), taillights, false);
  // Asphalt everywhere the lots leave open.
  scene.add(floor(matte(0x131418, .7), 220));

  // The evening sky from above, and from below the orange glow of the streets washing up the facades.
  const moon = new THREE.DirectionalLight(0x9fb2e0, .35);
  moon.position.set(-30, 40, -20);
  moon.castShadow = true;
  scene.add(moon, new THREE.HemisphereLight(0x55669a, 0x9a6a44, 2));

  return { scene, cameraPosition: [-44, 19, 18], target: [4, 5, -4], enclosed: true,
    settings: { background: { mode: "color", color: "#2a3452" }, vignette: .3, grain: .02, exposure: 1.35 } };
}

/** Every scene in the catalog, by id: the type makes a missing or misspelled one an error. */
export const SCENES: Record<SceneId, () => ExampleScene> = {
  studio: studioScene, room: roomScene, glass: glassScene, materials: materialsScene, hall: hallScene,
  textures: texturesScene, blending: blendingScene, shaders: shadersScene, galaxy: galaxyScene,
  neon: neonScene, road: roadScene, chess: chessScene, attic: atticScene, city: cityScene,
};
