// The gallery reads only this module, so the landing page loads without Three.js or the renderer.

export interface SceneInfo {
  id: string;
  title: string;
  description: string;
}

/** Gallery order. Each card shows `assets/scenes/<id>.webp`, rendered on the CPU by `scripts/capture-scenes.js`. */
export const SCENE_CATALOG = [
  {
    id: "studio",
    title: "Studio",
    description: "Physical, metallic and custom shader spheres beside a glass knot, under the sun and the sky.",
  },
  {
    id: "glass",
    title: "Glass and refraction",
    description: "Clear, dispersive, frosted and tinted glass bending the columns behind it.",
  },
  {
    id: "materials",
    title: "Material gallery",
    description: "Every shading model side by side, from Lambert and Toon to sheen, iridescence and anisotropy.",
  },
  {
    id: "chess",
    title: "Chess",
    description: "A game in progress on a lacquered board: turned ivory and ebony pieces, with depth of field.",
  },
  {
    id: "hall",
    title: "Hall",
    description: "A foggy hall where a spot light projects a stained-glass window onto the wall.",
  },
  {
    id: "attic",
    title: "Attic",
    description: "A shaft of afternoon sun through a round window, glowing in the dusty air of an old attic.",
  },
  {
    id: "textures",
    title: "Textures",
    description: "A sunlit workshop corner at real scale: brick, oak, crates and a chipped steel drum, all procedural maps.",
  },
  {
    id: "blending",
    title: "Blending",
    description: "Transparent panes over striped paint: normal, additive and multiply blending, and hashed alpha.",
  },
  {
    id: "shaders",
    title: "Custom shaders",
    description: "Four procedural GLSL materials interpreted on the CPU: marble, lava, a gas giant and pearl.",
  },
  {
    id: "galaxy",
    title: "Galaxy",
    description: "About 190 000 shader points: a warm bulge, spiral arms, pink nebulae and dust lanes that dim them.",
  },
  {
    id: "city",
    title: "City at dusk",
    description: "An instanced city at the blue hour: pitched-roof houses and mid-rises climbing to round and stepped towers.",
  },
  {
    id: "neon",
    title: "Neon",
    description: "A back street after rain: neon tubes on a brick wall, a lamp over a steel door and puddles catching the light.",
  },
  {
    id: "road",
    title: "Volumetric light",
    description: "A country road in the fog at night: one lamp pours a cold cone of light, cut by dead branches.",
  },
  {
    id: "room",
    title: "Room",
    description: "A closed box lit by a ceiling panel, with global illumination: light bounces and the walls bleed color.",
  },
] as const satisfies readonly SceneInfo[];

export type SceneId = (typeof SCENE_CATALOG)[number]["id"];

export const findScene = (id: string) => SCENE_CATALOG.find(scene => scene.id === id);
