import { SCENE_CATALOG, findScene } from "./catalog";
import type * as Viewer from "./viewer";
import tinypolyIcon from "../assets/tinypoly-icon.svg?raw";
import "./style.css";

// The page has two views: the gallery (no hash) and the viewer (`#<scene id>`). The hash makes each scene a link
// that the back button, a bookmark or a shared URL returns to.

const TITLE = document.title;
const gallery = document.getElementById("gallery")!;
const viewerRoot = document.getElementById("viewer")!;
const grid = document.getElementById("scene-grid")!;

// Inline, so the icon can take the surrounding text color.
for (const mark of document.querySelectorAll(".brand-mark"))
  mark.innerHTML = tinypolyIcon;

const ARROW = "<svg class=\"icon\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M5 12h14\" /><path d=\"m12 5 7 7-7 7\" /></svg>";

for (const scene of SCENE_CATALOG) {
  const card = document.createElement("a");
  card.className = "scene-card";
  card.href = `#${scene.id}`;

  const thumb = document.createElement("div");
  thumb.className = "scene-thumb";
  const image = new Image(960, 644);
  // Rendered by this library, not by WebGL: see scripts/capture-scenes.js.
  image.src = new URL(`../assets/scenes/${scene.id}.webp`, import.meta.url).href;
  image.alt = "";
  image.loading = "lazy";
  image.decoding = "async";
  thumb.append(image);

  const body = document.createElement("div");
  body.className = "scene-body";
  const title = document.createElement("h2");
  title.textContent = scene.title;
  title.insertAdjacentHTML("beforeend", ARROW);
  const description = document.createElement("p");
  description.textContent = scene.description;
  body.append(title, description);
  card.append(thumb, body);
  grid.append(card);
}

// The viewer module (Three.js, the renderer and its workers) loads on the first visit to a scene.
let viewer: Promise<typeof Viewer> | null = null;
let galleryScroll = 0;

async function route() {
  const scene = findScene(decodeURIComponent(location.hash.slice(1)));

  if (!scene) {
    if (gallery.hidden) {
      viewerRoot.hidden = true;
      gallery.hidden = false;
      window.scrollTo(0, galleryScroll);
    }

    document.title = TITLE;
    (await viewer)?.hideViewer();

    return;
  }

  if (!gallery.hidden)
    galleryScroll = window.scrollY;
  gallery.hidden = true;
  viewerRoot.hidden = false;
  document.title = `${scene.title} · ${TITLE}`;
  viewer ??= import("./viewer");
  (await viewer).showScene(scene.id);
}

window.addEventListener("hashchange", () => void route());
void route();
