// MFIstmas filteRSS – wersja przeglądarkowa.
//
// Obraz z kamery (getUserMedia) jest rysowany na <canvas> odbity lustrzanie,
// a efekty są nakładane bezpośrednio na niego. MediaPipe (Hand Landmarker
// i Face Detector) działa w przeglądarce (WebAssembly/WebGL) – obraz z kamery
// NIGDZIE nie jest wysyłany.

import {
  FilesetResolver,
  HandLandmarker,
  FaceDetector,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

import {
  FIST, WAVE, PEACE, THUMBS_UP,
  classifyGesture, WaveTracker, GestureStabilizer, Cooldown,
} from "./gesty.js";
import * as fx from "./efekty.js";

// =============================================================================
// KONFIGURACJA (te same wartości co w swiateczne_lustro.py)
// =============================================================================

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const HAND_MODEL = "models/hand_landmarker.task";
const FACE_MODEL = "models/blaze_face_short_range.tflite";

const COOLDOWNS = { [WAVE]: 3.0, [THUMBS_UP]: 2.0 }; // s – blokada ponownego wyzwolenia
const SANTA_DURATION = 3.4;   // czas przelotu sań (s)
const TREE_SHOW_TIME = 5.0;   // jak długo widać choinkę (s)
const HAT_SHOW_TIME = 8.0;    // gest V zakłada czapkę na tyle sekund
const HAT_WIDTH_SCALE = 1.6;  // szerokość czapki względem szerokości głowy
const HAT_FOREHEAD_UP = 0.85; // czoło: tyle rozstawów oczu nad linią oczu
const HAT_ANCHOR = [0.5, 0.85]; // punkt hat.png (ułamki szer./wys.), który trafia na czoło
const MAX_CANVAS_HEIGHT = 1080; // ograniczenie rozdzielczości rysowania (wydajność)

// Grafiki i czcionki leżą w grafiki/ i czcionki/. Na GitHub Pages są obok
// index.html; przy uruchomieniu lokalnym z folderu web/ – piętro wyżej.
const ASSET_BASES = ["grafiki/", "../grafiki/"];
const ASSETS = {
  santa: "santa.png",
  hat: "hat.png",
  tree: "tree.png",
  snowflake: "snowflake.png",
  logo: "logo_rss.png",
};
// Logo aplikacji – tylko na ekranie startowym i jako ikona karty przeglądarki (opcjonalne)
const APP_LOGO = "logo_filterss.png";

// =============================================================================
// ELEMENTY STRONY I STAN
// =============================================================================

const canvas = document.getElementById("scena");
const ctx = canvas.getContext("2d");
const video = document.getElementById("kamera");
const startScreen = document.getElementById("start");
const startBtn = document.getElementById("start-przycisk");
const statusEl = document.getElementById("status");

const state = {
  assets: null,
  handLandmarker: null,
  faceDetector: null,
  scene: null,
  debug: false,
  useFace: true,
  garland: true,
  hands: [],
  faces: null,
  active: new Set(),
  prevActive: new Set(),
  lastVideoTime: -1,
  prevT: performance.now() / 1000,
  fps: 0,
  fpsShown: 0,
  fpsShownT: 0,
  layout: null,
};

const waveTracker = new WaveTracker();
const stabilizer = new GestureStabilizer();
const cooldown = new Cooldown(COOLDOWNS);

// Do testów: window.lustro.forceGestures = new Set(["PEACE"]) wymusza aktywne gesty.
window.lustro = { state, forceGestures: null };

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("blad", isError);
}

// =============================================================================
// WCZYTYWANIE ZASOBÓW I MODELI
// =============================================================================

function loadImageFrom(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function loadImage(name) {
  for (const base of ASSET_BASES) {
    try {
      return await loadImageFrom(base + name);
    } catch {
      /* spróbuj następnej lokalizacji */
    }
  }
  console.warn(`Nie udało się wczytać ${name} – użyję grafiki zastępczej.`);
  return null;
}

async function showAppLogo() {
  const img = await loadImage(APP_LOGO);
  if (!img) return; // brak pliku -> zostaje napis "MFIstmas filteRSS"
  const src = fx.trimTransparent(img).toDataURL();
  document.getElementById("start-logo-aplikacji").src = src;
  document.querySelector(".karta").classList.add("ma-logo");
  const icon = document.createElement("link"); // ikona karty przeglądarki
  icon.rel = "icon";
  icon.href = src;
  document.head.appendChild(icon);
}

async function loadAssets() {
  const entries = await Promise.all(
    Object.entries(ASSETS).map(async ([key, file]) => [key, await loadImage(file)]),
  );
  const raw = Object.fromEntries(entries);
  // Przycięte przezroczyste marginesy – logo ma właściwą wielkość i jest wyśrodkowane
  if (raw.logo) document.getElementById("start-logo").src = fx.trimTransparent(raw.logo).toDataURL();
  showAppLogo(); // nie czekamy – logo aplikacji nie jest potrzebne do działania lustra
  return {
    santa: raw.santa ? fx.trimTransparent(raw.santa) : null, // brak -> baner "HO HO HO"
    hat: raw.hat ? fx.trimTransparent(raw.hat) : fx.fallbackHat(),
    tree: raw.tree ? fx.trimTransparent(raw.tree) : fx.fallbackTree(),
    snowflake: raw.snowflake ? fx.trimTransparent(raw.snowflake) : fx.fallbackSnowflake(),
    logo: raw.logo ? fx.trimTransparent(raw.logo) : null,
  };
}

/** Tworzy zadanie MediaPipe na GPU, a gdy się nie da – na CPU. */
async function createTask(TaskClass, vision, options) {
  for (const delegate of ["GPU", "CPU"]) {
    try {
      return await TaskClass.createFromOptions(vision, {
        ...options,
        baseOptions: { ...options.baseOptions, delegate },
      });
    } catch (err) {
      console.warn(`${TaskClass.name}: delegat ${delegate} niedostępny`, err);
    }
  }
  throw new Error(`Nie udało się uruchomić ${TaskClass.name}`);
}

async function loadModels() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const [handLandmarker, faceDetector] = await Promise.all([
    createTask(HandLandmarker, vision, {
      baseOptions: { modelAssetPath: HAND_MODEL },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    }),
    createTask(FaceDetector, vision, {
      baseOptions: { modelAssetPath: FACE_MODEL },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.6, // wyższy próg = mniej "twarzy" w tle
    }).catch((err) => {
      console.warn("Detekcja twarzy niedostępna – czapka będzie nad palcami.", err);
      return null;
    }),
  ]);
  return { handLandmarker, faceDetector };
}

// =============================================================================
// SCENA I UKŁAD OBRAZU
// =============================================================================

class Scene {
  constructor(w, h, assets) {
    Object.assign(this, { w, h });
    this.s = h / 540;
    this.hud = new fx.Hud(w, h, ctx);
    this.sparkles = new fx.Sparkles(this.s);
    this.snow = new fx.SnowEffect(w, h, assets.snowflake);
    this.santa = new fx.SantaEffect(w, h, assets.santa, SANTA_DURATION);
    this.hat = new fx.HatEffect(assets.hat, HAT_SHOW_TIME, HAT_ANCHOR, HAT_WIDTH_SCALE, HAT_FOREHEAD_UP);
    this.tree = new fx.TreeEffect(w, h, assets.tree, TREE_SHOW_TIME, this.hud.height + 6 * this.s);
    this.logo = assets.logo ? new fx.Logo(w, h, assets.logo) : null;
    this.garland = new fx.Garland(w, h, this.logo ? this.logo.left - this.logo.margin : null);
  }
}

/** Rozdzielczość płótna = rozmiar okna (z limitem), scena skalowana do niej. */
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const h = Math.min(Math.round(window.innerHeight * dpr), MAX_CANVAS_HEIGHT);
  const w = Math.round((h * window.innerWidth) / window.innerHeight);
  canvas.width = w;
  canvas.height = h;
  state.layout = null;
  if (state.assets) state.scene = new Scene(w, h, state.assets);
}

/**
 * Obraz z kamery wypełnia cały ekran (jak object-fit: cover) – skalowanie
 * jednakowe w poziomie i w pionie, więc obraz nie jest zniekształcony.
 */
function computeLayout() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const cw = canvas.width, ch = canvas.height;
  const s = Math.max(cw / vw, ch / vh);
  const dw = vw * s, dh = vh * s;
  return { vw, vh, cw, ch, dw, dh, ox: (cw - dw) / 2, oy: (ch - dh) / 2 };
}

/**
 * Współrzędne znormalizowane MediaPipe (0..1 względem klatki kamery) ->
 * piksele płótna PO odbiciu lustrzanym (x' = szerokość - x).
 */
function toCanvas(nx, ny) {
  const L = state.layout;
  return [L.cw - (L.ox + nx * L.dw), L.oy + ny * L.dh];
}

// =============================================================================
// DETEKCJA (dłonie + twarz) I LOGIKA WYZWALANIA
// =============================================================================

function detect(nowMs, now) {
  const scene = state.scene;
  const res = state.handLandmarker.detectForVideo(video, nowMs);
  state.hands = (res.landmarks || []).map((lms) => classifyGesture(lms.map((p) => toCanvas(p.x, p.y))));
  state.active = window.lustro.forceGestures ?? stabilizer.update(new Set(state.hands.map((h) => h.gesture)));

  // Twarz – tylko gdy czapka jest lub zaraz będzie widoczna (oszczędność mocy)
  state.faces = null;
  if (state.useFace && state.faceDetector && (state.active.has(PEACE) || scene.hat.visible(now))) {
    const L = state.layout;
    const det = state.faceDetector.detectForVideo(video, nowMs);
    state.faces = (det.detections || []).map((d) => {
      const e1 = toCanvas(d.keypoints[0].x, d.keypoints[0].y); // 0, 1 = oczy
      const e2 = toCanvas(d.keypoints[1].x, d.keypoints[1].y);
      const bb = d.boundingBox; // w pikselach klatki kamery
      const width = (bb.width / L.vw) * L.dw;
      const left = L.cw - (L.ox + ((bb.originX + bb.width) / L.vw) * L.dw);
      return {
        eyeLeft: e1[0] <= e2[0] ? e1 : e2,
        eyeRight: e1[0] <= e2[0] ? e2 : e1,
        box: { x: left, y: L.oy + (bb.originY / L.vh) * L.dh, width, height: (bb.height / L.vh) * L.dh },
      };
    });
  }

  const handWith = (g) => state.hands.find((h) => h.gesture === g) || null;

  // Rozbłysk iskier w miejscu dłoni, gdy gest właśnie się pojawił
  for (const g of state.active) {
    if (state.prevActive.has(g)) continue;
    const hi = handWith(g);
    if (!hi) continue;
    if (g === FIST) scene.sparkles.burst(...hi.palmCenter, 18, [fx.ICE, fx.WHITE]);
    else if (g === PEACE) scene.sparkles.burst((hi.pts[8][0] + hi.pts[12][0]) / 2, (hi.pts[8][1] + hi.pts[12][1]) / 2, 18);
    else if (g === THUMBS_UP) scene.sparkles.burst(...hi.pts[4], 18);
  }
  state.prevActive = state.active;

  const openHand = handWith(WAVE);
  if (waveTracker.update(now, state.active.has(WAVE) ? openHand : null) &&
      !scene.santa.active && cooldown.ready(WAVE, now)) {
    scene.santa.trigger(now);
    cooldown.fire(WAVE, now);
    if (openHand) scene.sparkles.burst(...openHand.palmCenter, 28);
  }
  if (state.active.has(THUMBS_UP) && cooldown.ready(THUMBS_UP, now)) {
    if (now > scene.tree.until) scene.sparkles.burst(...scene.tree.center(), 30);
    scene.tree.trigger(now);
    cooldown.fire(THUMBS_UP, now);
  }
  scene.hat.update(now, state.active.has(PEACE), handWith(PEACE), state.faces);
}

// =============================================================================
// PĘTLA GŁÓWNA
// =============================================================================

function frame() {
  requestAnimationFrame(frame);
  const scene = state.scene;
  if (!scene || video.readyState < 2 || !video.videoWidth) return;
  if (!state.layout) state.layout = computeLayout();

  const nowMs = performance.now();
  const now = nowMs / 1000;
  const dt = Math.min(now - state.prevT, 0.1);
  state.prevT = now;
  if (dt > 0) state.fps = state.fps ? 0.92 * state.fps + 0.08 / dt : 1 / dt;
  if (now - state.fpsShownT > 0.5) {
    state.fpsShown = Math.round(state.fps);
    state.fpsShownT = now;
  }

  // Detekcja tylko dla NOWEJ klatki z kamery
  if (video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    detect(nowMs, now);
  }
  scene.snow.update(dt, state.active.has(FIST), now);
  scene.sparkles.update(dt);

  // 1. BAZA: klatka z kamery, odbita lustrzanie – bez żadnych filtrów kolorów
  const L = state.layout;
  ctx.save();
  ctx.translate(L.cw, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, L.ox, L.oy, L.dw, L.dh);
  ctx.restore();

  // 2. Nakładki (od tyłu do przodu)
  if (state.debug) fx.drawDebug(ctx, state.hands, state.faces, scene.s);
  scene.snow.drawFar(ctx);
  scene.hat.draw(ctx, now);
  scene.tree.draw(ctx, now);
  scene.santa.draw(ctx, now, scene.sparkles, dt);
  scene.sparkles.draw(ctx, now);
  scene.snow.drawNear(ctx);
  if (state.garland) scene.garland.draw(ctx, now);
  if (scene.logo) scene.logo.draw(ctx);
  const info = `${state.fpsShown} FPS   ·   [d] debug   [f] twarz: ${state.useFace ? "wł." : "wył."}   ` +
    `[g] girlanda   [s] zdjęcie   [Enter] pełny ekran`;
  scene.hud.draw(ctx, state.active, { [WAVE]: cooldown.remaining(WAVE, now) / COOLDOWNS[WAVE] }, info);
}

// =============================================================================
// KAMERA, KLAWIATURA, START
// =============================================================================

function cameraErrorMessage(err) {
  if (!window.isSecureContext) return "Kamera działa tylko na stronie otwartej przez https:// lub localhost.";
  switch (err && err.name) {
    case "NotAllowedError":
      return "Brak zgody na użycie kamery. Zezwól na dostęp (ikona kamery lub kłódki w pasku adresu) i odśwież stronę.";
    case "NotFoundError":
      return "Nie znaleziono kamery. Podłącz kamerę i odśwież stronę.";
    case "NotReadableError":
      return "Kamera jest używana przez inny program (np. Teams, Zoom). Zamknij go i odśwież stronę.";
    default:
      return `Nie udało się uruchomić kamery (${(err && err.message) || err}).`;
  }
}

async function startCamera() {
  startBtn.disabled = true;
  setStatus("Uruchamianie kamery…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    setStatus(cameraErrorMessage(err), true);
    startBtn.disabled = false;
    return;
  }
  startScreen.hidden = true;
  toggleFullscreen(true);
  resize();
}

function toggleFullscreen(enter = !document.fullscreenElement) {
  if (enter && !document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {});
  else if (!enter && document.fullscreenElement) document.exitFullscreen?.();
}

function saveScreenshot() {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
  canvas.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `lustro_${stamp}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/png");
}

window.addEventListener("keydown", (e) => {
  if (startScreen.hidden === false) return;
  const k = e.key.toLowerCase();
  if (k === "d") state.debug = !state.debug;
  else if (k === "f" && state.faceDetector) state.useFace = !state.useFace;
  else if (k === "g") state.garland = !state.garland;
  else if (k === "s") saveScreenshot();
  else if (k === "enter") toggleFullscreen();
});
canvas.addEventListener("dblclick", () => toggleFullscreen());
window.addEventListener("resize", resize);
startBtn.addEventListener("click", startCamera);

async function init() {
  try {
    setStatus("Ładowanie modeli i grafik…");
    await Promise.all([
      document.fonts.load(`400 16px "DM Sans"`),
      document.fonts.load(`600 16px "DM Sans"`),
    ]).catch(() => {});
    const [assets, models] = await Promise.all([loadAssets(), loadModels()]);
    state.assets = assets;
    state.handLandmarker = models.handLandmarker;
    state.faceDetector = models.faceDetector;
    state.useFace = !!models.faceDetector;
    resize();
    setStatus("Gotowe! Kliknij przycisk i zezwól na dostęp do kamery.");
    startBtn.disabled = false;
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    setStatus(`Nie udało się wczytać aplikacji: ${err.message || err}. Sprawdź połączenie z internetem i odśwież stronę.`, true);
  }
}

init();
