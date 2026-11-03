// Rozpoznawanie gestów z 21 punktów dłoni (MediaPipe Hand Landmarker).
// Port 1:1 logiki z lustro/gesty.py – te same progi i ta sama geometria.
//
// Indeksy punktów:
//   0      – nadgarstek
//   1–4    – kciuk:      CMC, MCP, IP, TIP (opuszek)
//   5–8    – wskazujący: MCP, PIP, DIP, TIP
//   9–12   – środkowy
//   13–16  – serdeczny
//   17–20  – mały
// MCP = staw u nasady palca (knykieć), PIP = środkowy staw, TIP = opuszek.

export const FIST = "FIST";
export const WAVE = "WAVE";
export const PEACE = "PEACE";
export const THUMBS_UP = "THUMBS_UP";
export const NONE = "NONE";

// Trójki (MCP, PIP, TIP) dla czterech "długich" palców
const FINGERS = {
  index: [5, 6, 8],
  middle: [9, 10, 12],
  ring: [13, 14, 16],
  pinky: [17, 18, 20],
};
const LONG_FINGERS = ["index", "middle", "ring", "pinky"];

export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/**
 * Które palce są wyprostowane?
 *
 * Długie palce: zamiast porównywać współrzędne y (to działa tylko dla dłoni
 * skierowanej palcami w górę), porównujemy ODLEGŁOŚCI OD NADGARSTKA, niezależne
 * od obrotu dłoni:
 *   - palec wyprostowany -> opuszek leży o ponad 15% dalej od nadgarstka niż staw PIP,
 *   - palec zgięty       -> opuszek "zawija się" do dłoni i jest bliżej nadgarstka.
 *
 * Kciuk zgina się w innej płaszczyźnie, więc sprawdzamy, jak daleko jego opuszek
 * odstaje od nasady palca wskazującego (>0,55 rozmiaru dłoni) i od środka dłoni
 * (>0,7), oraz czy jest rozprostowany (TIP dalej od MCP kciuka niż staw IP).
 */
function fingerStates(pts, palmCenter, handSize) {
  const wrist = pts[0];
  const states = {};
  for (const name of LONG_FINGERS) {
    const [, pip, tip] = FINGERS[name];
    states[name] = dist(pts[tip], wrist) > dist(pts[pip], wrist) * 1.15;
  }
  const thumbOut =
    dist(pts[4], pts[5]) > 0.55 * handSize && dist(pts[4], palmCenter) > 0.7 * handSize;
  const thumbStraight = dist(pts[4], pts[2]) > dist(pts[3], pts[2]) * 1.2;
  states.thumb = thumbOut && thumbStraight;
  return states;
}

/**
 * Klasyfikacja gestu jednej dłoni (pts – 21 punktów [x, y] w pikselach).
 *
 * Skala dłoni = odległość nadgarstek (0) -> nasada palca środkowego (9); ta długość
 * prawie nie zmienia się przy zginaniu palców, więc wszystkie progi wyrażamy
 * względem niej – działa tak samo z bliska i z daleka.
 * Środek dłoni = średnia z nadgarstka i czterech knykci (0, 5, 9, 13, 17).
 */
export function classifyGesture(pts) {
  const palmIdx = [0, 5, 9, 13, 17];
  const palmCenter = [
    palmIdx.reduce((s, i) => s + pts[i][0], 0) / palmIdx.length,
    palmIdx.reduce((s, i) => s + pts[i][1], 0) / palmIdx.length,
  ];
  const handSize = Math.max(dist(pts[0], pts[9]), 1e-3);
  const ext = fingerStates(pts, palmCenter, handSize);
  const nLongExt = LONG_FINGERS.filter((f) => ext[f]).length;

  // Średnia odległość opuszków od środka dłoni (znormalizowana):
  // pięść ~0,5–0,8, otwarta dłoń ~1,3–1,9.
  const tipsToPalm =
    LONG_FINGERS.reduce((s, f) => s + dist(pts[FINGERS[f][2]], palmCenter), 0) / 4 / handSize;

  const info = (gesture) => ({ gesture, pts, palmCenter, handSize, extended: ext });

  // KCIUK W GÓRĘ – 4 palce zgięte, kciuk wyprostowany i skierowany pionowo w górę:
  // opuszek wyżej niż MCP kciuka o >0,5 rozmiaru dłoni, najwyższy punkt dłoni,
  // wektor MCP->TIP bliski pionowi. Test przed pięścią ("pięść z odstającym kciukiem").
  const tvx = pts[4][0] - pts[2][0];
  const tvy = pts[4][1] - pts[2][1];
  const minY = Math.min(...pts.map((p) => p[1]));
  if (
    nLongExt === 0 &&
    ext.thumb &&
    pts[2][1] - pts[4][1] > 0.5 * handSize &&
    pts[4][1] <= minY + 1 &&
    Math.abs(tvx) < -tvy
  ) {
    return info(THUMBS_UP);
  }

  // PIĘŚĆ – wszystkie 4 długie palce zgięte, opuszki blisko środka dłoni.
  if (nLongExt === 0 && tipsToPalm < 0.95) return info(FIST);

  // ZNAK V – wskazujący i środkowy wyprostowane, serdeczny i mały zgięte, a palce
  // rozchylone: opuszki (8, 12) dalej od siebie niż nasady (5, 9) razy 1,3.
  if (ext.index && ext.middle && !ext.ring && !ext.pinky) {
    if (dist(pts[8], pts[12]) > 1.3 * dist(pts[5], pts[9])) return info(PEACE);
  }

  // OTWARTA DŁOŃ – 4 długie palce wyprostowane, opuszki daleko od środka dłoni.
  if (nLongExt === 4 && tipsToPalm > 1.1) return info(WAVE);

  return info(NONE);
}

/**
 * Machanie = oscylacja środka dłoni w poziomie: ≥2 zmiany kierunku w ciągu 0,8 s
 * (ruchy mniejsze niż 0,15 rozmiaru dłoni ignorujemy). Sanie startują od razu przy
 * machaniu albo po przytrzymaniu otwartej dłoni przez 0,6 s.
 */
export class WaveTracker {
  static WINDOW_S = 0.8;
  static OPEN_HOLD = 0.6;

  constructor() {
    this.history = [];
    this.openSince = null;
  }

  update(now, openHand) {
    if (!openHand) {
      this.history = [];
      this.openSince = null;
      return false;
    }
    if (this.openSince === null) this.openSince = now;
    this.history.push([now, openHand.palmCenter[0], openHand.handSize]);
    while (this.history.length && now - this.history[0][0] > WaveTracker.WINDOW_S) {
      this.history.shift();
    }
    let reversals = 0;
    let direction = 0;
    let anchor = null;
    for (const [, x, size] of this.history) {
      if (anchor === null) {
        anchor = x;
        continue;
      }
      const delta = x - anchor;
      if (Math.abs(delta) > 0.15 * size) {
        const newDir = delta > 0 ? 1 : -1;
        if (direction !== 0 && newDir !== direction) reversals++;
        direction = newDir;
        anchor = x;
      }
    }
    return reversals >= 2 || now - this.openSince >= WaveTracker.OPEN_HOLD;
  }
}

/** Gest jest aktywny, gdy wystąpił w ≥60% z ostatnich 6 klatek (odsiewa pomyłki). */
export class GestureStabilizer {
  constructor(window = 6, ratio = 0.6) {
    this.window = window;
    this.ratio = ratio;
    this.frames = [];
  }

  update(gestures) {
    this.frames.push(gestures);
    if (this.frames.length > this.window) this.frames.shift();
    const counts = new Map();
    for (const set of this.frames) for (const g of set) counts.set(g, (counts.get(g) || 0) + 1);
    const out = new Set();
    for (const [g, c] of counts) if (g !== NONE && c / this.frames.length >= this.ratio) out.add(g);
    return out;
  }
}

/** Blokada czasowa (cooldown) ponownego wyzwolenia efektu. */
export class Cooldown {
  constructor(durations) {
    this.durations = durations;
    this.last = {};
  }
  ready(key, now) {
    return now - (this.last[key] ?? -1e9) >= (this.durations[key] ?? 0);
  }
  fire(key, now) {
    this.last[key] = now;
  }
  remaining(key, now) {
    return Math.max(0, (this.durations[key] ?? 0) - (now - (this.last[key] ?? -1e9)));
  }
}
