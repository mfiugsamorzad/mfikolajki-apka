// Efekty aplikacji christmas filteRSS rysowane na <canvas> NA klatce z kamery.
// Odpowiednik lustro/efekty.py (wersja Python). Kolory w formacie RGB.

export const GOLD = [255, 200, 40];
export const WHITE = [255, 255, 255];
export const ICE = [180, 225, 255];
const LIGHT_COLORS = [
  [255, 60, 60], [255, 200, 40], [40, 150, 255], [230, 80, 255], [120, 255, 90],
];
const FONT = '"DM Sans", system-ui, sans-serif';

const rnd = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const rgba = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// =============================================================================
// NARZĘDZIA GRAFICZNE
// =============================================================================

export function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

/**
 * Przycina przezroczyste marginesy obrazka. Rozmiar i położenie grafik (logo
 * w rogu, czapka na czole) zależą wtedy od samej treści, a nie od pustego
 * miejsca zostawionego w pliku PNG.
 */
export function trimTransparent(img) {
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const c = makeCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[(y * w + x) * 4 + 3] > 8) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return c;
  const out = makeCanvas(x1 - x0 + 1, y1 - y0 + 1);
  out.getContext("2d").drawImage(c, x0, y0, out.width, out.height, 0, 0, out.width, out.height);
  return out;
}

/** Wersja obrazka przeskalowana do zadanej szerokości (stopniowo – lepsza jakość). */
export function scaled(src, width) {
  width = Math.max(2, Math.round(width));
  let cur = src;
  // Zmniejszanie "po połowie" – unika ząbków przy dużej różnicy rozmiarów
  while (cur.width / 2 > width) {
    const half = makeCanvas(cur.width / 2, cur.height / 2);
    const hctx = half.getContext("2d");
    hctx.imageSmoothingQuality = "high";
    hctx.drawImage(cur, 0, 0, half.width, half.height);
    cur = half;
  }
  const out = makeCanvas(width, (src.height * width) / src.width);
  const ctx = out.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(cur, 0, 0, out.width, out.height);
  return out;
}

export function roundRectPath(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Miękka poświata z jasnym środkiem (lampki, gwiazdy) – rysowana addytywnie. */
function makeGlow(radius, color) {
  const R = Math.ceil(radius * 2.2);
  const c = makeCanvas(2 * R, 2 * R);
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(R, R, 0, R, R, R);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.12, rgba(color, 1));
  g.addColorStop(0.4, rgba(color, 0.38));
  g.addColorStop(1, rgba(color, 0));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

/** Drobny płatek śniegu "w oddali": biały krążek z miękką krawędzią. */
function makeSoftDot(radius) {
  const R = Math.ceil(radius * 2);
  const c = makeCanvas(2 * R, 2 * R);
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(R, R, 0, R, R, R);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.45, "rgba(255,255,255,0.95)");
  g.addColorStop(0.6, "rgba(255,255,255,0.3)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

/** Czteroramienna iskra (rysowana addytywnie). */
function makeSparkle(size, color) {
  const S = Math.ceil(size);
  const c = makeCanvas(2 * S, 2 * S);
  const ctx = c.getContext("2d");
  const th = Math.max(1, size * 0.1);
  for (const horizontal of [true, false]) {
    const g = horizontal
      ? ctx.createLinearGradient(0, 0, 2 * S, 0)
      : ctx.createLinearGradient(0, 0, 0, 2 * S);
    g.addColorStop(0, rgba(color, 0));
    g.addColorStop(0.5, rgba(color, 1));
    g.addColorStop(1, rgba(color, 0));
    ctx.fillStyle = g;
    if (horizontal) ctx.fillRect(0, S - th / 2, 2 * S, th);
    else ctx.fillRect(S - th / 2, 0, th, 2 * S);
  }
  const r = ctx.createRadialGradient(S, S, 0, S, S, size * 0.35);
  r.addColorStop(0, "rgba(255,255,255,1)");
  r.addColorStop(0.4, rgba(color, 0.8));
  r.addColorStop(1, rgba(color, 0));
  ctx.fillStyle = r;
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

// --- Grafiki zastępcze (gdyby nie udało się wczytać plików PNG) ----------------

export function fallbackSnowflake(size = 128) {
  const c = makeCanvas(size, size);
  const ctx = c.getContext("2d");
  const m = size / 2;
  const R = size * 0.42;
  ctx.strokeStyle = "white";
  ctx.lineCap = "round";
  ctx.shadowColor = "rgba(180,225,255,0.9)";
  ctx.shadowBlur = size / 20;
  ctx.translate(m, m);
  for (let k = 0; k < 6; k++) {
    ctx.lineWidth = size / 28;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, -R);
    for (const [f, len] of [[0.42, 0.26], [0.68, 0.19]]) {
      for (const side of [-1, 1]) {
        ctx.moveTo(0, -R * f);
        ctx.lineTo(side * R * len * Math.sin(0.73), -R * f - R * len * Math.cos(0.73));
      }
    }
    ctx.stroke();
    ctx.rotate(Math.PI / 3);
  }
  return c;
}

export function fallbackHat() {
  const c = makeCanvas(340, 300);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgb(200,35,40)";
  ctx.beginPath();
  ctx.moveTo(45, 225);
  ctx.quadraticCurveTo(110, 50, 285, 58);
  ctx.quadraticCurveTo(255, 150, 295, 225);
  ctx.fill();
  ctx.fillStyle = "rgb(245,245,245)";
  roundRectPath(ctx, 22, 204, 296, 58, 28);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(293, 52, 30, 0, Math.PI * 2);
  ctx.fill();
  return c;
}

export function fallbackTree() {
  const c = makeCanvas(320, 400);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgb(125,75,35)";
  ctx.fillRect(140, 330, 40, 62);
  ctx.fillStyle = "rgb(35,150,55)";
  for (const [top, bottom, hw] of [[205, 345, 150], [150, 280, 125], [100, 215, 98], [60, 150, 70]]) {
    ctx.beginPath();
    ctx.moveTo(160, top);
    ctx.lineTo(160 - hw, bottom);
    ctx.lineTo(160 + hw, bottom);
    ctx.fill();
  }
  ctx.fillStyle = "rgb(255,205,30)";
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? 34 : 14;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    ctx.lineTo(160 + r * Math.cos(a), 44 + r * Math.sin(a));
  }
  ctx.fill();
  return c;
}

/** Animacja "wyskoku" z lekkim przestrzeleniem (0 -> 1). */
function easeOutBack(x) {
  x = clamp(x, 0, 1);
  const c1 = 1.70158;
  return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2;
}

/** Największy rozmiar czcionki (≤ size), przy którym tekst mieści się w maxW. */
export function fitFont(ctx, text, weight, size, maxW) {
  size = Math.floor(size);
  while (size > 7) {
    ctx.font = `${weight} ${size}px ${FONT}`;
    if (ctx.measureText(text).width <= maxW) break;
    size--;
  }
  return size;
}

// =============================================================================
// ISKRY
// =============================================================================

export class Sparkles {
  static COLORS = [GOLD, WHITE, ICE];
  static MAX = 350;

  constructor(s) {
    this.s = s;
    this.sprites = new Map();
    for (const col of Sparkles.COLORS) {
      this.sprites.set(col, [makeSparkle(7 * s, col), makeSparkle(12 * s, col)]);
    }
    this.items = [];
  }

  _emit(x, y, vx, vy, life, colors) {
    if (this.items.length >= Sparkles.MAX) return;
    const col = colors[Math.floor(Math.random() * colors.length)];
    const spr = this.sprites.get(col)[Math.random() < 0.5 ? 0 : 1];
    this.items.push({ x, y, vx, vy, life, max: life, spr, seed: Math.random() * 10 });
  }

  burst(x, y, n = 22, colors = [GOLD, WHITE]) {
    for (let i = 0; i < n; i++) {
      const a = rnd(0, Math.PI * 2);
      const v = rnd(90, 300) * this.s;
      this._emit(x, y, Math.cos(a) * v, Math.sin(a) * v, rnd(0.45, 0.9), colors);
    }
  }

  trail(x, y, n, spread, colors = [GOLD, WHITE]) {
    for (let i = 0; i < n; i++) {
      this._emit(x + rnd(-0.3, 0.3) * spread, y + rnd(-1, 1) * spread,
        rnd(-60, 10) * this.s, rnd(10, 60) * this.s, rnd(0.5, 1.1), colors);
    }
  }

  update(dt) {
    for (const p of this.items) {
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= 1 - 2.5 * dt; // opór powietrza
      p.vy *= 1 - 2.5 * dt;
      p.vy += 40 * this.s * dt; // lekka grawitacja
      p.life -= dt;
    }
    this.items = this.items.filter((p) => p.life > 0);
  }

  draw(ctx, t) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter"; // światło dodaje się do obrazu
    for (const p of this.items) {
      ctx.globalAlpha = clamp((p.life / p.max) * (0.75 + 0.25 * Math.sin(t * 25 + p.seed)), 0, 1);
      ctx.drawImage(p.spr, p.x - p.spr.width / 2, p.y - p.spr.height / 2);
    }
    ctx.restore();
  }
}

// =============================================================================
// ŚNIEG – dwie warstwy głębi + zaspa
// =============================================================================

class FlakeLayer {
  constructor(nVariants, maxN, rate, speed, sway, spin) {
    Object.assign(this, { nVariants, maxN, rate, speed, sway, spin });
    this.items = [];
    this.acc = 0;
  }

  spawn(dt, w) {
    this.acc += this.rate * dt;
    let n = Math.min(Math.floor(this.acc), this.maxN - this.items.length);
    this.acc -= Math.floor(this.acc);
    while (n-- > 0) {
      const v = Math.floor(Math.random() * this.nVariants);
      const depth = v / Math.max(1, this.nVariants - 1);
      this.items.push({
        x: rnd(-30, w + 30),
        y: rnd(-80, -10),
        vel: rnd(this.speed[0], this.speed[1]) * (0.7 + 0.6 * depth), // większe = szybsze
        v,
        phase: rnd(0, Math.PI * 2),
        op: 0.55 + 0.45 * depth * rnd(0.8, 1),
        ang: rnd(0, Math.PI * 2),
        spin: rnd(-this.spin, this.spin),
      });
    }
  }

  /** Ruch płatków; zwraca x płatków, które opadły na zaspę. */
  update(dt, t, w, ground) {
    const landed = [];
    this.items = this.items.filter((f) => {
      f.y += f.vel * dt;
      f.x += Math.sin(t * 1.3 + f.phase) * this.sway * dt;
      f.ang += f.spin * dt;
      if (f.y >= ground.surfaceY(f.x)) {
        landed.push(f.x);
        return false;
      }
      return f.x > -80 && f.x < w + 80;
    });
    return landed;
  }
}

class SnowGround {
  static BINS = 64;

  constructor(w, h) {
    this.w = w;
    this.h = h;
    this.maxH = 0.07 * h;
    this.heights = new Float32Array(SnowGround.BINS);
  }

  idx(x) {
    return clamp(Math.floor((x / this.w) * SnowGround.BINS), 0, SnowGround.BINS - 1);
  }

  surfaceY(x) {
    return this.h - this.heights[this.idx(x)];
  }

  deposit(xs, amount) {
    for (const x of xs) this.heights[this.idx(x)] += amount;
  }

  update(dt, snowing) {
    const hgt = this.heights;
    const n = hgt.length;
    const next = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const melted = snowing ? hgt[i] : hgt[i] - (this.maxH / 4) * dt; // ~4 s topnienia
      const l = hgt[Math.max(0, i - 1)], r = hgt[Math.min(n - 1, i + 1)];
      const smooth = 0.25 * l + 0.5 * hgt[i] + 0.25 * r;
      next[i] = clamp(0.85 * melted + 0.15 * smooth, 0, this.maxH);
    }
    this.heights = next;
  }

  draw(ctx) {
    let top = 0;
    for (const v of this.heights) top = Math.max(top, v);
    if (top < 1) return;
    const n = SnowGround.BINS;
    const step = this.w / n;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, this.h);
    ctx.lineTo(0, this.h - this.heights[0]);
    for (let i = 0; i < n; i++) {
      const x = (i + 0.5) * step;
      const nx = (i + 1.5) * step;
      const y = this.h - this.heights[i];
      const ny = this.h - this.heights[Math.min(n - 1, i + 1)];
      ctx.quadraticCurveTo(x, y, (x + nx) / 2, (y + ny) / 2);
    }
    ctx.lineTo(this.w, this.h);
    ctx.closePath();
    ctx.fillStyle = "rgba(246,249,255,0.93)";
    ctx.shadowColor = "rgba(150,180,220,0.6)";
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.restore();
  }
}

export class SnowEffect {
  constructor(w, h, flakeImg) {
    const s = h / 540;
    Object.assign(this, { w, h, s });
    this.dots = [1.4, 2.0, 2.8].map((r) => makeSoftDot(r * s));
    this.flakes = [14, 19, 26, 34].map((size) => scaled(flakeImg, size * s));
    this.far = new FlakeLayer(3, 180, 95, [45 * s, 80 * s], 25 * s, 0);
    this.near = new FlakeLayer(4, 70, 26, [70 * s, 120 * s], 45 * s, 1.2);
    this.ground = new SnowGround(w, h);
  }

  update(dt, active, t) {
    if (active) {
      this.far.spawn(dt, this.w);
      this.near.spawn(dt, this.w);
    }
    this.ground.deposit(this.far.update(dt, t, this.w, this.ground), 1.1 * this.s);
    this.ground.deposit(this.near.update(dt, t, this.w, this.ground), 2.4 * this.s);
    this.ground.update(dt, active);
  }

  drawFar(ctx) {
    ctx.save();
    for (const f of this.far.items) {
      const d = this.dots[f.v];
      ctx.globalAlpha = f.op;
      ctx.drawImage(d, f.x - d.width / 2, f.y - d.height / 2);
    }
    ctx.restore();
  }

  drawNear(ctx) {
    for (const f of this.near.items) {
      const img = this.flakes[f.v];
      ctx.save();
      ctx.globalAlpha = f.op;
      ctx.translate(f.x, f.y);
      ctx.rotate(f.ang);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);
      ctx.restore();
    }
    this.ground.draw(ctx);
  }
}

// =============================================================================
// SANIE MIKOŁAJA
// =============================================================================

export class SantaEffect {
  constructor(w, h, img, duration) {
    Object.assign(this, { w, h, duration });
    this.sprite = img ? scaled(img, w * 0.36) : null;
    this.startT = null;
  }

  get active() {
    return this.startT !== null;
  }

  trigger(now) {
    this.startT = now;
  }

  /** Tor lotu: wznoszący się łuk + delikatne "falowanie". */
  path(p) {
    return [
      -0.25 * this.w + p * 1.5 * this.w,
      this.h * (0.3 - 0.14 * Math.sin(p * Math.PI) + 0.02 * Math.sin(p * 5 * Math.PI)),
    ];
  }

  draw(ctx, now, sparkles, dt) {
    if (this.startT === null) return;
    const p = (now - this.startT) / this.duration;
    if (p >= 1) {
      this.startT = null;
      return;
    }
    const [x, y] = this.path(p);
    const [x2, y2] = this.path(Math.min(p + 0.01, 1));
    // Przechył zgodny z kierunkiem lotu (w canvas dodatni kąt = zgodnie z zegarem)
    const tilt = clamp(Math.atan2(y2 - y, x2 - x), -0.25, 0.25);
    let rear, spread;
    if (this.sprite) {
      const sw = this.sprite.width, sh = this.sprite.height;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(tilt);
      ctx.drawImage(this.sprite, -sw / 2, -sh / 2);
      ctx.restore();
      rear = [x - sw * 0.42, y + sh * 0.12];
      spread = sh * 0.18;
    } else {
      rear = this.drawBanner(ctx, x, y);
      spread = this.h * 0.03;
    }
    const n = Math.round(110 * dt + Math.random() - 0.5);
    sparkles.trail(rear[0], rear[1], Math.max(0, n), spread);
  }

  /** Zastępstwo za santa.png: czerwony baner "HO HO HO". */
  drawBanner(ctx, cx, cy) {
    const s = this.h / 540;
    const bw = 250 * s, bh = 70 * s;
    const x0 = cx - bw / 2, y0 = cy - bh / 2;
    ctx.save();
    ctx.fillStyle = "rgb(200,30,35)";
    ctx.shadowColor = "rgba(0,0,0,0.4)";
    ctx.shadowBlur = 8 * s;
    roundRectPath(ctx, x0, y0, bw, bh, 16 * s);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = rgba(GOLD);
    ctx.lineWidth = 2 * s;
    roundRectPath(ctx, x0 + 5 * s, y0 + 5 * s, bw - 10 * s, bh - 10 * s, 12 * s);
    ctx.stroke();
    const size = fitFont(ctx, "HO HO HO", 600, 40 * s, bw - 36 * s);
    ctx.font = `600 ${size}px ${FONT}`;
    ctx.fillStyle = "white";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("HO HO HO", cx, cy + 1);
    ctx.restore();
    return [x0, y0 + bh * 0.6];
  }
}

// =============================================================================
// CZAPKA MIKOŁAJA NA GŁOWIE
// =============================================================================

/**
 * Pozycja czapki z położenia oczu (jak lustro/twarz.py):
 *  - rozstaw oczu d = skala głowy,
 *  - nachylenie odcinka oko–oko = przechylenie głowy,
 *  - czoło leży ok. foreheadUp·d nad linią oczu, prostopadle do niej.
 */
export function headPose(face, foreheadUp) {
  const [el, er] = [face.eyeLeft, face.eyeRight];
  const vx = er[0] - el[0], vy = er[1] - el[1];
  const d = Math.max(Math.hypot(vx, vy), 1);
  const theta = Math.atan2(vy, vx);
  const up = [Math.sin(theta), -Math.cos(theta)];
  const cx = (el[0] + er[0]) / 2, cy = (el[1] + er[1]) / 2;
  return {
    forehead: [cx + up[0] * foreheadUp * d, cy + up[1] * foreheadUp * d],
    faceWidth: Math.max(face.box.width, 2.2 * d),
    angle: theta,
  };
}

export class HatEffect {
  static FADE = 0.25;
  static TRACK_TTL = 0.5;

  constructor(img, showTime, anchor, widthScale, foreheadUp) {
    this.img = scaled(img, Math.min(img.width, 700)); // rozsądne źródło do skalowania
    Object.assign(this, { showTime, anchor, widthScale, foreheadUp });
    this.tracks = []; // {x, y, width, angle, seen}
    this.since = -1e9;
    this.until = -1;
    this.hand = null;
    this.handSeen = -1e9;
  }

  visible(now) {
    return now <= this.until;
  }

  update(now, peaceActive, peaceHand, faces) {
    if (peaceActive) {
      if (!this.visible(now)) this.since = now;
      this.until = now + this.showTime;
    }
    if (!this.visible(now)) {
      this.tracks = [];
      this.hand = null;
      return;
    }
    if (faces) {
      for (const face of faces) {
        const pose = headPose(face, this.foreheadUp);
        const t = { x: pose.forehead[0], y: pose.forehead[1], width: pose.faceWidth * this.widthScale, angle: pose.angle };
        let best = null, bestD = t.width * 0.5;
        for (const tr of this.tracks) {
          const d = Math.hypot(tr.x - t.x, tr.y - t.y);
          if (d < bestD) [best, bestD] = [tr, d];
        }
        if (!best) {
          this.tracks.push({ ...t, seen: now });
        } else {
          // wygładzanie wykładnicze – czapka nie drga
          for (const k of ["x", "y", "width", "angle"]) best[k] = best[k] * 0.45 + t[k] * 0.55;
          best.seen = now;
        }
      }
      this.tracks = this.tracks.filter((tr) => now - tr.seen < HatEffect.TRACK_TTL);
    }
    if (peaceHand) {
      const p = peaceHand.pts;
      const tips = [(p[8][0] + p[12][0]) / 2, (p[8][1] + p[12][1]) / 2];
      const dx = tips[0] - peaceHand.palmCenter[0], dy = tips[1] - peaceHand.palmCenter[1];
      const t = { x: tips[0], y: tips[1], width: peaceHand.handSize * 2, angle: Math.atan2(dx, -dy) };
      if (!this.hand) this.hand = t;
      else for (const k of ["x", "y", "width", "angle"]) this.hand[k] = this.hand[k] * 0.6 + t[k] * 0.4;
      this.handSeen = now;
    }
  }

  /** Punkt `anchor` obrazka trafia w (x, y); całość obrócona o `angle` wokół tego punktu. */
  drawAnchored(ctx, x, y, width, angle, anchor, opacity) {
    const h = (this.img.height * width) / this.img.width;
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.drawImage(this.img, -anchor[0] * width, -anchor[1] * h, width, h);
    ctx.restore();
  }

  draw(ctx, now) {
    if (!this.visible(now)) return;
    const op = clamp(Math.min((now - this.since) / HatEffect.FADE, (this.until - now) / HatEffect.FADE), 0, 1);
    if (this.tracks.length) {
      for (const tr of this.tracks) this.drawAnchored(ctx, tr.x, tr.y, tr.width, tr.angle, this.anchor, op);
    } else if (this.hand && now - this.handSeen < 0.3) {
      this.drawAnchored(ctx, this.hand.x, this.hand.y, this.hand.width, this.hand.angle, [0.5, 0.95], op);
    }
  }
}

// =============================================================================
// CHOINKA
// =============================================================================

/**
 * Lampki rozmieszczone na NIEPRZEZROCZYSTYCH gałęziach grafiki (z pominięciem
 * czubka i pnia), z minimalnym odstępem – działa dla dowolnego tree.png.
 */
function sampleLights(img, n = 14) {
  const small = scaled(img, 160);
  const w = small.width, h = small.height;
  const a = small.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
  const alpha = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : a[(y * w + x) * 4 + 3]);
  const k = Math.max(2, Math.round(w * 0.04));
  const minD = Math.max(w, h) * 0.1;
  const chosen = [];
  let seed = 3;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647); // powtarzalne losowanie
  for (let tries = 0; tries < 6000 && chosen.length < n; tries++) {
    const x = Math.floor(rand() * w), y = Math.floor(h * (0.2 + rand() * 0.62));
    if ([[0, 0], [k, 0], [-k, 0], [0, k], [0, -k]].some(([dx, dy]) => alpha(x + dx, y + dy) < 200)) continue;
    if (chosen.every(([cx, cy]) => Math.hypot(cx - x, cy - y) > minD)) chosen.push([x, y]);
  }
  return chosen.map(([x, y]) => [x / w, y / h]);
}

function topPoint(img) {
  const small = scaled(img, 160);
  const w = small.width, h = small.height;
  const a = small.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
  for (let y = 0; y < h; y++) {
    const cols = [];
    for (let x = 0; x < w; x++) if (a[(y * w + x) * 4 + 3] > 128) cols.push(x);
    if (cols.length) {
      const yy = Math.min(h - 1, y + Math.max(2, Math.round(h * 0.03)));
      return [cols.reduce((s, v) => s + v, 0) / cols.length / w, yy / h];
    }
  }
  return [0.5, 0.05];
}

export class TreeEffect {
  constructor(w, h, img, showTime, bottomMargin) {
    const s = h / 540;
    Object.assign(this, { w, h, showTime });
    const treeH = h * 0.42;
    this.base = scaled(img, (treeH * img.width) / img.height);
    this.lights = sampleLights(img);
    this.star = topPoint(img);
    this.phases = this.lights.map(() => rnd(0, Math.PI * 2));
    this.glows = LIGHT_COLORS.map((c) => makeGlow(9 * s, c));
    this.starGlow = makeSparkle(34 * s, GOLD);
    this.right = w - 14 * s;
    this.bottom = h - bottomMargin;
    this.since = -1e9;
    this.until = -1;
  }

  trigger(now) {
    if (now > this.until) this.since = now;
    this.until = now + this.showTime;
  }

  center() {
    return [this.right - this.base.width / 2, this.bottom - this.h * 0.21];
  }

  draw(ctx, now) {
    if (now > this.until) return;
    const pop = easeOutBack((now - this.since) / 0.5);
    const opacity = clamp(Math.min((now - this.since) / 0.15, (this.until - now) / 0.6), 0, 1);
    const k = 0.3 + 0.7 * pop;
    const tw = this.base.width * k, th = this.base.height * k;
    const x = this.right - this.base.width / 2 - tw / 2;
    const y = this.bottom - th;
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.drawImage(this.base, x, y, tw, th);
    if (pop >= 0.7) {
      ctx.globalCompositeOperation = "lighter";
      this.lights.forEach(([u, v], i) => {
        const b = 0.5 + 0.5 * Math.sin(now * 2.8 + this.phases[i]);
        const g = this.glows[i % this.glows.length];
        ctx.globalAlpha = (0.25 + 0.75 * b * b) * opacity;
        ctx.drawImage(g, x + u * tw - g.width / 2, y + v * th - g.height / 2);
      });
      const sg = this.starGlow;
      ctx.globalAlpha = (0.55 + 0.45 * Math.sin(now * 4)) * opacity;
      ctx.drawImage(sg, x + this.star[0] * tw - sg.width / 2, y + this.star[1] * th - sg.height / 2);
    }
    ctx.restore();
  }
}

// =============================================================================
// GIRLANDA, LOGO, PASEK GESTÓW
// =============================================================================

export class Garland {
  constructor(w, h, xEnd = null, swags = 3, perSwag = 7) {
    const s = h / 540;
    const width = xEnd ?? w;
    this.sag = 0.07 * h;
    this.s = s;
    this.segments = [];
    this.bulbs = [];
    const seg = width / swags;
    const y0 = -3 * s;
    for (let k = 0; k < swags; k++) {
      this.segments.push([k * seg, (k + 1) * seg]);
      for (let j = 0; j < perSwag; j++) {
        const t = (j + 0.5) / perSwag;
        this.bulbs.push([k * seg + t * seg, y0 + this.sag * 4 * t * (1 - t) + 6 * s]);
      }
    }
    this.y0 = y0;
    this.glows = LIGHT_COLORS.map((c) => makeGlow(10 * s, c));
  }

  draw(ctx, t) {
    ctx.save();
    ctx.strokeStyle = "rgb(25,60,30)";
    ctx.lineWidth = Math.max(1, 2 * this.s);
    ctx.beginPath();
    for (const [a, b] of this.segments) {
      ctx.moveTo(a, this.y0);
      ctx.quadraticCurveTo((a + b) / 2, this.y0 + 2 * this.sag, b, this.y0);
    }
    ctx.stroke();
    this.bulbs.forEach(([x, y], i) => {
      const col = LIGHT_COLORS[i % LIGHT_COLORS.length];
      const k = Math.max(0, Math.sin(t * 2.2 - i * 0.55)) ** 2; // biegnąca fala światła
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.fillStyle = rgba(col.map((c) => Math.round(c * (0.35 + 0.4 * k))));
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, 3.5 * this.s), 0, Math.PI * 2);
      ctx.fill();
      const g = this.glows[i % this.glows.length];
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = 0.2 + 0.8 * k;
      ctx.drawImage(g, x - g.width / 2, y - g.height / 2);
    });
    ctx.restore();
  }
}

/** Logo samorządu w prawym górnym rogu, na jasnej półprzezroczystej podkładce. */
export class Logo {
  constructor(w, h, img, heightFrac = 0.13, maxWidthFrac = 0.24) {
    const s = h / 540;
    const width = Math.min((img.width * h * heightFrac) / img.height, w * maxWidthFrac);
    this.img = scaled(img, width);
    this.pad = 8 * s;
    this.margin = 12 * s;
    this.r = 12 * s;
    this.x = w - this.img.width - this.margin - this.pad;
    this.y = this.margin + this.pad;
  }

  get left() {
    return this.x - this.pad;
  }

  draw(ctx) {
    ctx.save();
    ctx.fillStyle = "rgba(250,250,250,0.82)";
    roundRectPath(ctx, this.x - this.pad, this.y - this.pad,
      this.img.width + 2 * this.pad, this.img.height + 2 * this.pad, this.r);
    ctx.fill();
    ctx.drawImage(this.img, this.x, this.y);
    ctx.restore();
  }
}

/** Dolny pasek: 4 kafelki gestów (aktywny podświetlony) + pigułka z FPS i skrótami. */
export class Hud {
  static CHIPS = [
    ["FIST", "PIĘŚĆ", "śnieg"],
    ["WAVE", "OTWARTA DŁOŃ", "sanie"],
    ["PEACE", "ZNAK V", "czapka"],
    ["THUMBS_UP", "KCIUK W GÓRĘ", "choinka"],
  ];

  constructor(w, h, ctx) {
    const s = (this.s = h / 540);
    this.margin = 12 * s;
    this.gap = 8 * s;
    this.chipH = 54 * s;
    const avail = w * 0.76 - this.margin;
    this.chipW = Math.min(175 * s, (avail - 3 * this.gap) / 4);
    this.y1 = h - this.margin;
    this.y0 = this.y1 - this.chipH;
    this.height = this.chipH + this.margin;
    const pad = this.chipW - 18 * s;
    this.labelSize = Math.min(...Hud.CHIPS.map(([, l]) => fitFont(ctx, l, 600, 11 * s, pad)));
    this.effectSize = Math.min(...Hud.CHIPS.map(([, , e]) => fitFont(ctx, e, 600, 21 * s, pad)));
  }

  draw(ctx, active, cooldownFrac, infoLine) {
    const { s, margin: x0, y0, chipW, chipH, gap } = this;
    const r = 14 * s;
    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    Hud.CHIPS.forEach(([g, label, effect], i) => {
      const cx0 = x0 + i * (chipW + gap);
      const on = active.has(g);
      roundRectPath(ctx, cx0, y0, chipW, chipH, r);
      ctx.fillStyle = on ? "rgba(185,35,40,0.72)" : "rgba(20,24,30,0.72)";
      ctx.fill();
      ctx.lineWidth = Math.max(1, 1.5 * s);
      ctx.strokeStyle = on ? rgba(GOLD) : "rgba(80,85,90,1)";
      ctx.stroke();
      ctx.fillStyle = on ? "rgb(255,225,200)" : "rgb(165,170,175)";
      ctx.font = `600 ${this.labelSize}px ${FONT}`;
      ctx.fillText(label, cx0 + chipW / 2, y0 + 15 * s);
      ctx.fillStyle = "white";
      ctx.font = `600 ${this.effectSize}px ${FONT}`;
      ctx.fillText(effect, cx0 + chipW / 2, y0 + 35 * s);
      const frac = cooldownFrac[g] || 0;
      if (frac > 0) { // pasek ładowania (cooldown)
        const bx0 = cx0 + r, bx1 = cx0 + chipW - r, by = this.y1 - 5 * s;
        ctx.lineCap = "round";
        ctx.lineWidth = Math.max(2, 3 * s);
        ctx.strokeStyle = "rgb(60,65,70)";
        ctx.beginPath(); ctx.moveTo(bx0, by); ctx.lineTo(bx1, by); ctx.stroke();
        ctx.strokeStyle = rgba(GOLD);
        ctx.beginPath(); ctx.moveTo(bx0, by); ctx.lineTo(bx0 + (bx1 - bx0) * (1 - frac), by); ctx.stroke();
      }
    });
    // Linia informacyjna na półprzezroczystej "pigułce"
    ctx.font = `400 ${13 * s}px ${FONT}`;
    ctx.textAlign = "left";
    const tw = ctx.measureText(infoLine).width;
    const ph = 22 * s, py = y0 - 6 * s - ph;
    roundRectPath(ctx, x0, py, tw + 16 * s, ph, ph / 2);
    ctx.fillStyle = "rgba(18,20,25,0.55)";
    ctx.fill();
    ctx.fillStyle = "rgb(235,235,235)";
    ctx.fillText(infoLine, x0 + 8 * s, py + ph / 2 + 1);
    ctx.restore();
  }
}

// =============================================================================
// TRYB PODGLĄDU (debug)
// =============================================================================

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

export function drawDebug(ctx, hands, faces, s) {
  ctx.save();
  for (const hi of hands) {
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2 * s;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(...hi.pts[a]);
      ctx.lineTo(...hi.pts[b]);
    }
    ctx.stroke();
    ctx.fillStyle = "rgb(255,60,60)";
    for (const p of hi.pts) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 3.5 * s, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = "rgb(255,255,0)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(hi.palmCenter[0], hi.palmCenter[1], hi.handSize * 0.95, 0, Math.PI * 2); // próg pięści
    ctx.stroke();
    const fingers = ["thumb", "index", "middle", "ring", "pinky"].map((f) => (hi.extended[f] ? 1 : 0)).join("");
    const minX = Math.min(...hi.pts.map((p) => p[0]));
    const minY = Math.min(...hi.pts.map((p) => p[1]));
    ctx.font = `600 ${18 * s}px ${FONT}`;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "black";
    ctx.strokeText(`${hi.gesture} [${fingers}]`, minX, Math.max(70 * s, minY - 8 * s));
    ctx.fillStyle = "rgb(255,255,0)";
    ctx.fillText(`${hi.gesture} [${fingers}]`, minX, Math.max(70 * s, minY - 8 * s));
  }
  for (const f of faces || []) {
    ctx.strokeStyle = "rgb(0,200,255)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(f.box.x, f.box.y, f.box.width, f.box.height);
    ctx.fillStyle = "rgb(0,200,255)";
    for (const e of [f.eyeLeft, f.eyeRight]) {
      ctx.beginPath();
      ctx.arc(e[0], e[1], 3 * s, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}
