# -*- coding: utf-8 -*-
"""
Grafika i efekty świątecznego lustra.

Najważniejsze decyzje wizualne / wydajnościowe:

* Wszystkie grafiki są trzymane jako *sprite'y* w postaci "premultiplied alpha"
  (kolor już przemnożony przez przezroczystość, float32). Dzięki temu:
    - skalowanie i obracanie nie tworzy ciemnych "obwódek" na krawędziach
      (klasyczny problem przy interpolacji PNG z przezroczystym tłem),
    - nakładanie to jedno mnożenie i dodawanie: dst = src + dst * (1 - alfa),
    - te same sprite'y służą do mieszania addytywnego (świecące lampki, iskry).
* Mieszamy wyłącznie mały wycinek klatki (ROI) pod sprite'em – nigdy całą klatkę.
* Przeskalowane / obrócone warianty liczymy raz i trzymamy w pamięci podręcznej.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

RNG = np.random.default_rng()

# Paleta (BGR)
GOLD = (40, 200, 255)
WHITE = (255, 255, 255)
ICE = (255, 225, 180)
RED = (45, 45, 215)
DARK = (25, 20, 18)
LIGHT_COLORS = [(60, 60, 255), (40, 200, 255), (255, 150, 40), (220, 80, 255), (90, 255, 120)]


# =============================================================================
# SPRITE – podstawowy "klocek" graficzny
# =============================================================================

class Sprite:
    """Obraz z kanałem alfa w formacie premultiplied float32 (H, W, 4)."""

    __slots__ = ("pa", "h", "w")

    def __init__(self, pa: np.ndarray):
        self.pa = np.ascontiguousarray(pa, dtype=np.float32)
        self.h, self.w = self.pa.shape[:2]

    @classmethod
    def from_bgra(cls, img: np.ndarray) -> "Sprite":
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return cls(np.concatenate([img[:, :, :3].astype(np.float32) * a, a], axis=2))

    # --- transformacje ----------------------------------------------------
    def resized(self, width: float) -> "Sprite":
        width = max(2, int(round(width)))
        height = max(2, int(round(self.h * width / self.w)))
        interp = cv2.INTER_AREA if width < self.w else cv2.INTER_LINEAR
        return Sprite(cv2.resize(self.pa, (width, height), interpolation=interp))

    def rotated(self, angle_deg: float) -> "Sprite":
        """Obrót (stopnie, dodatni = przeciwnie do wskazówek zegara) z powiększeniem płótna."""
        if abs(angle_deg) < 0.2:
            return self
        cx, cy = self.w / 2, self.h / 2
        m = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw = int(self.h * sin + self.w * cos + 0.5)
        nh = int(self.h * cos + self.w * sin + 0.5)
        m[0, 2] += nw / 2 - cx
        m[1, 2] += nh / 2 - cy
        out = cv2.warpAffine(self.pa, m, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return Sprite(out)

    # --- rysowanie --------------------------------------------------------
    def _clip(self, dst: np.ndarray, x: float, y: float):
        H, W = dst.shape[:2]
        x, y = int(round(x)), int(round(y))
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + self.w, W), min(y + self.h, H)
        if x0 >= x1 or y0 >= y1:
            return None
        return dst[y0:y1, x0:x1], self.pa[y0 - y:y1 - y, x0 - x:x1 - x]

    def draw(self, dst: np.ndarray, x: float, y: float, opacity: float = 1.0) -> None:
        """Normalne nakładanie (alpha blending), (x, y) = lewy górny róg."""
        if opacity <= 0.01:
            return
        c = self._clip(dst, x, y)
        if c is None:
            return
        roi, s = c
        color, alpha = s[:, :, :3], s[:, :, 3:4]
        if opacity < 0.999:
            color, alpha = color * opacity, alpha * opacity
        roi[:] = (color + roi * (1.0 - alpha)).astype(np.uint8)

    def draw_center(self, dst, cx, cy, opacity=1.0) -> None:
        self.draw(dst, cx - self.w / 2, cy - self.h / 2, opacity)

    def add_center(self, dst, cx, cy, intensity=1.0) -> None:
        """Mieszanie addytywne (światło): dst = min(dst + kolor * intensywność, 255)."""
        if intensity <= 0.01:
            return
        c = self._clip(dst, cx - self.w / 2, cy - self.h / 2)
        if c is None:
            return
        roi, s = c
        roi[:] = np.minimum(roi + s[:, :, :3] * intensity, 255.0).astype(np.uint8)


class SpriteCache:
    """Przeskalowane warianty sprite'a (szerokość kwantyzowana do 4 px)."""

    def __init__(self, sprite: Sprite, max_items: int = 48):
        self.sprite = sprite
        self.max_items = max_items
        self._cache: dict = {}

    def get(self, width: float) -> Sprite:
        key = max(4, int(width) // 4 * 4)
        out = self._cache.get(key)
        if out is None:
            if len(self._cache) >= self.max_items:
                self._cache.pop(next(iter(self._cache)))
            out = self._cache[key] = self.sprite.resized(key)
        return out


# =============================================================================
# PROCEDURALNE SPRITE'Y (światła, iskry, płatki) i GRAFIKI ZASTĘPCZE
# =============================================================================

def _radial_grid(radius_px: float, extent: float = 2.0):
    r = int(math.ceil(radius_px * extent))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    return xx, yy, np.sqrt(xx * xx + yy * yy)


def make_glow(radius: float, color, core: float = 0.28) -> Sprite:
    """Miękka poświata z jasnym, prawie białym środkiem – lampki, gwiazdy."""
    _, _, d = _radial_grid(radius, 2.2)
    halo = np.exp(-(d / radius) ** 2 * 1.6)
    hot = np.exp(-(d / max(radius * core, 0.8)) ** 2)
    col = np.array(color, np.float32)
    rgb = col * halo[..., None] + (255.0 - col) * hot[..., None] * 0.9
    return Sprite(np.dstack([rgb, np.clip(halo + hot, 0, 1)]))


def make_soft_dot(radius: float) -> Sprite:
    """Płatek "daleki": biały krążek z miękką krawędzią i delikatną aureolą."""
    _, _, d = _radial_grid(radius, 2.0)
    disc = np.clip(radius + 0.7 - d, 0, 1)
    a = np.clip(disc * 0.95 + np.exp(-(d / (radius * 1.7)) ** 2) * 0.25, 0, 1)
    return Sprite(np.dstack([np.full(d.shape + (3,), 255.0, np.float32) * a[..., None], a]))


def make_sparkle(size: float, color) -> Sprite:
    """Czteroramienna iskra (do mieszania addytywnego)."""
    xx, yy, d = _radial_grid(size, 1.0)
    th = max(size * 0.07, 0.7)
    arms = np.maximum(np.exp(-np.abs(xx) / (size * 0.35)) * np.exp(-(yy / th) ** 2),
                      np.exp(-np.abs(yy) / (size * 0.35)) * np.exp(-(xx / th) ** 2))
    core = np.exp(-(d / (size * 0.18)) ** 2)
    a = np.clip(arms + core, 0, 1)
    col = np.array(color, np.float32)
    rgb = col * a[..., None] + (255.0 - col) * core[..., None] * 0.8
    return Sprite(np.dstack([rgb, a]))


def make_fallback_snowflake(size: int = 128) -> np.ndarray:
    """Sześcioramienna śnieżynka z rozgałęzieniami (BGRA) – gdy brak snowflake.png."""
    mask = np.zeros((size, size), np.uint8)
    c = size / 2
    R = size * 0.42
    sh = 4  # rysowanie z dokładnością subpikselową (shift=4 -> 1/16 px)
    S = 1 << sh

    def line(p, q, t):
        cv2.line(mask, (int(p[0] * S), int(p[1] * S)), (int(q[0] * S), int(q[1] * S)),
                 255, t, cv2.LINE_AA, sh)

    for k in range(6):
        a = math.radians(90 + k * 60)
        ux, uy = math.cos(a), -math.sin(a)
        tip = (c + ux * R, c + uy * R)
        line((c, c), tip, max(2, size // 28))
        for frac, blen in ((0.42, 0.26), (0.68, 0.19)):
            base = (c + ux * R * frac, c + uy * R * frac)
            for side in (-1, 1):
                b = a + side * math.radians(42)
                line(base, (base[0] + math.cos(b) * R * blen, base[1] - math.sin(b) * R * blen),
                     max(2, size // 40))
        cv2.circle(mask, (int(tip[0] * S), int(tip[1] * S)), int(size * 0.025 * S), 255, -1,
                   cv2.LINE_AA, sh)
    hexagon = [(c + math.cos(math.radians(30 + 60 * i)) * R * 0.14,
                c + math.sin(math.radians(30 + 60 * i)) * R * 0.14) for i in range(6)]
    cv2.fillPoly(mask, [np.array(hexagon, np.int32)], 255, cv2.LINE_AA)

    m = mask.astype(np.float32) / 255.0
    glow = cv2.GaussianBlur(m, (0, 0), size / 40) * 0.55
    alpha = np.maximum(m, glow)
    w = m[..., None]
    bgr = np.array(ICE, np.float32) * (1 - w) + np.array(WHITE, np.float32) * w
    return np.dstack([bgr, alpha * 255]).astype(np.uint8)


def _bezier(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.array(p, np.float32) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def make_fallback_hat() -> np.ndarray:
    """Czapka Mikołaja (BGRA): zakrzywiony stożek z cieniem, futrzany otok, pompon."""
    W, H = 340, 300
    img = np.zeros((H, W, 4), np.uint8)
    tip = (285, 58)
    left = _bezier((45, 225), (110, 50), tip)
    right = _bezier(tip, (255, 150), (295, 225))
    cone = np.vstack([left, right]).astype(np.int32)
    cv2.fillPoly(img, [cone], (40, 35, 200, 255), cv2.LINE_AA)
    shade = np.vstack([_bezier(tip, (255, 150), (295, 225)),
                       _bezier((220, 225), (225, 140), tip)]).astype(np.int32)
    cv2.fillPoly(img, [shade], (30, 25, 150, 255), cv2.LINE_AA)
    cv2.polylines(img, [_bezier((70, 200), (120, 75), (260, 65)).astype(np.int32)], False,
                  (80, 75, 235, 255), 5, cv2.LINE_AA)

    # Futrzany otok – zaokrąglony prostokąt + "puszyste" kółka na krawędziach
    fur, fur_shadow = (245, 245, 245, 255), (215, 212, 210, 255)
    rounded_rect(img, (22, 208), (318, 272), 30, fur_shadow)
    rounded_rect(img, (22, 204), (318, 262), 28, fur)
    rng = np.random.default_rng(7)
    for _ in range(70):
        x, y = int(rng.uniform(30, 310)), int(rng.choice([rng.uniform(203, 210), rng.uniform(256, 266)]))
        cv2.circle(img, (x, y), int(rng.uniform(5, 10)), fur, -1, cv2.LINE_AA)
    for _ in range(40):
        x, y = int(rng.uniform(40, 300)), int(rng.uniform(215, 255))
        cv2.circle(img, (x, y), int(rng.uniform(2, 4)), (228, 226, 225, 255), -1, cv2.LINE_AA)

    # Pompon
    px, py = tip[0] + 8, tip[1] - 6
    cv2.circle(img, (px + 4, py + 5), 32, fur_shadow, -1, cv2.LINE_AA)
    cv2.circle(img, (px, py), 30, fur, -1, cv2.LINE_AA)
    for k in range(14):
        a = k / 14 * 2 * math.pi
        cv2.circle(img, (int(px + math.cos(a) * 27), int(py + math.sin(a) * 27)), 8, fur, -1, cv2.LINE_AA)
    return img


def make_fallback_tree() -> np.ndarray:
    """Choinka (BGRA): piętra z falistym dołem, cieniowanie, pień, gwiazda, łańcuch."""
    W, H = 320, 400
    img = np.zeros((H, W, 4), np.uint8)
    cx = W // 2
    cv2.rectangle(img, (cx - 20, 330), (cx + 20, 392), (35, 75, 125, 255), -1)
    cv2.rectangle(img, (cx + 4, 330), (cx + 20, 392), (25, 55, 95, 255), -1)

    tiers = [(205, 345, 150), (150, 280, 125), (100, 215, 98), (60, 150, 70)]
    for top, bottom, hw in tiers:
        xs = np.linspace(cx - hw, cx + hw, 41)
        bumps = bottom + 9 * np.abs(np.sin(np.linspace(0, 5 * math.pi, 41)))
        bottom_edge = np.column_stack([xs, bumps])
        poly = np.vstack([[[cx, top]], bottom_edge]).astype(np.int32)
        cv2.fillPoly(img, [poly], (55, 150, 35, 255), cv2.LINE_AA)
        right = np.vstack([[[cx, top]], bottom_edge[20:]]).astype(np.int32)  # prawa połowa w cieniu
        cv2.fillPoly(img, [right], (38, 112, 25, 255), cv2.LINE_AA)
        # złoty łańcuch przewieszony przez piętro
        chain = _bezier((cx - hw * 0.75, bottom - (bottom - top) * 0.28),
                        (cx, bottom + 6), (cx + hw * 0.7, bottom - (bottom - top) * 0.45), 30)
        cv2.polylines(img, [chain.astype(np.int32)], False, (70, 205, 245, 255), 3, cv2.LINE_AA)

    # Bombki
    for (bx, by, col) in [(118, 300, (50, 50, 220)), (205, 255, (230, 140, 40)),
                          (140, 190, (40, 190, 250)), (190, 125, (50, 50, 220)), (95, 240, (200, 80, 220))]:
        cv2.circle(img, (bx, by), 10, col + (255,), -1, cv2.LINE_AA)
        cv2.circle(img, (bx - 3, by - 3), 3, (255, 255, 255, 255), -1, cv2.LINE_AA)

    # Gwiazda
    star = []
    for i in range(10):
        r = 34 if i % 2 == 0 else 14
        a = -math.pi / 2 + i * math.pi / 5
        star.append([cx + r * math.cos(a), 44 + r * math.sin(a)])
    star = np.array(star, np.int32)
    cv2.fillPoly(img, [star], (30, 205, 255, 255), cv2.LINE_AA)
    cv2.polylines(img, [star], True, (0, 150, 230, 255), 2, cv2.LINE_AA)
    return img


def rounded_rect(img, p0, p1, r, color, thickness=-1) -> None:
    """Prostokąt z zaokrąglonymi rogami (wypełniony lub obrys)."""
    x0, y0 = map(int, p0)
    x1, y1 = map(int, p1)
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        lt = cv2.LINE_AA
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, lt)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, lt)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, lt)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, lt)
        for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                              ((x0 + r, y1 - r), 90), ((x1 - r, y1 - r), 0)):
            cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color, thickness, lt)


def ease_out_back(x: float) -> float:
    """Animacja "wyskoku" z lekkim przestrzeleniem (0 -> 1)."""
    x = min(max(x, 0.0), 1.0)
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


# =============================================================================
# ISKRY (system cząsteczek addytywnych)
# =============================================================================

class Sparkles:
    """Złote / lodowe iskry: smuga za saniami i "rozbłyski" po rozpoznaniu gestu."""

    COLORS = [GOLD, WHITE, ICE]
    MAX = 350

    def __init__(self, scale: float):
        self.s = scale
        self.sprites = [make_sparkle(sz * scale, col) for col in self.COLORS for sz in (7, 12)]
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.max_life = np.zeros(0, np.float32)
        self.var = np.zeros(0, np.int32)

    def _emit(self, pos, vel, life, colors):
        n = len(pos)
        if n == 0 or len(self.pos) >= self.MAX:
            return
        col_idx = RNG.choice([self.COLORS.index(c) for c in colors], n)
        var = (col_idx * 2 + RNG.integers(0, 2, n)).astype(np.int32)
        self.pos = np.vstack([self.pos, pos]).astype(np.float32)
        self.vel = np.vstack([self.vel, vel]).astype(np.float32)
        self.life = np.concatenate([self.life, life]).astype(np.float32)
        self.max_life = np.concatenate([self.max_life, life]).astype(np.float32)
        self.var = np.concatenate([self.var, var])

    def burst(self, x, y, n=22, colors=(GOLD, WHITE)) -> None:
        ang = RNG.uniform(0, 2 * math.pi, n)
        spd = RNG.uniform(90, 300, n) * self.s
        vel = np.column_stack([np.cos(ang) * spd, np.sin(ang) * spd])
        pos = np.tile([x, y], (n, 1)) + RNG.normal(0, 4, (n, 2))
        self._emit(pos, vel, RNG.uniform(0.45, 0.9, n), colors)

    def trail(self, x, y, n, spread, colors=(GOLD, WHITE)) -> None:
        if n <= 0:
            return
        pos = np.column_stack([np.full(n, x) + RNG.normal(0, spread * 0.3, n),
                               np.full(n, y) + RNG.normal(0, spread, n)])
        vel = np.column_stack([RNG.uniform(-60, 10, n), RNG.uniform(10, 60, n)]) * self.s
        self._emit(pos, vel, RNG.uniform(0.5, 1.1, n), colors)

    def update(self, dt: float) -> None:
        if len(self.pos) == 0:
            return
        self.pos += self.vel * dt
        self.vel *= (1.0 - 2.5 * dt)          # opór powietrza
        self.vel[:, 1] += 40 * self.s * dt    # lekka grawitacja
        self.life -= dt
        keep = self.life > 0
        self.pos, self.vel, self.life = self.pos[keep], self.vel[keep], self.life[keep]
        self.max_life, self.var = self.max_life[keep], self.var[keep]

    def draw(self, frame: np.ndarray, t: float) -> None:
        for (x, y), life, ml, v, i in zip(self.pos, self.life, self.max_life, self.var,
                                          range(len(self.var))):
            k = life / ml
            twinkle = 0.75 + 0.25 * math.sin(t * 25 + i)
            self.sprites[v].add_center(frame, x, y, k * twinkle)


# =============================================================================
# ŚNIEG
# =============================================================================

class FlakeLayer:
    """Jedna warstwa płatków (tablice NumPy, wektorowa fizyka)."""

    def __init__(self, n_variants, max_n, rate, speed, sway, spin):
        self.n_variants, self.max_n, self.rate = n_variants, max_n, rate
        self.speed, self.sway, self.spin_range = speed, sway, spin
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros(0, np.float32)
        self.var = np.zeros(0, np.int32)
        self.phase = np.zeros(0, np.float32)
        self.op = np.zeros(0, np.float32)
        self.ang = np.zeros(0, np.float32)
        self.spin = np.zeros(0, np.float32)
        self._acc = 0.0

    def spawn(self, dt, w):
        self._acc += self.rate * dt
        n = min(int(self._acc), self.max_n - len(self.pos))
        self._acc -= int(self._acc)
        if n <= 0:
            return
        var = RNG.integers(0, self.n_variants, n).astype(np.int32)
        # Większe płatki opadają szybciej i są bardziej nieprzezroczyste – złudzenie głębi
        depth = var / max(1, self.n_variants - 1)
        vel = RNG.uniform(*self.speed, n) * (0.7 + 0.6 * depth)
        pos = np.column_stack([RNG.uniform(-30, w + 30, n), RNG.uniform(-80, -10, n)])
        self.pos = np.vstack([self.pos, pos]).astype(np.float32)
        self.vel = np.concatenate([self.vel, vel]).astype(np.float32)
        self.var = np.concatenate([self.var, var])
        self.phase = np.concatenate([self.phase, RNG.uniform(0, 2 * math.pi, n)]).astype(np.float32)
        self.op = np.concatenate([self.op, 0.55 + 0.45 * depth * RNG.uniform(0.8, 1.0, n)]).astype(np.float32)
        self.ang = np.concatenate([self.ang, RNG.uniform(0, 360, n)]).astype(np.float32)
        self.spin = np.concatenate([self.spin, RNG.uniform(-self.spin_range, self.spin_range, n)]).astype(np.float32)

    def update(self, dt, t, w, ground) -> np.ndarray:
        """Ruch płatków. Zwraca pozycje x płatków, które właśnie opadły na zaspę."""
        if len(self.pos) == 0:
            return np.zeros(0, np.float32)
        self.pos[:, 1] += self.vel * dt
        self.pos[:, 0] += np.sin(t * 1.3 + self.phase) * self.sway * dt
        self.ang += self.spin * dt
        landed = self.pos[:, 1] >= ground.surface_y(self.pos[:, 0])
        out = (self.pos[:, 0] < -80) | (self.pos[:, 0] > w + 80)
        landed_x = self.pos[landed, 0].copy()
        keep = ~(landed | out)
        for name in ("pos", "vel", "var", "phase", "op", "ang", "spin"):
            setattr(self, name, getattr(self, name)[keep])
        return landed_x


class SnowGround:
    """
    Zaspa śniegu rosnąca u dołu ekranu, gdy pada; topnieje, gdy śnieg ustaje.
    Wysokość trzymana w ~64 "kubełkach" i wygładzana, a rysowana jako jeden
    wielokąt, mieszany wyłącznie w pasie u dołu kadru.
    """

    BINS = 64

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.max_h = 0.07 * h
        self.heights = np.zeros(self.BINS, np.float32)
        self.centers = (np.arange(self.BINS) + 0.5) * w / self.BINS

    def _idx(self, xs):
        return np.clip((xs / self.w * self.BINS).astype(np.int32), 0, self.BINS - 1)

    def surface_y(self, xs):
        return self.h - self.heights[self._idx(xs)]

    def deposit(self, xs, amount):
        if len(xs):
            np.add.at(self.heights, self._idx(xs), amount)

    def update(self, dt, snowing):
        hgt = self.heights
        if not snowing:
            hgt -= self.max_h / 4.0 * dt  # ~4 s do całkowitego stopnienia
        smooth = hgt.copy()
        smooth[1:-1] = 0.25 * hgt[:-2] + 0.5 * hgt[1:-1] + 0.25 * hgt[2:]
        self.heights = np.clip(0.85 * hgt + 0.15 * smooth, 0, self.max_h)

    def draw(self, frame):
        top_h = float(self.heights.max())
        if top_h < 1.0:
            return
        y0 = int(self.h - top_h - 4)
        strip = frame[y0:self.h]
        layer = strip.copy()
        xs = np.linspace(0, self.w, self.BINS * 4)
        ys = self.h - np.interp(xs, self.centers, self.heights) - y0
        pts = np.vstack([[[0, self.h - y0]], np.column_stack([xs, ys]), [[self.w, self.h - y0]]])
        cv2.fillPoly(layer, [pts.astype(np.int32)], (250, 247, 244), cv2.LINE_AA)
        cv2.polylines(layer, [np.column_stack([xs, ys + 3]).astype(np.int32)], False,
                      (235, 220, 205), 2, cv2.LINE_AA)  # błękitnawy cień pod krawędzią
        cv2.addWeighted(layer, 0.93, strip, 0.07, 0, dst=strip)


class SnowEffect:
    """
    Śnieg w dwóch warstwach głębi:
      * "daleka" – dużo drobnych, miękkich kropek (wolniejsze, półprzezroczyste),
      * "bliska" – obracające się śnieżynki (snowflake.png lub proceduralna).
    Warstwa daleka rysowana jest POD saniami, bliska NAD nimi – daje to głębię.
    """

    N_ANGLES = 24  # obrót płatków kwantyzowany co 15°, warianty liczone raz

    def __init__(self, w, h, flake_img: np.ndarray):
        s = h / 540
        self.w, self.h, self.s = w, h, s
        self.dots = [make_soft_dot(r * s) for r in (1.4, 2.0, 2.8)]
        base = Sprite.from_bgra(flake_img)
        self.flakes = []
        for size in (14, 19, 26, 34):
            sized = base.resized(size * s)
            self.flakes.append([sized.rotated(a * 360 / self.N_ANGLES) for a in range(self.N_ANGLES)])
        self.far = FlakeLayer(3, 180, 95, (45, 80), 25 * s, 0)
        self.near = FlakeLayer(4, 70, 26, (70, 120), 45 * s, 70)
        self.far.speed = tuple(v * s for v in self.far.speed)
        self.near.speed = tuple(v * s for v in self.near.speed)
        self.ground = SnowGround(w, h)

    def update(self, dt, active, t):
        if active:
            self.far.spawn(dt, self.w)
            self.near.spawn(dt, self.w)
        self.ground.deposit(self.far.update(dt, t, self.w, self.ground), 1.1 * self.s)
        self.ground.deposit(self.near.update(dt, t, self.w, self.ground), 2.4 * self.s)
        self.ground.update(dt, active)

    def draw_far(self, frame):
        L = self.far
        for (x, y), v, op in zip(L.pos, L.var, L.op):
            self.dots[v].draw_center(frame, x, y, op)

    def draw_near(self, frame):
        L = self.near
        step = 360 / self.N_ANGLES
        for (x, y), v, op, ang in zip(L.pos, L.var, L.op, L.ang):
            self.flakes[v][int(ang // step) % self.N_ANGLES].draw_center(frame, x, y, op)
        self.ground.draw(frame)


# =============================================================================
# SANIE MIKOŁAJA
# =============================================================================

class SantaEffect:
    """
    Sanie przelatują łukiem z lewej do prawej; przechylają się zgodnie z kierunkiem
    lotu i zostawiają smugę złotych iskier. Bez santa.png – czerwony baner
    z napisem "HO HO HO" (fallback ze specyfikacji, w estetycznej formie).
    """

    def __init__(self, w, h, img: np.ndarray | None, duration: float, text):
        self.w, self.h, self.duration = w, h, duration
        self.sprite = Sprite.from_bgra(img).resized(w * 0.36) if img is not None else None
        self.text = text
        self.start_t = None

    @property
    def active(self):
        return self.start_t is not None

    def trigger(self, now):
        self.start_t = now

    def _path(self, p):
        """Tor lotu: wznoszący się łuk + delikatne "falowanie"."""
        x = -0.25 * self.w + p * 1.5 * self.w
        y = self.h * (0.30 - 0.14 * math.sin(p * math.pi) + 0.02 * math.sin(p * 5 * math.pi))
        return x, y

    def draw(self, frame, now, sparkles: Sparkles, dt):
        if self.start_t is None:
            return
        p = (now - self.start_t) / self.duration
        if p >= 1.0:
            self.start_t = None
            return
        x, y = self._path(p)
        x2, y2 = self._path(min(p + 0.01, 1.0))
        tilt = max(-14.0, min(14.0, -math.degrees(math.atan2(y2 - y, x2 - x))))

        if self.sprite is not None:
            spr = self.sprite.rotated(tilt)
            spr.draw_center(frame, x, y)
            rear = (x - spr.w * 0.42, y + spr.h * 0.12)
            spread = spr.h * 0.18
        else:
            rear, spread = self._draw_banner(frame, x, y, tilt), self.h * 0.03
        sparkles.trail(rear[0], rear[1], RNG.poisson(110 * dt), spread)

    def _draw_banner(self, frame, cx, cy, tilt):
        s = self.h / 540
        bw, bh = int(250 * s), int(70 * s)
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        # Płozy sań pod banerem
        runner = _bezier((x0 + 10 * s, y0 + bh + 14 * s), (x0 + bw * 0.5, y0 + bh + 22 * s),
                         (x0 + bw + 22 * s, y0 + bh - 2 * s))
        cv2.polylines(frame, [runner.astype(np.int32)], False, GOLD, max(2, int(4 * s)), cv2.LINE_AA)
        for fx in (0.25, 0.7):
            cv2.line(frame, (int(x0 + bw * fx), y0 + bh), (int(x0 + bw * fx), int(y0 + bh + 18 * s)),
                     GOLD, max(2, int(3 * s)), cv2.LINE_AA)
        rounded_rect(frame, (x0 + 4, y0 + 5), (x0 + bw + 4, y0 + bh + 5), 16 * s, (15, 15, 90))
        rounded_rect(frame, (x0, y0), (x0 + bw, y0 + bh), 16 * s, (35, 30, 200))
        rounded_rect(frame, (x0 + 5, y0 + 5), (x0 + bw - 5, y0 + bh - 5), 12 * s, GOLD, max(1, int(2 * s)))
        size = self.text.fit_size("HO HO HO", bw - 36 * s, int(40 * s), bold=True)
        self.text.draw(frame, "HO HO HO", (x0 + bw / 2, y0 + bh / 2), size, WHITE,
                       anchor="mm", bold=True, shadow=True)
        return x0, y0 + bh * 0.6


# =============================================================================
# CZAPKA MIKOŁAJA
# =============================================================================

class HatEffect:
    """
    Czapka Mikołaja nakładana NA GŁOWĘ osoby w kadrze.

    Pozycja, skala i obrót pochodzą z detekcji twarzy (twarz.py -> HeadPose):
      * szerokość czapki = width_scale × szerokość głowy,
      * punkt kotwiczenia czapki (anchor_uv – ułamek szerokości/wysokości
        obrazka, domyślnie środek otoku) ląduje na czole,
      * czapka obraca się razem z przechyleniem głowy (linia oczu).
    Każda twarz ma swój "tor" z wygładzaniem wykładniczym (czapka nie drga).

    Gest V "zakłada" czapkę na show_time sekund (jak filtr na Instagramie –
    można opuścić rękę i pozować). Gdy w kadrze nie ma twarzy, czapka siedzi
    na palcach pokazujących V.
    """

    FADE = 0.25
    TRACK_TTL = 0.5  # tyle sekund tor twarzy przeżywa bez detekcji

    def __init__(self, img: np.ndarray, show_time: float, anchor_uv=(0.5, 0.85),
                 width_scale: float = 1.6, forehead_up: float = 0.85):
        self.cache = SpriteCache(Sprite.from_bgra(img))
        self.show_time = show_time
        self.anchor_uv = anchor_uv
        self.width_scale = width_scale
        self.forehead_up = forehead_up
        self.tracks: list = []        # [stan np.array([x, y, szer, kąt]), ostatnia detekcja]
        self.since, self.until = -1e9, -1.0
        self.hand_state = None
        self.hand_seen = -1e9

    def visible(self, now) -> bool:
        return now <= self.until

    def update(self, now, peace_active, peace_hand, faces, head_pose_fn):
        if peace_active:
            if not self.visible(now):
                self.since = now
            self.until = now + self.show_time
        if not self.visible(now):
            self.tracks, self.hand_state = [], None
            return

        if faces is not None:  # None = w tej klatce nie uruchamialiśmy detekcji
            for face in faces:
                pose = head_pose_fn(face, self.forehead_up)
                target = np.array([pose.forehead[0], pose.forehead[1],
                                   pose.face_width * self.width_scale, pose.angle], np.float32)
                best, best_d = None, target[2] * 0.5
                for tr in self.tracks:
                    d = float(np.hypot(*(tr[0][:2] - target[:2])))
                    if d < best_d:
                        best, best_d = tr, d
                if best is None:
                    self.tracks.append([target, now])
                else:
                    best[0] = best[0] * 0.45 + target * 0.55
                    best[1] = now
            self.tracks = [t for t in self.tracks if now - t[1] < self.TRACK_TTL]

        if peace_hand is not None:
            tips = (peace_hand.pts[8] + peace_hand.pts[12]) / 2
            d = tips - peace_hand.palm_center
            angle = math.degrees(math.atan2(d[0], -d[1]))  # 0 = palce pionowo w górę
            target = np.array([tips[0], tips[1], peace_hand.hand_size * 2.0, angle], np.float32)
            self.hand_state = target if self.hand_state is None else self.hand_state * 0.6 + target * 0.4
            self.hand_seen = now

    def _draw_anchored(self, frame, x, y, width, angle, anchor_uv, opacity):
        """
        Rysuje czapkę tak, by punkt anchor_uv obrazka trafił w (x, y), a całość
        była obrócona o `angle` stopni (zgodnie z ruchem wskazówek zegara).
        Obracamy wektor "środek obrazka -> punkt kotwiczenia" o ten sam kąt
        i odejmujemy go od punktu docelowego, otrzymując środek obróconej czapki.
        """
        hat = self.cache.get(width)
        ox = (anchor_uv[0] - 0.5) * hat.w
        oy = (anchor_uv[1] - 0.5) * hat.h
        t = math.radians(angle)
        rx = ox * math.cos(t) - oy * math.sin(t)
        ry = ox * math.sin(t) + oy * math.cos(t)
        hat.rotated(-angle).draw_center(frame, x - rx, y - ry, opacity)

    def draw(self, frame, now):
        if not self.visible(now):
            return
        opacity = max(0.0, min(1.0, (now - self.since) / self.FADE, (self.until - now) / self.FADE))
        if self.tracks:
            for (x, y, width, angle), _ in self.tracks:
                self._draw_anchored(frame, x, y, width, angle, self.anchor_uv, opacity)
        elif self.hand_state is not None and now - self.hand_seen < 0.3:
            x, y, width, angle = self.hand_state
            self._draw_anchored(frame, x, y, width, angle, (0.5, 0.95), opacity)


# =============================================================================
# CHOINKA
# =============================================================================

def sample_light_uv(img: np.ndarray, n: int = 14, seed: int = 3):
    """
    Rozmieszcza lampki na NIEPRZEZROCZYSTYM obszarze grafiki choinki (alfa > 200),
    z pominięciem czubka i pnia oraz z minimalnym odstępem między lampkami.
    Działa dla dowolnego tree.png – lampki zawsze trafiają w gałęzie.
    """
    h, w = img.shape[:2]
    mask = (img[:, :, 3] > 200).astype(np.uint8)
    k = max(3, int(w * 0.05)) | 1
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero(mask)
    sel = (ys > h * 0.2) & (ys < h * 0.82)
    ys, xs = ys[sel], xs[sel]
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))[:5000]
    min_d = max(w, h) * 0.1
    chosen = []
    for i in order:
        p = (xs[i], ys[i])
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > min_d ** 2 for q in chosen):
            chosen.append(p)
            if len(chosen) >= n:
                break
    return [(x / w, y / h) for x, y in chosen]


def top_point_uv(img: np.ndarray):
    """Najwyższy nieprzezroczysty punkt grafiki – tam błyszczy gwiazda."""
    a = img[:, :, 3] > 128
    rows = np.nonzero(a.any(axis=1))[0]
    if len(rows) == 0:
        return 0.5, 0.05
    r = rows[0] + max(2, int(img.shape[0] * 0.03))
    cols = np.nonzero(a[min(r, img.shape[0] - 1)])[0]
    return (cols.mean() / img.shape[1] if len(cols) else 0.5), r / img.shape[0]


class TreeEffect:
    """Choinka w prawym dolnym rogu: "wyskok" przy pojawieniu, mrugające lampki, gwiazda."""

    def __init__(self, w, h, img: np.ndarray, show_time: float, bottom_margin: int):
        self.w, self.h = w, h
        self.show_time = show_time
        tree_h = h * 0.42
        self.base_w = tree_h * img.shape[1] / img.shape[0]
        self.cache = SpriteCache(Sprite.from_bgra(img))
        self.lights = sample_light_uv(img)
        self.star_uv = top_point_uv(img)
        s = h / 540
        self.glows = [make_glow(9 * s, c) for c in LIGHT_COLORS]
        self.star = make_sparkle(34 * s, GOLD)
        self.right = w - 14 * s
        self.bottom = h - bottom_margin
        self.since, self.until = -1e9, -1.0
        self.phases = RNG.uniform(0, 2 * math.pi, max(1, len(self.lights)))

    def trigger(self, now):
        if now > self.until:
            self.since = now
        self.until = now + self.show_time

    def center(self):
        return self.right - self.base_w / 2, self.bottom - self.h * 0.21

    def draw(self, frame, now):
        if now > self.until:
            return
        pop = ease_out_back((now - self.since) / 0.5)
        opacity = max(0.0, min(1.0, (now - self.since) / 0.15, (self.until - now) / 0.6))
        spr = self.cache.get(self.base_w * (0.3 + 0.7 * pop))
        x = self.right - self.base_w / 2 - spr.w / 2
        y = self.bottom - spr.h
        spr.draw(frame, x, y, opacity)
        if pop < 0.7:
            return
        # Lampki: każda "oddycha" własnym rytmem (sinusoida z losową fazą)
        for i, (u, v) in enumerate(self.lights):
            k = 0.5 + 0.5 * math.sin(now * 2.8 + self.phases[i])
            self.glows[i % len(self.glows)].add_center(frame, x + u * spr.w, y + v * spr.h,
                                                     (0.25 + 0.75 * k * k) * opacity)
        su, sv = self.star_uv
        self.star.add_center(frame, x + su * spr.w, y + sv * spr.h,
                             (0.55 + 0.45 * math.sin(now * 4.0)) * opacity)


# =============================================================================
# DEKORACJE STAŁE: girlanda lampek + winieta
# =============================================================================

class Garland:
    """Girlanda kolorowych lampek zwisająca łukami u góry ekranu (efekt "biegnącego światła")."""

    def __init__(self, w, h, x_end: float | None = None, swags: int = 3, per_swag: int = 7):
        s = h / 540
        w = w if x_end is None else x_end
        self.sag = 0.07 * h
        self.wire = []
        self.bulbs = []
        seg = w / swags
        for k in range(swags):
            t = np.linspace(0, 1, 30)
            xs = k * seg + t * seg
            ys = -3 * s + self.sag * 4 * t * (1 - t)
            self.wire.append(np.column_stack([xs, ys]).astype(np.int32))
            for j in range(per_swag):
                tt = (j + 0.5) / per_swag
                self.bulbs.append((k * seg + tt * seg, -3 * s + self.sag * 4 * tt * (1 - tt) + 6 * s))
        self.glows = [make_glow(10 * s, c, core=0.22) for c in LIGHT_COLORS]
        self.r = max(2, int(3.5 * s))
        self.thick = max(1, int(2 * s))

    def draw(self, frame, t):
        cv2.polylines(frame, self.wire, False, (25, 55, 25), self.thick, cv2.LINE_AA)
        for i, (x, y) in enumerate(self.bulbs):
            col = LIGHT_COLORS[i % len(LIGHT_COLORS)]
            k = max(0.0, math.sin(t * 2.2 - i * 0.55)) ** 2  # fala światła biegnąca w prawo
            cv2.circle(frame, (int(x), int(y)), self.r, tuple(int(c * (0.35 + 0.4 * k)) for c in col),
                       -1, cv2.LINE_AA)
            self.glows[i % len(self.glows)].add_center(frame, x, y, 0.2 + 0.8 * k)


# =============================================================================
# INTERFEJS (logo + pasek gestów)
# =============================================================================

class Logo:
    """Logo samorządu (logo_rss.png) stale w prawym górnym rogu, z przezroczystością."""

    def __init__(self, w, h, img: np.ndarray, height_frac: float = 0.13, max_width_frac: float = 0.22):
        s = h / 540
        spr = Sprite.from_bgra(img)
        width = min(spr.w * (h * height_frac) / spr.h, w * max_width_frac)
        self.sprite = spr.resized(width)
        self.margin = int(12 * s)
        self.x = w - self.sprite.w - self.margin
        self.y = self.margin

    def draw(self, frame):
        self.sprite.draw(frame, self.x, self.y)


class Hud:
    """
    Dolny pasek z czterema "kafelkami" gestów: aktywny gest jest podświetlony,
    kafelek sań pokazuje ładowanie cooldownu. Nad paskiem – linia z FPS
    i skrótami klawiszowymi. Wszystkie napisy: DM Sans (tekst.py).
    """

    # (gest, nazwa gestu, efekt)
    CHIPS = [("FIST", "PIĘŚĆ", "śnieg"),
             ("WAVE", "OTWARTA DŁOŃ", "sanie"),
             ("PEACE", "ZNAK V", "czapka"),
             ("THUMBS_UP", "KCIUK W GÓRĘ", "choinka")]

    def __init__(self, w, h, text):
        s = self.s = h / 540
        self.text = text
        self.margin = int(12 * s)
        self.gap = int(8 * s)
        self.chip_h = int(54 * s)
        avail = w * 0.76 - self.margin
        self.chip_w = int(min(175 * s, (avail - 3 * self.gap) / 4))
        self.y1 = h - self.margin
        self.y0 = self.y1 - self.chip_h
        self.height = self.chip_h + self.margin  # zajmowana wysokość (dla choinki)
        pad = self.chip_w - 18 * s
        # Rozmiary czcionek dobrane raz, tak by najdłuższy napis zmieścił się w kafelku
        self.label_size = min(text.fit_size(l, pad, int(11 * s), bold=True) for _, l, _ in self.CHIPS)
        self.effect_size = min(text.fit_size(e, pad, int(21 * s), bold=True) for _, _, e in self.CHIPS)

    def draw(self, frame, active, cooldown_frac: dict, info_line: str):
        s, x0, y0, y1 = self.s, self.margin, self.y0, self.y1
        x_end = x0 + 4 * self.chip_w + 3 * self.gap
        # Półprzezroczyste tło kafelków – mieszane tylko w pasie pod nimi
        roi = frame[y0:y1 + 1, x0:x_end + 1]
        layer = roi.copy()
        r = int(14 * s)
        for i, (g, _, _) in enumerate(self.CHIPS):
            cx0 = i * (self.chip_w + self.gap)
            fill = (40, 35, 185) if g in active else (30, 24, 20)
            rounded_rect(layer, (cx0, 0), (cx0 + self.chip_w, self.chip_h), r, fill)
        cv2.addWeighted(layer, 0.72, roi, 0.28, 0, dst=roi)

        for i, (g, label, effect) in enumerate(self.CHIPS):
            cx0 = x0 + i * (self.chip_w + self.gap)
            cx = cx0 + self.chip_w / 2
            on = g in active
            rounded_rect(frame, (cx0, y0), (cx0 + self.chip_w, y1), r,
                         GOLD if on else (90, 85, 80), max(1, int(1.5 * s)))
            self.text.draw(frame, label, (cx, y0 + 15 * s), self.label_size,
                           (200, 225, 255) if on else (175, 170, 165), anchor="mm", bold=True)
            self.text.draw(frame, effect, (cx, y0 + 35 * s), self.effect_size, WHITE,
                           anchor="mm", bold=True)
            frac = cooldown_frac.get(g, 0.0)
            if frac > 0:  # pasek ładowania (cooldown) przy dolnej krawędzi kafelka
                bx0, bx1 = int(cx0 + r), int(cx0 + self.chip_w - r)
                by = int(y1 - 5 * s)
                cv2.line(frame, (bx0, by), (bx1, by), (70, 65, 60), max(2, int(3 * s)), cv2.LINE_AA)
                cv2.line(frame, (bx0, by), (int(bx0 + (bx1 - bx0) * (1 - frac)), by), GOLD,
                         max(2, int(3 * s)), cv2.LINE_AA)

        # Linia informacyjna na półprzezroczystej "pigułce" (czytelna na każdym tle)
        spr = self.text.sprite(info_line, int(13 * s), (235, 235, 235))
        px0, py1 = x0, int(y0 - 6 * s)
        px1, py0 = int(px0 + spr.w + 16 * s), int(py1 - spr.h - 4 * s)
        if py0 >= 0 and px1 < frame.shape[1]:
            roi = frame[py0:py1 + 1, px0:px1 + 1]
            layer = roi.copy()
            rounded_rect(layer, (0, 0), (px1 - px0, py1 - py0), (py1 - py0) // 2, (25, 20, 18))
            cv2.addWeighted(layer, 0.55, roi, 0.45, 0, dst=roi)
            spr.draw(frame, px0 + 8 * s, py0 + 2 * s)
