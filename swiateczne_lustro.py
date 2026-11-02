#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Świąteczne Lustro AR
====================

Interaktywne "lustro" działające na obrazie z kamery internetowej. MediaPipe Hands
wyznacza 21 punktów charakterystycznych (landmarków) każdej dłoni, a na podstawie
ich geometrii rozpoznajemy 4 gesty, które uruchamiają efekty świąteczne:

    Gest                    Efekt
    ----------------------  -----------------------------------------------
    Zaciśnięta pięść  (FIST)    padający śnieg
    Otwarta dłoń/machanie (WAVE) przelatujące sanie Mikołaja (santa.png)
    Znak V            (PEACE)   czapka Mikołaja na twarzy / nad palcami (hat.png)
    Kciuk w górę      (THUMBS_UP) migająca choinka w rogu ekranu (tree.png)

Sterowanie z klawiatury:
    q / ESC – wyjście
    d       – włącz/wyłącz rysowanie szkieletu dłoni (tryb debug)
    f       – włącz/wyłącz detekcję twarzy (Haar Cascade)
    s       – zapisz zrzut ekranu

Uruchomienie:
    python swiateczne_lustro.py [--camera 0] [--width 1280 --height 720]

Wymagania: Python 3.9–3.12, opencv-python, numpy, mediapipe<=0.10.21
(nowsze wydania MediaPipe usunęły API ``mediapipe.solutions``).

Indeksy landmarków MediaPipe Hands (przydatne przy czytaniu kodu):

        0  – nadgarstek (WRIST)
        1–4   – kciuk:       CMC, MCP, IP, TIP (opuszek)
        5–8   – wskazujący:  MCP, PIP, DIP, TIP
        9–12  – środkowy:    MCP, PIP, DIP, TIP
        13–16 – serdeczny:   MCP, PIP, DIP, TIP
        17–20 – mały:        MCP, PIP, DIP, TIP

    MCP = staw u nasady palca (knykieć), PIP = środkowy staw palca,
    DIP = staw przy opuszku, TIP = opuszek.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
except (ImportError, AttributeError):  # pragma: no cover - zależy od instalacji
    sys.exit(
        "BŁĄD: nie znaleziono 'mediapipe.solutions'.\n"
        "Zainstaluj zgodną wersję: pip install \"mediapipe<=0.10.21\" "
        "(wymaga Pythona 3.9–3.12)."
    )


# =============================================================================
# KONFIGURACJA
# =============================================================================

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
SANTA_PATH = os.path.join(ASSETS_DIR, "santa.png")
HAT_PATH = os.path.join(ASSETS_DIR, "hat.png")
TREE_PATH = os.path.join(ASSETS_DIR, "tree.png")

WINDOW_NAME = "Swiateczne Lustro AR  (q - wyjscie)"

# Nazwy gestów
FIST = "FIST"
WAVE = "WAVE"
PEACE = "PEACE"
THUMBS_UP = "THUMBS_UP"
NONE = "NONE"

# Cooldown (s) – minimalny odstęp między kolejnymi WYZWOLENIAMI danego efektu.
# Dla sań to kluczowe: bez tego przy 30 FPS każdy kadr z otwartą dłonią
# wysyłałby nowe sanie i animacje nakładałyby się na siebie.
COOLDOWNS = {
    WAVE: 3.0,       # sanie Mikołaja
    THUMBS_UP: 2.0,  # (ponowne) pokazanie choinki
}

# Stabilizacja gestów: gest uznajemy za aktywny, jeżeli wystąpił w co najmniej
# STABLE_RATIO ostatnich STABLE_WINDOW klatek. Eliminuje to "migotanie"
# klasyfikacji, gdy dłoń jest w pozycji przejściowej.
STABLE_WINDOW = 6
STABLE_RATIO = 0.6

# Parametry efektów
SNOW_MAX_FLAKES = 450
SNOW_SPAWN_PER_SEC = 160
SANTA_DURATION = 3.2          # czas przelotu sań przez ekran (s)
TREE_SHOW_TIME = 5.0          # jak długo widać choinkę po geście (s)
HAT_HOLD_TIME = 0.4           # czapka "dotrzymuje" chwilę po zniknięciu gestu
FACE_DETECT_EVERY_N = 4       # Haar co N klatek (oszczędność CPU)


# =============================================================================
# NARZĘDZIA GRAFICZNE
# =============================================================================

def load_rgba(path: str) -> np.ndarray | None:
    """Wczytuje PNG razem z kanałem alfa (BGRA). Zwraca None, jeśli pliku brak."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:  # obraz w skali szarości
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:  # brak przezroczystości – dodajemy pełną alfę
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    """Skaluje obraz do zadanej szerokości z zachowaniem proporcji."""
    width = max(4, int(width))
    h, w = img.shape[:2]
    height = max(4, int(round(h * width / w)))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(img, (width, height), interpolation=interp)


def overlay_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int,
                 opacity: float = 1.0) -> None:
    """
    Nakłada obraz BGRA (src) na klatkę BGR (dst) w punkcie (x, y) – lewy górny róg.

    Obsługuje przycinanie do krawędzi ekranu (obraz może częściowo wychodzić
    poza kadr – potrzebne dla sań wlatujących zza krawędzi). Mieszanie alfa
    liczymy wyłącznie na wyciętym fragmencie (ROI), a nie na całej klatce,
    dzięki czemu operacja jest tania.
    """
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x, y = int(x), int(y)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return  # całkowicie poza ekranem

    crop = src[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = dst[y0:y1, x0:x1]
    alpha = crop[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
    blended = crop[:, :, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    roi[:] = blended.astype(np.uint8)


class ScaledCache:
    """
    Pamięć podręczna przeskalowanych wersji grafiki. cv2.resize w każdej klatce
    jest zbędnym kosztem – szerokość kwantyzujemy do 8 px i trzymamy kilkanaście
    ostatnio używanych rozmiarów.
    """

    def __init__(self, img: np.ndarray, max_items: int = 32):
        self.img = img
        self.max_items = max_items
        self._cache: dict[int, np.ndarray] = {}

    def get(self, width: int) -> np.ndarray:
        key = max(8, int(width) // 8 * 8)
        out = self._cache.get(key)
        if out is None:
            if len(self._cache) >= self.max_items:
                self._cache.pop(next(iter(self._cache)))
            out = resize_to_width(self.img, key)
            self._cache[key] = out
        return out


# --- Proceduralne grafiki zastępcze (gdy brak hat.png / tree.png) -----------

def make_fallback_hat() -> np.ndarray:
    """Rysuje prostą czapkę Mikołaja (BGRA) – używana, gdy brak hat.png."""
    img = np.zeros((260, 300, 4), np.uint8)
    red, white = (30, 30, 210, 255), (245, 245, 245, 255)
    # Czerwony "stożek" lekko przechylony w prawo
    cone = np.array([[30, 210], [270, 210], [215, 70], [150, 25], [255, 40]], np.int32)
    cv2.fillPoly(img, [cone], red, cv2.LINE_AA)
    # Biały futrzany otok
    cv2.rectangle(img, (15, 195), (285, 245), white, -1)
    cv2.circle(img, (15, 220), 25, white, -1, cv2.LINE_AA)
    cv2.circle(img, (285, 220), 14, white, -1, cv2.LINE_AA)
    # Pompon
    cv2.circle(img, (258, 42), 26, white, -1, cv2.LINE_AA)
    return img


def make_fallback_tree() -> np.ndarray:
    """Rysuje prostą choinkę (BGRA) – używana, gdy brak tree.png."""
    img = np.zeros((340, 260, 4), np.uint8)
    green, dark = (40, 140, 20, 255), (20, 90, 10, 255)
    cv2.rectangle(img, (110, 285), (150, 335), (20, 60, 110, 255), -1)  # pień
    tiers = [(60, 150, 115), (110, 215, 100), (165, 285, 125)]
    for top, bottom, half in tiers:
        pts = np.array([[130, top], [130 - half, bottom], [130 + half, bottom]], np.int32)
        cv2.fillPoly(img, [pts], green, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, dark, 3, cv2.LINE_AA)
    # Gwiazda na czubku
    star = []
    for i in range(10):
        r = 32 if i % 2 == 0 else 13
        a = -math.pi / 2 + i * math.pi / 5
        star.append([130 + r * math.cos(a), 45 + r * math.sin(a)])
    cv2.fillPoly(img, [np.array(star, np.int32)], (0, 215, 255, 255), cv2.LINE_AA)
    return img


# Względne pozycje lampek na choince (u, v w zakresie 0..1) – używane do
# animowanego migania niezależnie od tego, czy grafika pochodzi z pliku.
TREE_LIGHTS_UV = [
    (0.45, 0.30), (0.58, 0.38), (0.38, 0.45), (0.62, 0.52), (0.30, 0.60),
    (0.50, 0.58), (0.70, 0.66), (0.40, 0.72), (0.22, 0.78), (0.58, 0.80),
    (0.78, 0.79), (0.48, 0.47),
]
LIGHT_COLORS = [(0, 0, 255), (0, 215, 255), (255, 120, 0), (255, 0, 255), (0, 255, 120)]


# =============================================================================
# ROZPOZNAWANIE GESTÓW Z LANDMARKÓW
# =============================================================================

# Trójki (MCP, PIP, TIP) dla czterech "długich" palców
FINGERS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


@dataclass
class HandInfo:
    """Wynik analizy pojedynczej dłoni (współrzędne w pikselach)."""
    gesture: str
    pts: np.ndarray            # (21, 2) – landmarki w pikselach
    palm_center: np.ndarray    # (2,)
    hand_size: float           # skala dłoni w pikselach
    extended: dict = field(default_factory=dict)


def landmarks_to_array(hand_landmarks, w: int, h: int) -> np.ndarray:
    """
    Zamienia znormalizowane landmarki MediaPipe (x, y w zakresie 0..1) na tablicę
    pikseli (21, 2). Pracujemy w pikselach, bo znormalizowane x i y mają różną
    skalę przy kadrze 16:9 / 4:3, co zniekształcałoby odległości euklidesowe.
    """
    return np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark], dtype=np.float32)


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def finger_states(pts: np.ndarray, palm_center: np.ndarray, hand_size: float) -> dict:
    """
    Określa, które palce są wyprostowane.

    Długie palce (wskazujący … mały)
    --------------------------------
    Nie porównujemy po prostu współrzędnych y opuszka i stawu (np. tip.y < pip.y),
    bo taki test działa tylko dla dłoni skierowanej palcami do góry. Zamiast tego
    używamy odległości od NADGARSTKA, które są niezależne od obrotu dłoni:

        * palec wyprostowany  -> opuszek (TIP) leży DALEJ od nadgarstka niż staw
                                 PIP, i to wyraźnie (o > 15 %),
        * palec zgięty        -> opuszek "zawija się" do wnętrza dłoni, więc jest
                                 BLIŻEJ nadgarstka niż PIP.

    Kciuk
    -----
    Kciuk zgina się w innej płaszczyźnie, więc test "dalej od nadgarstka" jest dla
    niego zawodny. Sprawdzamy, jak daleko opuszek kciuka (4) odstaje od boku
    dłoni – od nasady palca wskazującego (5) i od środka dłoni. Kciuk schowany
    (w pięści lub przyciśnięty do dłoni) leży blisko tych punktów; kciuk
    odchylony/wyprostowany jest od nich oddalony o ponad ~0,55–0,7 rozmiaru dłoni.
    Dodatkowo kciuk musi być "rozprostowany": TIP dalej od MCP kciuka (2) niż IP (3).
    """
    wrist = pts[0]
    states = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        d_tip = dist(pts[tip], wrist)
        d_pip = dist(pts[pip], wrist)
        states[name] = d_tip > d_pip * 1.15

    thumb_tip, thumb_ip, thumb_mcp = pts[4], pts[3], pts[2]
    thumb_out = (dist(thumb_tip, pts[5]) > 0.55 * hand_size and
                 dist(thumb_tip, palm_center) > 0.70 * hand_size)
    thumb_straight = dist(thumb_tip, thumb_mcp) > dist(thumb_ip, thumb_mcp) * 1.2
    states["thumb"] = thumb_out and thumb_straight
    return states


def classify_gesture(pts: np.ndarray) -> HandInfo:
    """
    Klasyfikuje gest pojedynczej dłoni na podstawie 21 landmarków (w pikselach).

    Skala dłoni
    -----------
    Wszystkie progi wyrażamy względem "rozmiaru dłoni" = odległość nadgarstek (0)
    -> nasada palca środkowego (9). Ta długość praktycznie nie zmienia się przy
    zginaniu palców, a dzięki normalizacji klasyfikator działa tak samo dla dłoni
    blisko i daleko od kamery.

    Środek dłoni
    ------------
    Średnia z nadgarstka i czterech knykci (0, 5, 9, 13, 17) – punkt stabilny,
    leżący mniej więcej w środku wewnętrznej strony dłoni.
    """
    palm_idx = [0, 5, 9, 13, 17]
    palm_center = pts[palm_idx].mean(axis=0)
    hand_size = max(dist(pts[0], pts[9]), 1e-3)
    ext = finger_states(pts, palm_center, hand_size)

    long_fingers = ["index", "middle", "ring", "pinky"]
    n_long_ext = sum(ext[f] for f in long_fingers)

    # Średnia odległość opuszków 4 długich palców od środka dłoni (znormalizowana).
    # W pięści opuszki dotykają wnętrza dłoni -> wartość ~0.5–0.8.
    # Przy otwartej dłoni opuszki są daleko -> wartość ~1.3–1.9.
    tips_to_palm = np.mean([dist(pts[FINGERS[f][2]], palm_center) for f in long_fingers]) / hand_size

    # --- KCIUK W GÓRĘ ------------------------------------------------------
    # Cztery palce zgięte, kciuk wyprostowany ORAZ skierowany do góry:
    #   * opuszek kciuka leży wyraźnie wyżej (mniejsze y w obrazie) niż jego
    #     staw MCP – o co najmniej 0,5 rozmiaru dłoni,
    #   * opuszek kciuka jest najwyżej położonym punktem całej dłoni,
    #   * wektor MCP->TIP kciuka jest bliski pionowi (|dx| < dy).
    # Ten test wykonujemy PRZED pięścią, bo kciuk w górę to geometrycznie
    # "pięść z odstającym kciukiem".
    thumb_vec = pts[4] - pts[2]
    thumb_up = (
        n_long_ext == 0
        and ext["thumb"]
        and (pts[2][1] - pts[4][1]) > 0.5 * hand_size
        and pts[4][1] <= pts[:, 1].min() + 1.0
        and abs(thumb_vec[0]) < -thumb_vec[1]
    )
    if thumb_up:
        return HandInfo(THUMBS_UP, pts, palm_center, hand_size, ext)

    # --- PIĘŚĆ -------------------------------------------------------------
    # Wszystkie 4 długie palce zgięte, a ich opuszki blisko środka dłoni
    # (średnia odległość < 0,95 rozmiaru dłoni). Kciuk może być przyciśnięty
    # z boku lub na palcach – nie sprawdzamy go, poza wykluczeniem kciuka w górę
    # (obsłużone wyżej).
    if n_long_ext == 0 and tips_to_palm < 0.95:
        return HandInfo(FIST, pts, palm_center, hand_size, ext)

    # --- ZNAK V (PEACE) ----------------------------------------------------
    # Wskazujący i środkowy wyprostowane, serdeczny i mały zgięte. Dodatkowo
    # palce muszą być rozchylone w literę "V": odległość między ich opuszkami
    # (8 i 12) jest większa niż odległość między ich nasadami (5 i 9) –
    # inaczej byłyby to dwa złączone palce ("pistolet"/"przysięga").
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        spread_tips = dist(pts[8], pts[12])
        spread_base = dist(pts[5], pts[9])
        if spread_tips > 1.3 * spread_base:
            return HandInfo(PEACE, pts, palm_center, hand_size, ext)

    # --- OTWARTA DŁOŃ (WAVE) -----------------------------------------------
    # Wszystkie 4 długie palce wyprostowane, opuszki daleko od środka dłoni
    # (> 1,1 rozmiaru dłoni – wyraźny margines względem progu pięści 0,95).
    # Kciuk dopuszczamy w dowolnym stanie (przy machaniu często jest przy dłoni).
    # Samo machanie (ruch na boki) dodatkowo wykrywa WaveTracker – patrz niżej.
    if n_long_ext == 4 and tips_to_palm > 1.1:
        return HandInfo(WAVE, pts, palm_center, hand_size, ext)

    return HandInfo(NONE, pts, palm_center, hand_size, ext)


class WaveTracker:
    """
    Wykrywa MACHANIE otwartą dłonią – oscylację w poziomie.

    Zapamiętujemy pozycję x środka dłoni z ostatnich ~0,8 s. Liczymy zmiany
    kierunku ruchu (znak prędkości), ignorując drobne drgania (ruch mniejszy niż
    0,15 rozmiaru dłoni). Dwie lub więcej zmian kierunku = machanie.
    Machanie wyzwala sanie natychmiast, a sama nieruchoma otwarta dłoń – po
    przytrzymaniu przez OPEN_HOLD sekund (żeby przypadkowe rozłożenie palców nie
    wysyłało sań).
    """

    WINDOW_S = 0.8
    OPEN_HOLD = 0.6

    def __init__(self):
        self.history: deque = deque()   # (czas, x, rozmiar_dłoni)
        self.open_since: float | None = None

    def update(self, now: float, open_hand: HandInfo | None) -> bool:
        if open_hand is None:
            self.history.clear()
            self.open_since = None
            return False

        if self.open_since is None:
            self.open_since = now
        self.history.append((now, float(open_hand.palm_center[0]), open_hand.hand_size))
        while self.history and now - self.history[0][0] > self.WINDOW_S:
            self.history.popleft()

        # Zliczanie zmian kierunku z histerezą
        reversals, direction, anchor = 0, 0, None
        for _, x, size in self.history:
            if anchor is None:
                anchor = x
                continue
            delta = x - anchor
            if abs(delta) > 0.15 * size:
                new_dir = 1 if delta > 0 else -1
                if direction != 0 and new_dir != direction:
                    reversals += 1
                direction = new_dir
                anchor = x
        waving = reversals >= 2
        held = now - self.open_since >= self.OPEN_HOLD
        return waving or held


class GestureStabilizer:
    """
    Głosowanie większościowe w przesuwnym oknie klatek. Gest jest "aktywny",
    gdy pojawił się w >= STABLE_RATIO klatek z ostatnich STABLE_WINDOW. Dzięki
    temu pojedyncze błędne klasyfikacje nie uruchamiają ani nie gaszą efektów.
    """

    def __init__(self, window: int = STABLE_WINDOW, ratio: float = STABLE_RATIO):
        self.frames: deque = deque(maxlen=window)
        self.ratio = ratio

    def update(self, gestures: set) -> set:
        self.frames.append(gestures)
        n = len(self.frames)
        counts: dict = {}
        for s in self.frames:
            for g in s:
                counts[g] = counts.get(g, 0) + 1
        return {g for g, c in counts.items() if c / n >= self.ratio and g != NONE}


class Cooldown:
    """Prosta blokada czasowa (debouncing) wyzwalania efektów."""

    def __init__(self, durations: dict):
        self.durations = durations
        self.last: dict = {}

    def ready(self, key: str, now: float) -> bool:
        return now - self.last.get(key, -1e9) >= self.durations.get(key, 0.0)

    def fire(self, key: str, now: float) -> None:
        self.last[key] = now

    def remaining(self, key: str, now: float) -> float:
        return max(0.0, self.durations.get(key, 0.0) - (now - self.last.get(key, -1e9)))


# =============================================================================
# EFEKTY
# =============================================================================

class SnowEffect:
    """
    Śnieg jako system cząsteczek trzymany w tablicach NumPy (wektorowa
    aktualizacja pozycji – brak pętli Pythona przy fizyce). Płatki dodajemy,
    dopóki trwa gest pięści; po jego zakończeniu istniejące płatki spadają do
    końca, więc efekt gaśnie łagodnie.
    """

    def __init__(self):
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0,), np.float32)
        self.radius = np.zeros((0,), np.int32)
        self.phase = np.zeros((0,), np.float32)
        self._spawn_acc = 0.0

    def update(self, dt: float, active: bool, w: int, h: int, t: float) -> None:
        if active:
            self._spawn_acc += SNOW_SPAWN_PER_SEC * dt
            n_new = min(int(self._spawn_acc), SNOW_MAX_FLAKES - len(self.pos))
            self._spawn_acc -= int(self._spawn_acc)
            if n_new > 0:
                radius = np.random.randint(2, 6, n_new).astype(np.int32)
                new_pos = np.column_stack([
                    np.random.uniform(0, w, n_new),
                    np.random.uniform(-40, -5, n_new),
                ]).astype(np.float32)
                # Większe płatki spadają szybciej – prosta iluzja głębi
                new_vel = (np.random.uniform(50, 90, n_new) + radius * 22).astype(np.float32)
                self.pos = np.vstack([self.pos, new_pos])
                self.vel = np.concatenate([self.vel, new_vel])
                self.radius = np.concatenate([self.radius, radius])
                self.phase = np.concatenate([self.phase, np.random.uniform(0, 6.28, n_new).astype(np.float32)])

        if len(self.pos) == 0:
            return
        # Opadanie + łagodne kołysanie na boki (sinusoida z losową fazą)
        self.pos[:, 1] += self.vel * dt
        self.pos[:, 0] += np.sin(t * 1.8 + self.phase) * 35.0 * dt
        alive = self.pos[:, 1] < h + 10
        self.pos, self.vel = self.pos[alive], self.vel[alive]
        self.radius, self.phase = self.radius[alive], self.phase[alive]

    def draw(self, frame: np.ndarray) -> None:
        for (x, y), r in zip(self.pos.astype(np.int32), self.radius):
            cv2.circle(frame, (int(x), int(y)), int(r), (255, 255, 255), -1, cv2.LINE_AA)

    @property
    def visible(self) -> bool:
        return len(self.pos) > 0


class SantaEffect:
    """
    Sanie Mikołaja przelatujące z lewej do prawej po łuku z lekkim kołysaniem.
    Jeśli santa.png nie istnieje – rysujemy czerwony prostokąt z napisem
    "HO HO HO" (fallback zgodny ze specyfikacją).
    """

    def __init__(self, image: np.ndarray | None):
        self.image = image
        self.scaled: np.ndarray | None = None
        self.start_t: float | None = None
        self.frame_w = 0

    def trigger(self, now: float) -> None:
        self.start_t = now

    @property
    def active(self) -> bool:
        return self.start_t is not None

    def draw(self, frame: np.ndarray, now: float) -> None:
        if self.start_t is None:
            return
        h, w = frame.shape[:2]
        progress = (now - self.start_t) / SANTA_DURATION
        if progress >= 1.0:
            self.start_t = None
            return

        if self.image is not None:
            if self.scaled is None or self.frame_w != w:
                self.scaled = resize_to_width(self.image, w * 0.38)
                self.frame_w = w
            sw, sh = self.scaled.shape[1], self.scaled.shape[0]
        else:
            sw, sh = int(w * 0.30), int(h * 0.12)

        # Tor lotu: od -sw do w (całe sanie wlatują i wylatują), wysokość po
        # paraboli (najwyżej w połowie) + drobne "podskoki" sinusoidalne.
        x = int(-sw + progress * (w + sw))
        y = int(h * 0.22 - math.sin(progress * math.pi) * h * 0.10
                + math.sin(progress * 6 * math.pi) * h * 0.015)

        if self.image is not None:
            overlay_rgba(frame, self.scaled, x, y)
        else:
            cv2.rectangle(frame, (x, y), (x + sw, y + sh), (0, 0, 200), -1)
            cv2.rectangle(frame, (x, y), (x + sw, y + sh), (255, 255, 255), 3)
            font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, sh / 45.0, max(2, sh // 25)
            (tw, th), _ = cv2.getTextSize("HO HO HO", font, scale, thick)
            cv2.putText(frame, "HO HO HO", (x + (sw - tw) // 2, y + (sh + th) // 2),
                        font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        # Smuga brokatu za saniami
        for i in range(12):
            px = x - 10 - i * 14 + np.random.randint(-4, 5)
            py = y + sh // 2 + np.random.randint(-sh // 3, sh // 3 + 1)
            cv2.circle(frame, (px, py), max(1, 4 - i // 4), (0, 215, 255), -1, cv2.LINE_AA)


class HatEffect:
    """
    Czapka Mikołaja nakładana na twarz (Haar Cascade) lub – gdy twarzy nie ma –
    nad palcami pokazującymi znak V.
    """

    def __init__(self, image: np.ndarray, face_cascade: cv2.CascadeClassifier | None):
        self.cache = ScaledCache(image)
        self.aspect = image.shape[0] / image.shape[1]
        self.face_cascade = face_cascade
        self.faces: list = []
        self.frame_idx = 0
        self.visible_until = 0.0
        self.anchor_hand: HandInfo | None = None
        # Wygładzanie pozycji (filtr wykładniczy) – czapka nie "drga"
        self.smooth: np.ndarray | None = None

    def detect_faces(self, gray_small: np.ndarray, scale: float) -> None:
        """
        Haar Cascade na pomniejszonym obrazie w skali szarości – to kilkukrotnie
        szybsze niż na pełnej rozdzielczości, a dla twarzy w lustrze (duże
        obiekty) dokładność jest wystarczająca.
        """
        if self.face_cascade is None:
            self.faces = []
            return
        found = self.face_cascade.detectMultiScale(gray_small, scaleFactor=1.15,
                                                   minNeighbors=5, minSize=(40, 40))
        self.faces = [tuple(int(v * scale) for v in f) for f in found]

    def update(self, frame: np.ndarray, now: float, peace_active: bool,
               peace_hand: HandInfo | None, use_face: bool) -> None:
        if peace_active:
            self.visible_until = now + HAT_HOLD_TIME
            if peace_hand is not None:
                self.anchor_hand = peace_hand
        if now > self.visible_until:
            self.smooth = None
            self.faces = []
            return

        # Detekcja twarzy tylko gdy czapka jest widoczna i tylko co N klatek
        self.frame_idx += 1
        if use_face and self.frame_idx % FACE_DETECT_EVERY_N == 1:
            small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            self.detect_faces(gray, 2.0)
        elif not use_face:
            self.faces = []

    def draw(self, frame: np.ndarray, now: float) -> None:
        if now > self.visible_until:
            return
        if self.faces:
            # Wszystkie twarze dostają czapkę (świąteczne lustro dla całej grupy!)
            for (fx, fy, fw, fh) in self.faces:
                hat_w = int(fw * 1.45)
                hat = self.cache.get(hat_w)
                hx = fx + fw // 2 - hat.shape[1] // 2
                # Dół czapki (otok) nachodzi na górne ~25% prostokąta twarzy (czoło)
                hy = fy + int(fh * 0.25) - hat.shape[0]
                overlay_rgba(frame, hat, hx, hy)
            return

        hand = self.anchor_hand
        if hand is None:
            return
        # Brak twarzy -> czapka "siedzi" na czubkach palców wskazującego i
        # środkowego; szerokość proporcjonalna do rozmiaru dłoni.
        tips = hand.pts[[8, 12]]
        target = np.array([tips[:, 0].mean(), tips[:, 1].min(), hand.hand_size * 1.9], np.float32)
        self.smooth = target if self.smooth is None else self.smooth * 0.6 + target * 0.4
        cx, top, width = self.smooth
        hat = self.cache.get(width)
        overlay_rgba(frame, hat, int(cx - hat.shape[1] / 2), int(top - hat.shape[0] * 0.85))


class TreeEffect:
    """Choinka w prawym dolnym rogu z migającymi lampkami i łagodnym pojawieniem."""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.scaled: np.ndarray | None = None
        self.frame_h = 0
        self.show_since = -1e9
        self.show_until = 0.0

    def trigger(self, now: float) -> None:
        if now > self.show_until:
            self.show_since = now
        self.show_until = now + TREE_SHOW_TIME

    def draw(self, frame: np.ndarray, now: float) -> None:
        if now > self.show_until:
            return
        h, w = frame.shape[:2]
        if self.scaled is None or self.frame_h != h:
            self.scaled = resize_to_width(self.image, int(h * 0.42 * self.image.shape[1] / self.image.shape[0]))
            self.frame_h = h
        th, tw = self.scaled.shape[:2]
        x, y = w - tw - 15, h - th - 10

        # Pojawianie się (fade-in 0,3 s) i znikanie (fade-out 0,5 s)
        opacity = min(1.0, (now - self.show_since) / 0.3, (self.show_until - now) / 0.5)
        overlay_rgba(frame, self.scaled, x, y, opacity=max(0.0, opacity))

        # Migające lampki: każda lampka ma własną fazę -> efekt "chodzącego światła"
        step = int(now * 4)
        for i, (u, v) in enumerate(TREE_LIGHTS_UV):
            if (i + step) % 3 == 0:
                continue  # ta lampka chwilowo zgaszona
            color = LIGHT_COLORS[(i + step) % len(LIGHT_COLORS)]
            cx, cy = int(x + u * tw), int(y + v * th)
            r = max(3, tw // 40)
            cv2.circle(frame, (cx, cy), r + 3, color, 1, cv2.LINE_AA)  # poświata
            cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)


# =============================================================================
# INTERFEJS (HUD)
# =============================================================================

GESTURE_LABELS = {
    FIST: "Piesc -> snieg",
    WAVE: "Otwarta dlon -> sanie",
    PEACE: "Znak V -> czapka",
    THUMBS_UP: "Kciuk w gore -> choinka",
}


def draw_hud(frame: np.ndarray, fps: float, active: set, cooldown: Cooldown, now: float,
             debug: bool, face_on: bool) -> None:
    """Pasek informacyjny: FPS, aktywne gesty i stan cooldownów."""
    h, w = frame.shape[:2]
    panel = frame[0:34, :]
    panel[:] = (panel * 0.45).astype(np.uint8)  # półprzezroczyste tło
    txt = ", ".join(GESTURE_LABELS[g] for g in sorted(active)) or "pokaz gest dloni"
    cv2.putText(frame, f"FPS {fps:4.1f}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, txt, (120, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    flags = f"[d]ebug:{'on' if debug else 'off'}  [f]ace:{'on' if face_on else 'off'}"
    cv2.putText(frame, flags, (w - 260, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Pasek cooldownu sań – widać, kiedy można wysłać kolejne
    rem = cooldown.remaining(WAVE, now)
    if rem > 0:
        frac = rem / COOLDOWNS[WAVE]
        cv2.rectangle(frame, (10, 42), (10 + int(150 * frac), 50), (0, 0, 255), -1)
        cv2.rectangle(frame, (10, 42), (160, 50), (255, 255, 255), 1)
        cv2.putText(frame, "sanie: cooldown", (168, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_debug_hand(frame: np.ndarray, hand_landmarks, info: HandInfo) -> None:
    """Szkielet dłoni + etykieta gestu + środek dłoni i okrąg skali (debug)."""
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                           mp_styles.get_default_hand_landmarks_style(),
                           mp_styles.get_default_hand_connections_style())
    c = tuple(int(v) for v in info.palm_center)
    cv2.circle(frame, c, 5, (0, 255, 255), -1)
    cv2.circle(frame, c, int(info.hand_size * 0.95), (0, 255, 255), 1)  # próg pięści
    fingers = "".join("1" if info.extended.get(f) else "0"
                      for f in ["thumb", "index", "middle", "ring", "pinky"])
    x, y = int(info.pts[:, 0].min()), int(info.pts[:, 1].min()) - 10
    cv2.putText(frame, f"{info.gesture} [{fingers}]", (x, max(50, y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


# =============================================================================
# PĘTLA GŁÓWNA
# =============================================================================

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"BŁĄD: nie można otworzyć kamery o indeksie {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimalne opóźnienie (gdy backend wspiera)
    return cap


def load_assets():
    santa = load_rgba(SANTA_PATH)
    hat = load_rgba(HAT_PATH)
    tree = load_rgba(TREE_PATH)
    if santa is None:
        print("[info] Brak santa.png – sanie zastąpi czerwony prostokąt 'HO HO HO'.")
    if hat is None:
        print("[info] Brak hat.png – używam rysowanej czapki zastępczej.")
        hat = make_fallback_hat()
    if tree is None:
        print("[info] Brak tree.png – używam rysowanej choinki zastępczej.")
        tree = make_fallback_tree()

    face_cascade = None
    try:
        # Najpierw plik obok skryptu (gdyby instalacja OpenCV go nie zawierała),
        # potem kaskada dołączona do pakietu opencv-python.
        candidates = [os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")]
        if hasattr(cv2, "data"):
            candidates.append(os.path.join(cv2.data.haarcascades,
                                           "haarcascade_frontalface_default.xml"))
        for path in candidates:
            if os.path.isfile(path):
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    break
                face_cascade = None
    except (AttributeError, cv2.error):
        face_cascade = None
    if face_cascade is None:
        print("[info] Haar Cascade niedostępny – czapka będzie nakładana nad palcami.")
    return santa, hat, tree, face_cascade


def main() -> None:
    parser = argparse.ArgumentParser(description="Świąteczne lustro AR (OpenCV + MediaPipe)")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamery (domyślnie 0)")
    parser.add_argument("--width", type=int, default=960, help="szerokość obrazu kamery")
    parser.add_argument("--height", type=int, default=540, help="wysokość obrazu kamery")
    parser.add_argument("--fullscreen", action="store_true", help="tryb pełnoekranowy")
    parser.add_argument("--no-face", action="store_true", help="wyłącz detekcję twarzy")
    args = parser.parse_args()

    santa_img, hat_img, tree_img, face_cascade = load_assets()
    snow = SnowEffect()
    santa = SantaEffect(santa_img)
    hat = HatEffect(hat_img, face_cascade)
    tree = TreeEffect(tree_img)

    wave_tracker = WaveTracker()
    stabilizer = GestureStabilizer()
    cooldown = Cooldown(COOLDOWNS)

    cap = open_camera(args.camera, args.width, args.height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    debug = False
    use_face = not args.no_face and face_cascade is not None
    fps = 0.0
    prev_t = time.perf_counter()

    # model_complexity=0 – lżejszy model dłoni, wyraźnie szybszy na CPU,
    # w zupełności wystarczający do rozpoznawania gestów.
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] Nie udało się odczytać klatki z kamery.")
                break

            now = time.perf_counter()
            dt = min(now - prev_t, 0.1)  # ograniczenie dt po "zawieszeniu" okna
            prev_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)

            # Efekt lustra – obraz odwrócony w poziomie, jak w prawdziwym lustrze
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # --- 1. Detekcja dłoni (MediaPipe oczekuje RGB) ----------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False  # pozwala MediaPipe uniknąć kopiowania
            results = hands.process(rgb)

            # --- 2. Klasyfikacja gestów każdej dłoni -----------------------
            hand_infos: list[HandInfo] = []
            raw_landmarks = results.multi_hand_landmarks or []
            for lm in raw_landmarks:
                hand_infos.append(classify_gesture(landmarks_to_array(lm, w, h)))

            frame_gestures = {hi.gesture for hi in hand_infos}
            active = stabilizer.update(frame_gestures)

            # --- 3. Logika wyzwalania efektów (z cooldownem) ---------------
            open_hand = next((hi for hi in hand_infos if hi.gesture == WAVE), None)
            wave_ready = wave_tracker.update(now, open_hand if WAVE in active else None)
            if wave_ready and not santa.active and cooldown.ready(WAVE, now):
                santa.trigger(now)
                cooldown.fire(WAVE, now)

            if THUMBS_UP in active and cooldown.ready(THUMBS_UP, now):
                tree.trigger(now)
                cooldown.fire(THUMBS_UP, now)

            peace_hand = next((hi for hi in hand_infos if hi.gesture == PEACE), None)

            # --- 4. Aktualizacja efektów ------------------------------------
            snow.update(dt, FIST in active, w, h, now)
            hat.update(frame, now, PEACE in active, peace_hand, use_face)

            # --- 5. Rysowanie (kolejność = warstwy) ------------------------
            if debug:
                for lm, info in zip(raw_landmarks, hand_infos):
                    draw_debug_hand(frame, lm, info)
            hat.draw(frame, now)
            tree.draw(frame, now)
            santa.draw(frame, now)
            snow.draw(frame)  # śnieg na samej górze – pada "przed" wszystkim
            draw_hud(frame, fps, active, cooldown, now, debug, use_face)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            elif key == ord("f") and face_cascade is not None:
                use_face = not use_face
            elif key == ord("s"):
                name = time.strftime("lustro_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, frame)
                print(f"[info] Zapisano {name}")
            # Zamknięcie okna krzyżykiem
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
