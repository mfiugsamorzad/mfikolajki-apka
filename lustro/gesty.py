# -*- coding: utf-8 -*-
"""
Rozpoznawanie gestów dłoni z 21 landmarków MediaPipe Hands.

Indeksy landmarków (przydatne przy czytaniu kodu):

        0      – nadgarstek (WRIST)
        1–4    – kciuk:       CMC, MCP, IP, TIP (opuszek)
        5–8    – wskazujący:  MCP, PIP, DIP, TIP
        9–12   – środkowy:    MCP, PIP, DIP, TIP
        13–16  – serdeczny:   MCP, PIP, DIP, TIP
        17–20  – mały:        MCP, PIP, DIP, TIP

    MCP = staw u nasady palca (knykieć), PIP = środkowy staw palca,
    DIP = staw przy opuszku, TIP = opuszek.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Nazwy gestów
FIST = "FIST"
WAVE = "WAVE"
PEACE = "PEACE"
THUMBS_UP = "THUMBS_UP"
NONE = "NONE"

# Stabilizacja gestów: gest uznajemy za aktywny, jeżeli wystąpił w co najmniej
# STABLE_RATIO ostatnich STABLE_WINDOW klatek. Eliminuje to "migotanie"
# klasyfikacji, gdy dłoń jest w pozycji przejściowej.
STABLE_WINDOW = 6
STABLE_RATIO = 0.6


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
