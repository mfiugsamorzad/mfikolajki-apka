#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Świąteczne Lustro AR
====================

Interaktywne "lustro" działające na obrazie z kamery internetowej. MediaPipe Hands
wyznacza 21 punktów charakterystycznych (landmarków) każdej dłoni, a na podstawie
ich geometrii rozpoznajemy 4 gesty, które uruchamiają efekty świąteczne:

    Gest                          Efekt
    ----------------------------  ---------------------------------------------
    Zaciśnięta pięść  (FIST)      padający śnieg (snowflake.png) + zaspa
    Otwarta dłoń / machanie (WAVE) przelatujące sanie Mikołaja (santa.png)
    Znak V            (PEACE)     czapka Mikołaja na twarzy / nad palcami (hat.png)
    Kciuk w górę      (THUMBS_UP) choinka z mrugającymi lampkami (tree.png)

Struktura projektu:
    swiateczne_lustro.py – pętla główna: kamera, MediaPipe, logika wyzwalania
    gesty.py             – rozpoznawanie gestów z landmarków (+ stabilizacja, cooldown)
    efekty.py            – grafika: sprite'y, śnieg, sanie, czapka, choinka, HUD

Sterowanie z klawiatury:
    q / ESC – wyjście            d – szkielet dłoni i rozpoznany gest (debug)
    f       – detekcja twarzy    g – dekoracje (girlanda + winieta)
    s       – zrzut ekranu

Wymagania: Python 3.9–3.12, opencv-python, numpy, mediapipe<=0.10.21
(nowsze wydania MediaPipe usunęły API ``mediapipe.solutions``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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

import efekty as fx
from gesty import (FIST, PEACE, THUMBS_UP, WAVE, Cooldown, GestureStabilizer, HandInfo,
                   WaveTracker, classify_gesture, landmarks_to_array)

# =============================================================================
# KONFIGURACJA
# =============================================================================

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_FILES = {
    "santa": "santa.png",
    "hat": "hat.png",
    "tree": "tree.png",
    "snowflake": "snowflake.png",
}

WINDOW_NAME = "Swiateczne Lustro AR"

# Cooldown (s) – minimalny odstęp między kolejnymi WYZWOLENIAMI danego efektu.
# Dla sań to kluczowe: bez tego przy 30 FPS każdy kadr z otwartą dłonią
# wysyłałby nowe sanie i animacje nakładałyby się na siebie.
COOLDOWNS = {
    WAVE: 3.0,       # sanie Mikołaja
    THUMBS_UP: 2.0,  # (ponowne) pokazanie choinki
}

SANTA_DURATION = 3.4          # czas przelotu sań przez ekran (s)
TREE_SHOW_TIME = 5.0          # jak długo widać choinkę po geście (s)
HAT_HOLD_TIME = 0.45          # czapka "dotrzymuje" (i gaśnie) chwilę po zniknięciu gestu
FACE_DETECT_EVERY_N = 4       # Haar co N klatek (oszczędność CPU)


# =============================================================================
# ZASOBY
# =============================================================================

def load_rgba(path: str) -> np.ndarray | None:
    """Wczytuje PNG razem z kanałem alfa (BGRA). Zwraca None, jeśli pliku brak."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype != np.uint8:  # np. 16-bitowe PNG
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        print(f"[uwaga] {os.path.basename(path)} nie ma przezroczystości – "
              "będzie widoczny jako prostokąt. Użyj PNG z przezroczystym tłem.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def load_face_cascade():
    """Haar Cascade do detekcji twarzy (plik obok skryptu lub z pakietu OpenCV)."""
    name = "haarcascade_frontalface_default.xml"
    candidates = [os.path.join(ASSETS_DIR, name)]
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades, name))
    for path in candidates:
        if os.path.isfile(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    print("[info] Haar Cascade niedostępny – czapka będzie nakładana nad palcami.")
    return None


def load_assets() -> dict:
    assets = {k: load_rgba(os.path.join(ASSETS_DIR, f)) for k, f in ASSET_FILES.items()}
    if assets["santa"] is None:
        print("[info] Brak santa.png – sanie zastąpi czerwony baner 'HO HO HO'.")
    if assets["hat"] is None:
        print("[info] Brak hat.png – używam rysowanej czapki.")
        assets["hat"] = fx.make_fallback_hat()
    if assets["tree"] is None:
        print("[info] Brak tree.png – używam rysowanej choinki.")
        assets["tree"] = fx.make_fallback_tree()
    if assets["snowflake"] is None:
        print("[info] Brak snowflake.png – używam rysowanej śnieżynki.")
        assets["snowflake"] = fx.make_fallback_snowflake()
    assets["face_cascade"] = load_face_cascade()
    return assets


# =============================================================================
# SCENA – wszystkie efekty zależne od rozmiaru klatki
# =============================================================================

class Scene:
    """Tworzona po odczytaniu pierwszej klatki (efekty skalują się do jej rozmiaru)."""

    def __init__(self, w: int, h: int, assets: dict):
        self.w, self.h = w, h
        self.hud = fx.Hud(w, h)
        self.sparkles = fx.Sparkles(h / 540)
        self.snow = fx.SnowEffect(w, h, assets["snowflake"])
        self.santa = fx.SantaEffect(w, h, assets["santa"], SANTA_DURATION)
        self.hat = fx.HatEffect(assets["hat"], assets["face_cascade"], HAT_HOLD_TIME,
                                FACE_DETECT_EVERY_N)
        self.tree = fx.TreeEffect(w, h, assets["tree"], TREE_SHOW_TIME, self.hud.height + 6)
        self.garland = fx.Garland(w, h)
        self.vignette = fx.make_vignette(w, h)


def draw_debug_hand(frame: np.ndarray, hand_landmarks, info: HandInfo) -> None:
    """Szkielet dłoni + etykieta gestu + środek dłoni i okrąg progu pięści (debug)."""
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                           mp_styles.get_default_hand_landmarks_style(),
                           mp_styles.get_default_hand_connections_style())
    c = tuple(int(v) for v in info.palm_center)
    cv2.circle(frame, c, 5, (0, 255, 255), -1)
    cv2.circle(frame, c, int(info.hand_size * 0.95), (0, 255, 255), 1)
    fingers = "".join("1" if info.extended.get(f) else "0"
                      for f in ["thumb", "index", "middle", "ring", "pinky"])
    x, y = int(info.pts[:, 0].min()), int(info.pts[:, 1].min()) - 10
    fx.text_outlined(frame, f"{info.gesture} [{fingers}]", (x, max(60, y)), 0.6, (0, 255, 255), 2)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Świąteczne lustro AR (OpenCV + MediaPipe)")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamery (domyślnie 0)")
    parser.add_argument("--width", type=int, default=960, help="szerokość obrazu kamery")
    parser.add_argument("--height", type=int, default=540, help="wysokość obrazu kamery")
    parser.add_argument("--fullscreen", action="store_true", help="tryb pełnoekranowy")
    parser.add_argument("--no-face", action="store_true", help="wyłącz detekcję twarzy")
    args = parser.parse_args()

    assets = load_assets()
    wave_tracker = WaveTracker()
    stabilizer = GestureStabilizer()
    cooldown = Cooldown(COOLDOWNS)
    scene: Scene | None = None

    cap = open_camera(args.camera, args.width, args.height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    debug, decorations = False, True
    use_face = not args.no_face and assets["face_cascade"] is not None
    fps, prev_t = 0.0, time.perf_counter()
    prev_active: set = set()

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
            inst = 1.0 / max(dt, 1e-6)
            fps = inst if fps == 0 else 0.92 * fps + 0.08 * inst

            # Efekt lustra – obraz odwrócony w poziomie, jak w prawdziwym lustrze
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if scene is None or (scene.w, scene.h) != (w, h):
                scene = Scene(w, h, assets)

            # --- 1. Detekcja dłoni (MediaPipe oczekuje RGB) ----------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False  # pozwala MediaPipe uniknąć kopiowania
            results = hands.process(rgb)

            # --- 2. Klasyfikacja gestów każdej dłoni -----------------------
            raw_landmarks = results.multi_hand_landmarks or []
            hand_infos = [classify_gesture(landmarks_to_array(lm, w, h)) for lm in raw_landmarks]
            active = stabilizer.update({hi.gesture for hi in hand_infos})

            def hand_with(gesture):
                return next((hi for hi in hand_infos if hi.gesture == gesture), None)

            # --- 3. Logika wyzwalania efektów (z cooldownem) ---------------
            # "Rozbłysk" iskier w miejscu dłoni, gdy gest właśnie się pojawił –
            # natychmiastowa informacja zwrotna, że gest został rozpoznany.
            for g in active - prev_active:
                hi = hand_with(g)
                if hi is None:
                    continue
                if g == FIST:
                    scene.sparkles.burst(*hi.palm_center, 18, (fx.ICE, fx.WHITE))
                elif g == PEACE:
                    scene.sparkles.burst(*((hi.pts[8] + hi.pts[12]) / 2), 18)
                elif g == THUMBS_UP:
                    scene.sparkles.burst(*hi.pts[4], 18)
            prev_active = active

            open_hand = hand_with(WAVE)
            wave_ready = wave_tracker.update(now, open_hand if WAVE in active else None)
            if wave_ready and not scene.santa.active and cooldown.ready(WAVE, now):
                scene.santa.trigger(now)
                cooldown.fire(WAVE, now)
                if open_hand is not None:
                    scene.sparkles.burst(*open_hand.palm_center, 28)

            if THUMBS_UP in active and cooldown.ready(THUMBS_UP, now):
                if now > scene.tree.until:
                    scene.sparkles.burst(*scene.tree.center(), 30)
                scene.tree.trigger(now)
                cooldown.fire(THUMBS_UP, now)

            # --- 4. Aktualizacja efektów ------------------------------------
            scene.snow.update(dt, FIST in active, now)
            scene.hat.update(frame, now, PEACE in active, hand_with(PEACE), use_face)
            scene.sparkles.update(dt)

            # --- 5. Rysowanie (kolejność = warstwy, od tyłu do przodu) -----
            if decorations:
                cv2.multiply(frame, scene.vignette, dst=frame, scale=1 / 255)
            if debug:
                for lm, info in zip(raw_landmarks, hand_infos):
                    draw_debug_hand(frame, lm, info)
            scene.snow.draw_far(frame)              # daleki śnieg – za wszystkim
            scene.hat.draw(frame, now)
            scene.tree.draw(frame, now)
            scene.santa.draw(frame, now, scene.sparkles, dt)
            scene.sparkles.draw(frame, now)
            scene.snow.draw_near(frame)             # bliskie śnieżynki + zaspa – z przodu
            if decorations:
                scene.garland.draw(frame, now)
            cd = {WAVE: cooldown.remaining(WAVE, now) / COOLDOWNS[WAVE]}
            flags = (f"[d] debug  [f] twarz:{'on' if use_face else 'off'}  "
                     f"[g] dekoracje  [s] zdjecie  [q] wyjscie")
            scene.hud.draw(frame, active, cd, fps, flags)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            elif key == ord("g"):
                decorations = not decorations
            elif key == ord("f") and assets["face_cascade"] is not None:
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
