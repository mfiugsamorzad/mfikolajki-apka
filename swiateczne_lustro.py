#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
christmas filteRSS
==================

Lustro AR w stylu filtrów z Instagrama: na ekranie widać obraz NA ŻYWO z kamery
(odbity lustrzanie), a efekty świąteczne są nakładane bezpośrednio na tę klatkę
z uwzględnieniem przezroczystości (alpha blending). Nie istnieje żadne osobne
"płótno" – bazą każdej operacji rysowania jest klatka z cv2.VideoCapture.

MediaPipe Hands wyznacza 21 landmarków każdej dłoni, a z ich geometrii
rozpoznajemy 4 gesty:

    Gest                           Efekt
    -----------------------------  --------------------------------------------
    Zaciśnięta pięść  (FIST)       padający śnieg (snowflake.png) + zaspa
    Otwarta dłoń / machanie (WAVE) przelatujące sanie Mikołaja (santa.png)
    Znak V            (PEACE)      czapka Mikołaja NA GŁOWIE (hat.png) – twarz
                                   lokalizuje MediaPipe Face Detection
    Kciuk w górę      (THUMBS_UP)  choinka z mrugającymi lampkami (tree.png)

Stale widoczne: logo samorządu (logo_rss.png) w prawym górnym rogu.
Wszystkie napisy: czcionka DM Sans (DMSans.ttf) renderowana przez Pillow.

Struktura projektu:
    swiateczne_lustro.py – pętla główna: kamera, MediaPipe, logika wyzwalania
    lustro/gesty.py      – rozpoznawanie gestów z landmarków (+ stabilizacja, cooldown)
    lustro/twarz.py      – lokalizacja twarzy i pozycji czapki na głowie
    lustro/efekty.py     – grafika: sprite'y, śnieg, sanie, czapka, choinka, logo, HUD
    lustro/tekst.py      – napisy DM Sans (Pillow -> NumPy/OpenCV)
    grafiki/             – santa.png, hat.png, tree.png, snowflake.png, logo_rss.png
    czcionki/            – DMSans.ttf (opcjonalnie DMSans-Bold.ttf)

Sterowanie z klawiatury:
    q / ESC – wyjście            d – szkielet dłoni i rozpoznany gest (debug)
    f       – detekcja twarzy    g – girlanda lampek
    s       – zrzut ekranu

Wymagania: Python 3.9–3.12, opencv-python, numpy, Pillow, mediapipe<=0.10.21
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
    mp_face_detection = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
except (ImportError, AttributeError):  # pragma: no cover - zależy od instalacji
    sys.exit(
        "BŁĄD: nie znaleziono 'mediapipe.solutions'.\n"
        "Zainstaluj zgodną wersję: pip install \"mediapipe<=0.10.21\" "
        "(wymaga Pythona 3.9–3.12)."
    )

from lustro import efekty as fx
from lustro.gesty import (FIST, PEACE, THUMBS_UP, WAVE, Cooldown, GestureStabilizer, HandInfo,
                          WaveTracker, classify_gesture, landmarks_to_array)
from lustro.tekst import TextRenderer
from lustro.twarz import FaceLocator, head_pose

# =============================================================================
# KONFIGURACJA
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHICS_DIR = os.path.join(BASE_DIR, "grafiki")   # pliki PNG
FONTS_DIR = os.path.join(BASE_DIR, "czcionki")     # DM Sans (.ttf)
ASSET_FILES = {
    "santa": "santa.png",
    "hat": "hat.png",
    "tree": "tree.png",
    "snowflake": "snowflake.png",
    "logo": "logo_rss.png",
}
# Czcionka DM Sans. Akceptujemy też nazwy plików z paczki pobranej z Google Fonts.
FONT_FILES = ["DMSans.ttf", "DMSans-Regular.ttf", "DMSans[opsz,wght].ttf",
              "DMSans-VariableFont_opsz,wght.ttf"]
# Pogrubienie (opcjonalne): pierwszy znaleziony plik; bez niego – oś wagi czcionki zmiennej
FONT_BOLD_FILES = ["DMSans-Bold.ttf", "DMSans-SemiBold.ttf", "DMSans-ExtraBold.ttf"]

WINDOW_NAME = "christmas filteRSS"

# Cooldown (s) – minimalny odstęp między kolejnymi WYZWOLENIAMI danego efektu.
# Dla sań to kluczowe: bez tego przy 30 FPS każdy kadr z otwartą dłonią
# wysyłałby nowe sanie i animacje nakładałyby się na siebie.
COOLDOWNS = {
    WAVE: 3.0,       # sanie Mikołaja
    THUMBS_UP: 2.0,  # (ponowne) pokazanie choinki
}

SANTA_DURATION = 3.4          # czas przelotu sań przez ekran (s)
TREE_SHOW_TIME = 5.0          # jak długo widać choinkę po geście (s)

# --- Czapka na głowie ---------------------------------------------------------
HAT_SHOW_TIME = 8.0           # gest V "zakłada" czapkę na tyle sekund (można opuścić rękę)
HAT_WIDTH_SCALE = 1.6         # szerokość czapki względem szerokości głowy
HAT_FOREHEAD_UP = 0.85        # czoło: tyle rozstawów oczu nad linią oczu
HAT_ANCHOR_UV = (0.5, 0.85)   # punkt obrazka hat.png (ułamki szer./wys.), który ląduje na czole
                              # – domyślnie środek dolnej części otoku; dostrój pod swój plik
FACE_DETECT_EVERY_N = 1       # MediaPipe Face Detection jest szybkie – co klatkę (płynne śledzenie)
FACE_MODEL = 0                # 0 = twarze do ~2 m od kamery, 1 = do ~5 m


# =============================================================================
# ZASOBY
# =============================================================================

def find_file(folder: str, name: str) -> str:
    """
    Ścieżka do pliku zasobu: najpierw właściwy podfolder (grafiki/ lub czcionki/),
    a gdy go tam nie ma – główny folder projektu (plik wrzucony "nie tam" też zadziała).
    """
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return path
    fallback = os.path.join(BASE_DIR, name)
    return fallback if os.path.isfile(fallback) else path


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
    return trim_transparent(img)


def trim_transparent(img: np.ndarray, pad: int = 2) -> np.ndarray:
    """
    Przycina przezroczyste marginesy grafiki. Dzięki temu rozmiar i położenie
    (np. logo w rogu, czapka na czole) zależą od samej treści obrazka, a nie od
    tego, ile pustego miejsca zostawiono wokół niej w pliku PNG.
    """
    ys, xs = np.nonzero(img[:, :, 3] > 8)
    if len(xs) == 0:
        return img
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def load_assets() -> dict:
    assets = {k: load_rgba(find_file(GRAPHICS_DIR, f)) for k, f in ASSET_FILES.items()}
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
    if assets["logo"] is None:
        print("[uwaga] Brak logo_rss.png – logo nie będzie wyświetlane.")
    return assets


# =============================================================================
# SCENA – wszystkie efekty zależne od rozmiaru klatki
# =============================================================================

class Scene:
    """Tworzona po odczytaniu pierwszej klatki (efekty skalują się do jej rozmiaru)."""

    def __init__(self, w: int, h: int, assets: dict, text: TextRenderer):
        self.w, self.h = w, h
        self.hud = fx.Hud(w, h, text)
        self.sparkles = fx.Sparkles(h / 540)
        self.snow = fx.SnowEffect(w, h, assets["snowflake"])
        self.santa = fx.SantaEffect(w, h, assets["santa"], SANTA_DURATION, text)
        self.hat = fx.HatEffect(assets["hat"], HAT_SHOW_TIME, HAT_ANCHOR_UV,
                                HAT_WIDTH_SCALE, HAT_FOREHEAD_UP)
        self.tree = fx.TreeEffect(w, h, assets["tree"], TREE_SHOW_TIME, self.hud.height + 6)
        self.logo = fx.Logo(w, h, assets["logo"]) if assets["logo"] is not None else None
        garland_end = self.logo.left - self.logo.margin if self.logo else None
        self.garland = fx.Garland(w, h, garland_end)


def draw_debug(frame, raw_landmarks, hand_infos, faces, text: TextRenderer) -> None:
    """Szkielet dłoni, rozpoznany gest, prostokąty twarzy i oczy (tryb debug)."""
    for lm, info in zip(raw_landmarks, hand_infos):
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                               mp_styles.get_default_hand_landmarks_style(),
                               mp_styles.get_default_hand_connections_style())
        c = tuple(int(v) for v in info.palm_center)
        cv2.circle(frame, c, 5, (0, 255, 255), -1)
        cv2.circle(frame, c, int(info.hand_size * 0.95), (0, 255, 255), 1)  # próg pięści
        fingers = "".join("1" if info.extended.get(f) else "0"
                          for f in ["thumb", "index", "middle", "ring", "pinky"])
        x, y = int(info.pts[:, 0].min()), int(info.pts[:, 1].min()) - 8
        text.draw(frame, f"{info.gesture} [{fingers}]", (x, max(70, y)), 18, (0, 255, 255),
                  anchor="lb", bold=True, outline=1)
    for f in faces or []:
        x, y, w, h = (int(v) for v in f.box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 1)
        for e in (f.eye_left, f.eye_right):
            cv2.circle(frame, (int(e[0]), int(e[1])), 3, (255, 200, 0), -1)


# =============================================================================
# KAMERA
# =============================================================================

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # Na Windows backend DirectShow startuje szybciej i rzadziej zwraca czarne
    # klatki niż domyślny MSMF.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"BŁĄD: nie można otworzyć kamery o indeksie {index}. "
                 "Sprawdź, czy inna aplikacja jej nie używa, lub spróbuj --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimalne opóźnienie (gdy backend wspiera)
    return cap


class BlackFrameWatch:
    """Ostrzega, gdy kamera przez dłuższy czas zwraca czarny obraz (brak uprawnień itp.)."""

    def __init__(self, seconds: float = 3.0):
        self.seconds = seconds
        self.dark_since = None
        self.warned = False

    def check(self, frame: np.ndarray, now: float) -> None:
        if self.warned:
            return
        if float(frame[::16, ::16].mean()) < 4.0:
            self.dark_since = self.dark_since or now
            if now - self.dark_since > self.seconds:
                self.warned = True
                print("[uwaga] Kamera zwraca czarny obraz. Sprawdź: uprawnienia do kamery "
                      "(macOS: Ustawienia > Prywatność > Kamera; Windows: Ustawienia > "
                      "Prywatność > Kamera), zasłonkę obiektywu lub inny indeks (--camera 1).")
        else:
            self.dark_since = None


# =============================================================================
# PĘTLA GŁÓWNA
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="christmas filteRSS – świąteczne lustro AR (OpenCV + MediaPipe)")
    parser.add_argument("--camera", type=int, default=0, help="indeks kamery (domyślnie 0)")
    parser.add_argument("--width", type=int, default=960, help="szerokość obrazu kamery")
    parser.add_argument("--height", type=int, default=540, help="wysokość obrazu kamery")
    parser.add_argument("--fullscreen", action="store_true", help="tryb pełnoekranowy")
    parser.add_argument("--no-face", action="store_true", help="wyłącz detekcję twarzy")
    args = parser.parse_args()

    assets = load_assets()
    font = next((find_file(FONTS_DIR, f) for f in FONT_FILES
                 if os.path.isfile(find_file(FONTS_DIR, f))), os.path.join(FONTS_DIR, FONT_FILES[0]))
    bold = next((find_file(FONTS_DIR, f) for f in FONT_BOLD_FILES
                 if os.path.isfile(find_file(FONTS_DIR, f))), None)
    text = TextRenderer(font, bold)
    haar_dirs = [GRAPHICS_DIR, BASE_DIR] + ([cv2.data.haarcascades] if hasattr(cv2, "data") else [])
    faces_locator = FaceLocator(mp_face_detection, haar_dirs, FACE_MODEL)
    use_face = not args.no_face and faces_locator.available

    wave_tracker = WaveTracker()
    stabilizer = GestureStabilizer()
    cooldown = Cooldown(COOLDOWNS)
    black_watch = BlackFrameWatch()
    scene: Scene | None = None

    cap = open_camera(args.camera, args.width, args.height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    debug, garland_on = False, True
    fps, fps_shown, fps_shown_t = 0.0, 0, 0.0
    prev_t = time.perf_counter()
    prev_active: set = set()
    frame_idx = 0

    # model_complexity=0 – lżejszy model dłoni, wyraźnie szybszy na CPU,
    # w zupełności wystarczający do rozpoznawania gestów.
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            # ---------------------------------------------------------------
            # 0. Klatka z kamery = BAZA wszystkich operacji rysowania.
            # ---------------------------------------------------------------
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] Nie udało się odczytać klatki z kamery.")
                break
            frame_idx += 1

            now = time.perf_counter()
            dt = min(now - prev_t, 0.1)  # ograniczenie dt po "zawieszeniu" okna
            prev_t = now
            inst = 1.0 / max(dt, 1e-6)
            fps = inst if fps == 0 else 0.92 * fps + 0.08 * inst
            if now - fps_shown_t > 0.5:  # licznik odświeżany 2× na s (czytelność + cache napisów)
                fps_shown, fps_shown_t = int(round(fps)), now
            black_watch.check(frame, now)

            # Odbicie lustrzane: lewa ręka osoby jest po lewej stronie ekranu,
            # dokładnie jak w prawdziwym lustrze. Całe przetwarzanie (dłonie,
            # twarz, efekty) działa już na odbitej klatce, więc współrzędne się zgadzają.
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if scene is None or (scene.w, scene.h) != (w, h):
                scene = Scene(w, h, assets, text)

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

            # --- 3. Detekcja twarzy (tylko gdy czapka jest / zaraz będzie widoczna)
            faces = None
            if use_face and (PEACE in active or scene.hat.visible(now)) \
                    and frame_idx % FACE_DETECT_EVERY_N == 0:
                faces = faces_locator.locate(rgb, frame)

            # --- 4. Logika wyzwalania efektów (z cooldownem) ---------------
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

            # --- 5. Aktualizacja efektów ------------------------------------
            scene.snow.update(dt, FIST in active, now)
            scene.hat.update(now, PEACE in active, hand_with(PEACE), faces, head_pose)
            scene.sparkles.update(dt)

            # --- 6. Nakładki na klatkę z kamery (od tyłu do przodu) --------
            # Każda nakładka to alpha blending na wycinku klatki – piksele
            # kamery poza grafikami pozostają nietknięte.
            if debug:
                draw_debug(frame, raw_landmarks, hand_infos, faces, text)
            scene.snow.draw_far(frame)              # daleki śnieg – za wszystkim
            scene.hat.draw(frame, now)
            scene.tree.draw(frame, now)
            scene.santa.draw(frame, now, scene.sparkles, dt)
            scene.sparkles.draw(frame, now)
            scene.snow.draw_near(frame)             # bliskie śnieżynki + zaspa – z przodu
            if garland_on:
                scene.garland.draw(frame, now)
            if scene.logo is not None:
                scene.logo.draw(frame)              # logo zawsze widoczne, na wierzchu
            cd = {WAVE: cooldown.remaining(WAVE, now) / COOLDOWNS[WAVE]}
            info = (f"{fps_shown} FPS   ·   [d] debug   [f] twarz: {'wł.' if use_face else 'wył.'}   "
                    f"[g] girlanda   [s] zdjęcie   [q] wyjście")
            scene.hud.draw(frame, active, cd, info)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            elif key == ord("g"):
                garland_on = not garland_on
            elif key == ord("f") and faces_locator.available:
                use_face = not use_face
            elif key == ord("s"):
                name = time.strftime("lustro_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, frame)
                print(f"[info] Zapisano {name}")
            # Zamknięcie okna krzyżykiem
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    faces_locator.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
