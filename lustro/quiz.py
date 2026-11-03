# -*- coding: utf-8 -*-
"""
Tryb QUIZ: zadanie matematyczne, na które odpowiada się liczbą palców.

Włączanie w trakcie działania aplikacji: klawisz `m` albo kliknięcie przycisku
„QUIZ” w lewym górnym rogu ekranu. W trybie quizu efekty gestów są wstrzymane
(np. pięść nie włącza śniegu), żeby pokazywanie odpowiedzi ich nie wyzwalało.

Jak liczymy palce: dla każdej wykrytej dłoni sumujemy palce wyprostowane według
tych samych reguł co przy gestach (gesty.finger_states: palec długi jest
wyprostowany, gdy czubek leży o >15% dalej od nadgarstka niż staw PIP; kciuk –
gdy odstaje od dłoni). Dwie dłonie dają odpowiedzi od 1 do 10.

Zatwierdzanie odpowiedzi: liczba palców musi być stała przez HOLD_TIME sekund
(i dodatkowo jest wygładzana „głosowaniem” z ostatnich klatek), więc chwilowe
pomyłki detekcji nie liczą się jako odpowiedź.

Poziomy (co 2 poprawne odpowiedzi poziom rośnie):
    1 – dodawanie, 2 – odejmowanie, 3 – mnożenie i dzielenie,
    4 – potęgi, pierwiastki i kolejność działań.
"""

from __future__ import annotations

import random
from collections import Counter, deque

import cv2

from .efekty import GOLD, WHITE, rounded_rect

HOLD_TIME = 1.2      # tyle sekund trzeba trzymać tę samą liczbę palców
SUCCESS_TIME = 2.5   # czas pokazywania „Brawo!” przed kolejnym pytaniem
WRONG_TIME = 1.8     # czas pokazywania podpowiedzi po złej odpowiedzi
MAX_LEVEL = 4
SUP = {2: "²", 3: "³", 4: "⁴"}


# =============================================================================
# PYTANIA
# =============================================================================

def make_question(level: int, rng: random.Random) -> tuple[str, int]:
    """Losuje działanie dla poziomu `level`. Wynik zawsze 1–10 (do pokazania na palcach)."""
    if level <= 1:                                   # dodawanie
        a = rng.randint(1, 9)
        b = rng.randint(1, 10 - a)
        return f"{a} + {b}", a + b
    if level == 2:                                   # odejmowanie
        ans = rng.randint(1, 10)
        b = rng.randint(1, 9)
        return f"{ans + b} − {b}", ans
    if level == 3:                                   # mnożenie / dzielenie
        if rng.random() < 0.5:
            a, b = rng.choice([(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (2, 5), (5, 2), (3, 3), (1, 7), (1, 9)])
            return f"{a} · {b}", a * b
        ans = rng.randint(2, 10)
        b = rng.randint(2, 5)
        return f"{ans * b} : {b}", ans
    # poziom 4: potęgi, pierwiastki, kolejność działań
    kind = rng.randrange(4)
    if kind == 0:
        n = rng.randint(2, 10)
        return f"√{n * n}", n
    if kind == 1:
        base, exp = rng.choice([(2, 2), (2, 3), (3, 2), (2, 4)])
        v = base ** exp
        c = rng.randint(max(0, v - 10), v - 1)
        return (f"{base}{SUP[exp]} − {c}" if c else f"{base}{SUP[exp]}"), v - c
    if kind == 2:
        a, b = rng.choice([(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (1, 5)])
        return f"10 − {a} · {b}", 10 - a * b
    ans = rng.randint(2, 10)
    c = rng.randint(2, 4)
    total = ans * c
    a = rng.randint(1, total - 1)
    return f"({a} + {total - a}) : {c}", ans


def count_fingers(hands) -> int | None:
    """Suma wyprostowanych palców wszystkich dłoni; None, gdy w kadrze nie ma dłoni."""
    if not hands:
        return None
    return sum(sum(1 for v in hi.extended.values() if v) for hi in hands)


# =============================================================================
# LOGIKA QUIZU
# =============================================================================

class Quiz:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.on = False
        self.score = 0
        self.question, self.answer = "", 0
        self.state = "ask"           # ask | correct | wrong
        self.state_until = 0.0
        self.counts: deque = deque(maxlen=8)
        self.seen: int | None = None  # wygładzona liczba palców
        self.hold_value: int | None = None
        self.hold_since = 0.0
        self.locked: int | None = None  # zła odpowiedź – trzeba zmienić liczbę palców
        self.wrong_value: int | None = None

    @property
    def level(self) -> int:
        return min(MAX_LEVEL, 1 + self.score // 2)

    def toggle(self, now: float) -> None:
        self.on = not self.on
        if self.on:
            self.score = 0
            self._next(now)

    def _next(self, now: float) -> None:
        prev = self.question
        for _ in range(20):  # unikamy powtórzenia tego samego pytania
            self.question, self.answer = make_question(self.level, self.rng)
            if self.question != prev:
                break
        self.state = "ask"
        self.hold_value, self.locked = None, None
        self.counts.clear()

    def hold_progress(self, now: float) -> float:
        if self.state != "ask" or self.seen is None or self.seen == self.locked or not self.seen:
            return 0.0
        return min(1.0, (now - self.hold_since) / HOLD_TIME)

    def update(self, now: float, hands) -> str | None:
        """Wywoływane co klatkę. Zwraca 'correct' / 'wrong' w chwili oceny odpowiedzi."""
        if not self.on:
            return None
        self.counts.append(count_fingers(hands))
        visible = [c for c in self.counts if c is not None]
        # Wygładzanie: najczęstsza liczba palców z ostatnich klatek (gdy dłoń jest widoczna
        # w co najmniej połowie z nich)
        self.seen = Counter(visible).most_common(1)[0][0] if len(visible) >= len(self.counts) / 2 else None

        if self.state == "correct":
            if now >= self.state_until:
                self._next(now)
            return None
        if self.state == "wrong" and now >= self.state_until:
            self.state = "ask"

        if self.seen is None or self.seen == 0:
            self.hold_value = None
            return None
        if self.seen != self.hold_value:
            self.hold_value, self.hold_since = self.seen, now
        if self.locked is not None and self.seen != self.locked:
            self.locked = None
        if self.locked is not None or self.state != "ask":
            return None
        if now - self.hold_since >= HOLD_TIME:
            if self.seen == self.answer:
                self.score += 1
                self.state, self.state_until = "correct", now + SUCCESS_TIME
                return "correct"
            self.state, self.state_until = "wrong", now + WRONG_TIME
            self.locked = self.wrong_value = self.seen
            return "wrong"
        return None


# =============================================================================
# RYSOWANIE (wersja Python / OpenCV)
# =============================================================================

def _blend_rounded(frame, x0, y0, x1, y1, r, color, alpha):
    """Półprzezroczysty zaokrąglony prostokąt (mieszany tylko w swoim obszarze)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W - 1, int(x1)), min(H - 1, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1 + 1, x0:x1 + 1]
    layer = roi.copy()
    rounded_rect(layer, (0, 0), (x1 - x0, y1 - y0), r, color)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


class QuizView:
    """Panel z pytaniem (u góry, na środku) i przełącznik QUIZ (lewy górny róg)."""

    def __init__(self, w: int, h: int, text):
        self.w, self.h, self.text = w, h, text
        s = self.s = h / 540
        # Przełącznik – pod girlandą, w lewym górnym rogu
        self.btn = (int(12 * s), int(58 * s), int(12 * s + 150 * s), int(58 * s + 34 * s))
        # Panel z pytaniem
        pw = min(w * 0.56, 520 * s)
        self.panel = (int((w - pw) / 2), int(0.11 * h), int((w + pw) / 2), int(0.11 * h + 128 * s))

    def hit_button(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.btn
        return x0 <= x <= x1 and y0 <= y <= y1

    def draw_button(self, frame, on: bool) -> None:
        s = self.s
        x0, y0, x1, y1 = self.btn
        r = (y1 - y0) // 2
        _blend_rounded(frame, x0, y0, x1, y1, r, (40, 35, 185) if on else (30, 24, 20), 0.8)
        rounded_rect(frame, (x0, y0), (x1, y1), r, GOLD if on else (120, 115, 110), max(1, int(1.5 * s)))
        self.text.draw(frame, f"QUIZ: {'wł.' if on else 'wył.'}  [m]", ((x0 + x1) / 2, (y0 + y1) / 2),
                       int(14 * s), WHITE, anchor="mm", bold=True)

    def draw_panel(self, frame, quiz: Quiz, now: float) -> None:
        if not quiz.on:
            return
        s = self.s
        x0, y0, x1, y1 = self.panel
        cx = (x0 + x1) / 2
        bg = {"ask": (30, 24, 20), "correct": (50, 120, 40), "wrong": (40, 40, 150)}[quiz.state]
        _blend_rounded(frame, x0, y0, x1, y1, int(18 * s), bg, 0.78)
        rounded_rect(frame, (x0, y0), (x1, y1), int(18 * s), GOLD, max(1, int(2 * s)))

        self.text.draw(frame, f"QUIZ  ·  poziom {quiz.level}  ·  wynik {quiz.score}", (cx, y0 + 18 * s),
                       int(13 * s), GOLD, anchor="mm", bold=True)
        q = f"{quiz.question} = {quiz.answer}" if quiz.state == "correct" else f"{quiz.question} = ?"
        size = self.text.fit_size(q, (x1 - x0) - 40 * s, int(46 * s), bold=True)
        self.text.draw(frame, q, (cx, y0 + 60 * s), size, WHITE, anchor="mm", bold=True)

        if quiz.state == "correct":
            line = "Brawo! Świetna odpowiedź!"
        elif quiz.state == "wrong":
            line = f"Pokazujesz {quiz.wrong_value} – spróbuj jeszcze raz"
        else:
            seen = "–" if not quiz.seen else str(quiz.seen)
            line = f"Pokaż wynik na palcach   ·   widzę: {seen}"
        self.text.draw(frame, line, (cx, y0 + 98 * s), int(15 * s), (235, 235, 235), anchor="mm")

        # Pasek „trzymania” odpowiedzi
        p = quiz.hold_progress(now)
        if p > 0:
            bx0, bx1, by = int(x0 + 24 * s), int(x1 - 24 * s), int(y1 - 10 * s)
            cv2.line(frame, (bx0, by), (bx1, by), (70, 65, 60), max(2, int(4 * s)), cv2.LINE_AA)
            cv2.line(frame, (bx0, by), (int(bx0 + (bx1 - bx0) * p), by), GOLD, max(2, int(4 * s)), cv2.LINE_AA)

    def center(self):
        x0, y0, x1, y1 = self.panel
        return (x0 + x1) / 2, (y0 + y1) / 2

