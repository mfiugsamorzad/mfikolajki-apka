# -*- coding: utf-8 -*-
"""
Renderowanie tekstu czcionką DM Sans (plik DMSans.ttf) przez Pillow.

cv2.putText obsługuje tylko wbudowane czcionki Hershey (bez plików .ttf i bez
polskich znaków), dlatego każdy napis rysujemy biblioteką PIL na małym,
przezroczystym obrazku RGBA, konwertujemy go do tablicy NumPy (format OpenCV,
BGRA) i nakładamy na klatkę z kamery przez alpha blending.

Dlaczego nie konwertujemy CAŁEJ klatki do PIL i z powrotem? Taka podwójna
konwersja kosztuje kilka milisekund na każdą klatkę. Napisy w aplikacji
powtarzają się (te same etykiety co klatkę), więc gotowe obrazki napisów
trzymamy w pamięci podręcznej – efekt na ekranie jest identyczny, a koszt
rysowania tekstu spada praktycznie do zera.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from efekty import Sprite


class TextRenderer:
    """Napisy DM Sans jako sprite'y (z cache LRU)."""

    CACHE_SIZE = 400

    def __init__(self, font_path: str, bold_path: str | None = None):
        self.font_path = font_path if os.path.isfile(font_path) else None
        self.bold_path = bold_path if bold_path and os.path.isfile(bold_path) else None
        if self.font_path is None:
            print(f"[uwaga] Brak pliku {os.path.basename(font_path)} – napisy zostaną "
                  "wyrenderowane domyślną czcionką Pillow. Dodaj DMSans.ttf obok skryptu.")
        self._fonts: dict = {}
        self._cache: OrderedDict = OrderedDict()
        self._faux_bold = False  # gdy brak wersji pogrubionej – pogrubiamy obrysem

    # --- czcionki --------------------------------------------------------
    def font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        size = max(6, int(size))
        key = (size, bold)
        f = self._fonts.get(key)
        if f is not None:
            return f
        if self.font_path is None:
            f = ImageFont.load_default(size=size)
        elif bold and self.bold_path:
            f = ImageFont.truetype(self.bold_path, size)
        else:
            f = ImageFont.truetype(self.font_path, size)
            if bold:
                f = self._make_bold(f)
        self._fonts[key] = f
        return f

    def _make_bold(self, f):
        """DM Sans z Google Fonts to czcionka zmienna (oś 'wght') – ustawiamy wagę 700."""
        try:
            values = []
            for axis in f.get_variation_axes():
                name = axis.get("name", b"")
                name = name.decode() if isinstance(name, bytes) else str(name)
                if name.lower().startswith("weight") or name.lower() == "wght":
                    values.append(min(max(700, axis["minimum"]), axis["maximum"]))
                else:
                    values.append(axis.get("default", axis["minimum"]))
            f.set_variation_by_axes(values)
        except (OSError, AttributeError, KeyError):
            self._faux_bold = True  # czcionka statyczna – pogrubienie obrysem
        return f

    def fit_size(self, text: str, max_w: float, size: int, bold: bool = False) -> int:
        """Największy rozmiar (<= size), przy którym tekst mieści się w max_w pikseli."""
        size = int(size)
        while size > 7 and self.font(size, bold).getlength(text) > max_w:
            size -= 1
        return size

    # --- renderowanie ----------------------------------------------------
    def sprite(self, text: str, size: int, color=(255, 255, 255), bold: bool = False,
               outline: int = 0, outline_color=(20, 18, 16), shadow: bool = False) -> Sprite:
        """
        Zwraca napis jako Sprite. Kolory w kolejności BGR (jak w OpenCV).
        Wysokość obrazka = ascent + descent czcionki, więc napisy o tym samym
        rozmiarze są wyrównane do wspólnej linii bazowej niezależnie od treści.
        """
        key = (text, int(size), tuple(color), bold, outline, tuple(outline_color), shadow)
        spr = self._cache.get(key)
        if spr is not None:
            self._cache.move_to_end(key)
            return spr

        font = self.font(size, bold)
        stroke = outline
        stroke_rgb = tuple(outline_color[::-1])
        rgb = tuple(int(c) for c in color[::-1])
        if bold and self._faux_bold and outline == 0:
            stroke, stroke_rgb = 1, rgb

        ascent, descent = font.getmetrics()
        left, _, right, _ = font.getbbox(text, stroke_width=stroke, anchor="ls")
        sh = max(1, int(size * 0.06)) if shadow else 0
        pad = 2
        W = int(right - left) + 2 * pad + sh
        H = ascent + descent + 2 * stroke + 2 * pad + sh
        img = Image.new("RGBA", (max(W, 1), max(H, 1)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        origin = (pad - left, pad + stroke + ascent)
        if shadow:
            draw.text((origin[0] + sh, origin[1] + sh), text, font=font, anchor="ls",
                      fill=(0, 0, 0, 120), stroke_width=stroke, stroke_fill=(0, 0, 0, 120))
        draw.text(origin, text, font=font, anchor="ls", fill=rgb + (255,),
                  stroke_width=stroke, stroke_fill=stroke_rgb + (255,))

        rgba = np.asarray(img)
        spr = Sprite.from_bgra(rgba[:, :, [2, 1, 0, 3]])  # RGBA (PIL) -> BGRA (OpenCV)
        self._cache[key] = spr
        if len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return spr

    def draw(self, frame: np.ndarray, text: str, pos, size: int, color=(255, 255, 255),
             anchor: str = "lt", opacity: float = 1.0, **kw) -> Sprite:
        """
        Rysuje napis na klatce. anchor: pierwsza litera = poziom (l/m/r),
        druga = pion (t/m/b) – np. "mm" centruje napis w punkcie pos.
        """
        spr = self.sprite(text, size, color, **kw)
        x, y = pos
        if anchor[0] == "m":
            x -= spr.w / 2
        elif anchor[0] == "r":
            x -= spr.w
        if anchor[1] == "m":
            y -= spr.h / 2
        elif anchor[1] == "b":
            y -= spr.h
        spr.draw(frame, x, y, opacity)
        return spr
