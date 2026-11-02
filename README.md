# mfikolajki-apka
Interaktywne lustro AR wykorzystujące OpenCV i MediaPipe do rozpoznawania gestów dłoni w czasie rzeczywistym. Projekt przygotowany na stoisko RSS podczas Mikołajek na WMFiI UG.

## Uruchomienie

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.9–3.12
pip install -r requirements.txt
python swiateczne_lustro.py            # opcje: --camera 1 --width 1280 --height 720 --fullscreen --no-face
```

### Pliki obok skryptu

| Plik | Wskazówki |
|---|---|
| `logo_rss.png` | logo samorządu, stale w prawym górnym rogu (PNG z przezroczystością) |
| `DMSans.ttf` | czcionka wszystkich napisów ([Google Fonts](https://fonts.google.com/specimen/DM+Sans)); działa też `DMSans-Regular.ttf` lub plik zmienny `DMSans[opsz,wght].ttf`, opcjonalnie `DMSans-Bold.ttf` |
| `santa.png` | sanie zwrócone **w prawo** (lecą od lewej do prawej), kadr podłużny ~3:1 |
| `hat.png` | biały otok przy **dolnej krawędzi** obrazka, przycięte puste marginesy |
| `tree.png` | choinka pionowo; lampki są rozmieszczane automatycznie na gałęziach |
| `snowflake.png` | biała śnieżynka, kwadratowy kadr (płatki są obracane i skalowane) |

Brakujące grafiki zastąpi rysowana wersja zastępcza. Wszystkie PNG muszą mieć przezroczyste tło.

### Czapka na głowie

Twarz lokalizuje **MediaPipe Face Detection** (zapasowo Haar Cascade). Czapka jest skalowana do szerokości
głowy, kotwiczona na czole (wyznaczonym z położenia oczu) i obraca się razem z przechyleniem głowy.
Gest V zakłada czapkę na 8 s – można opuścić rękę i pozować. Dopasowanie do własnego `hat.png`
ustawisz stałymi `HAT_WIDTH_SCALE`, `HAT_FOREHEAD_UP`, `HAT_ANCHOR_UV` w `swiateczne_lustro.py`
(klawisz `d` pokazuje wykryte twarze i oczy).

| Gest | Efekt |
|---|---|
| Zaciśnięta pięść | padający śnieg + zaspa |
| Otwarta dłoń (przytrzymana ~0,6 s) lub machanie | sanie Mikołaja (cooldown 3 s) |
| Znak V | czapka Mikołaja na głowie (8 s), bez twarzy w kadrze – nad palcami |
| Kciuk w górę | choinka z mrugającymi lampkami w prawym dolnym rogu (5 s) |

Klawisze: `q`/`Esc` – wyjście, `d` – podgląd szkieletu dłoni i rozpoznanego gestu, `f` – detekcja twarzy on/off, `g` – girlanda lampek, `s` – zrzut ekranu.

Struktura: `swiateczne_lustro.py` (pętla główna), `gesty.py` (rozpoznawanie gestów), `twarz.py` (twarz i pozycja czapki), `efekty.py` (grafika i efekty), `tekst.py` (napisy DM Sans przez Pillow).
