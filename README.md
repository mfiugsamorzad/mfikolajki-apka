# mfikolajki-apka
Interaktywne lustro AR wykorzystujące OpenCV i MediaPipe do rozpoznawania gestów dłoni w czasie rzeczywistym. Projekt przygotowany na stoisko RSS podczas Mikołajek na WMFiI UG.

## Uruchomienie

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.9–3.12
pip install -r requirements.txt
python swiateczne_lustro.py            # opcje: --camera 1 --width 1280 --height 720 --fullscreen --no-face
```

### Grafiki

Umieść obok skryptu 4 pliki **PNG z przezroczystym tłem** (brakujące zastąpi rysowana grafika):

| Plik | Wskazówki |
|---|---|
| `santa.png` | sanie zwrócone **w prawo** (lecą od lewej do prawej), kadr podłużny ~3:1 |
| `hat.png` | biały otok przy **dolnej krawędzi** obrazka, przycięte puste marginesy |
| `tree.png` | choinka pionowo; lampki są rozmieszczane automatycznie na gałęziach |
| `snowflake.png` | biała śnieżynka, kwadratowy kadr (płatki są obracane i skalowane) |

| Gest | Efekt |
|---|---|
| Zaciśnięta pięść | padający śnieg + zaspa |
| Otwarta dłoń (przytrzymana ~0,6 s) lub machanie | sanie Mikołaja (cooldown 3 s) |
| Znak V | czapka Mikołaja na twarzy (Haar Cascade) lub nad palcami |
| Kciuk w górę | choinka z mrugającymi lampkami w prawym dolnym rogu (5 s) |

Klawisze: `q`/`Esc` – wyjście, `d` – podgląd szkieletu dłoni i rozpoznanego gestu, `f` – detekcja twarzy on/off, `g` – dekoracje (girlanda + winieta), `s` – zrzut ekranu.

Struktura: `swiateczne_lustro.py` (pętla główna), `gesty.py` (rozpoznawanie gestów), `efekty.py` (grafika i efekty).
