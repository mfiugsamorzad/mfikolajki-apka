# mfikolajki-apka
Interaktywne lustro AR wykorzystujące OpenCV i MediaPipe do rozpoznawania gestów dłoni w czasie rzeczywistym. Projekt przygotowany na stoisko RSS podczas Mikołajek na WMFiI UG.

## Uruchomienie

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.9–3.12
pip install -r requirements.txt
python swiateczne_lustro.py            # opcje: --camera 1 --width 1280 --height 720 --fullscreen --no-face
```

Grafiki `santa.png`, `hat.png`, `tree.png` (PNG z przezroczystością) umieść obok skryptu.
Gdy ich brak, aplikacja rysuje zastępcze grafiki (sanie: czerwony prostokąt „HO HO HO”).

| Gest | Efekt |
|---|---|
| Zaciśnięta pięść | padający śnieg |
| Otwarta dłoń (przytrzymana ~0,6 s) lub machanie | sanie Mikołaja (cooldown 3 s) |
| Znak V | czapka Mikołaja na twarzy (Haar Cascade) lub nad palcami |
| Kciuk w górę | migająca choinka w prawym dolnym rogu (5 s) |

Klawisze: `q`/`Esc` – wyjście, `d` – podgląd szkieletu dłoni i rozpoznanego gestu, `f` – detekcja twarzy on/off, `s` – zrzut ekranu.
