# lustro/

Moduły aplikacji. Plik uruchamiany, `swiateczne_lustro.py`, leży w głównym folderze projektu i korzysta z tych modułów.

| Plik | Zawartość |
|---|---|
| `gesty.py` | rozpoznawanie 4 gestów (pięść, otwarta dłoń, znak V, kciuk w górę) z 21 punktów dłoni wyznaczonych przez MediaPipe Hands; wykrywanie machania, wygładzanie wyników między klatkami, blokada czasowa (cooldown) |
| `twarz.py` | wykrywanie twarzy (MediaPipe Face Detection, zapasowo Haar Cascade) i wyznaczanie z położenia oczu miejsca, rozmiaru i przechylenia czapki |
| `efekty.py` | grafika i animacje nakładane na obraz z kamery: śnieg z zaspą, sanie, czapka, choinka z lampkami, iskry, girlanda, logo i dolny pasek z gestami |
| `tekst.py` | rysowanie napisów czcionką DM Sans (Pillow), zamienianych na grafikę z przezroczystością nakładaną na obraz |
| `__init__.py` | oznacza folder jako pakiet Pythona |
