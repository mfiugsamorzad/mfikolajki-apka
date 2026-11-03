# web/

Wersja przeglądarkowa aplikacji christmas filteRSS, publikowana na GitHub Pages. Działa w przeglądarce bez instalacji. Obraz z kamery jest analizowany na miejscu i nigdzie nie jest wysyłany.

| Plik / folder | Zawartość |
|---|---|
| `index.html` | strona: płótno z obrazem z kamery i ekran startowy z opisem gestów |
| `style.css` | wygląd ekranu startowego i czcionka DM Sans |
| `js/main.js` | kamera, modele MediaPipe (dłonie i twarz), logika wyzwalania efektów, główna pętla rysowania |
| `js/gesty.js` | rozpoznawanie 4 gestów z 21 punktów dłoni, przeniesione 1:1 z `lustro/gesty.py` |
| `js/efekty.js` | efekty nakładane na obraz: śnieg z zaspą, sanie, czapka, choinka, iskry, girlanda, logo, pasek gestów |
| `models/hand_landmarker.task` | model MediaPipe wykrywający dłonie i ich 21 punktów |
| `models/blaze_face_short_range.tflite` | model MediaPipe wykrywający twarze (do ok. 2 m od kamery) |

Grafiki i czcionki są wspólne z wersją w Pythonie (foldery `grafiki/` i `czcionki/`). Przy publikacji workflow `.github/workflows/pages.yml` kopiuje je obok `index.html`.
