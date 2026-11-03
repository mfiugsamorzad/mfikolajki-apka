# 🎄 MFIstmas filteRSS

Interaktywne „magiczne lustro”, które reaguje na gesty dłoni. Stajesz przed kamerą, widzisz się na ekranie
jak w lustrze i wystarczy jeden gest: zaczyna padać śnieg, nad głową przelatują sanie Mikołaja, na głowie
pojawia się czapka Mikołaja, a w rogu ekranu rozbłyska choinka. Działa to podobnie do filtrów
na Instagramie, z tą różnicą, że efekty uruchamia ruch ręki, a nie przycisk.

Aplikacja ma dwie wersje z tymi samymi gestami i efektami:

- **🌐 w przeglądarce:** **https://mfiugsamorzad.github.io/mfikolajki-apka/**. Nic nie trzeba instalować: otwierasz link, klikasz „Uruchom kamerę” i gotowe.
- **🐍 w Pythonie:** program uruchamiany na laptopie, opisany niżej w „Uruchomienie krok po kroku”.

## Po co powstała ta aplikacja?

Aplikacja powstała na **stoisko Wydziałowej Rady Samorządu Studenckiego (RSS) Wydziału Matematyki,
Fizyki i Informatyki Uniwersytetu Gdańskiego** podczas **wydziałowych Mikołajek naukowych** dla uczniów szkół podstawowych
i ponadpodstawowych.

Chcemy pokazać uczniom, że informatyka i matematyka to nie tylko wzory w zeszycie. Każdy może podejść,
pomachać do kamery i od razu zobaczyć, co potrafi komputer, który „widzi”. Przy okazji łatwo
opowiedzieć, jak to działa: komputer znajduje na obrazie dłoń, zamienia ją na 21 punktów i na podstawie
**odległości między tymi punktami** rozpoznaje gest. To zwykła geometria, ta sama, której uczniowie
uczą się w szkole.

## Gesty i efekty

| Gest | Efekt |
|---|---|
| ✊ **Zaciśnięta pięść** | pada śnieg, a u dołu ekranu rośnie zaspa |
| 🖐️ **Otwarta dłoń** (przytrzymana ~0,6 s) lub **machanie** | przez ekran przelatują sanie Mikołaja (kolejne najwcześniej po 3 s) |
| ✌️ **Znak V** | czapka Mikołaja na głowie przez 8 s (można opuścić rękę i pozować) |
| 👍 **Kciuk w górę** | choinka z mrugającymi lampkami w prawym dolnym rogu (5 s) |

Na dole ekranu stale widać cztery kafelki z gestami, więc nie trzeba ich pamiętać. Kafelek gestu,
który komputer właśnie rozpoznał, podświetla się na czerwono. W prawym górnym rogu jest logo aplikacji.

## 🧮 Tryb QUIZ

Quiz włącza się i wyłącza **w trakcie działania aplikacji**: klawiszem **`m`** albo kliknięciem przycisku
**„QUIZ”** w lewym górnym rogu ekranu.

- U góry ekranu pojawia się działanie, np. `7 + 1 = ?`. Odpowiada się, **pokazując wynik na palcach**
  jednej albo dwóch rąk (1–10).
- Odpowiedź zatwierdza się, trzymając tę samą liczbę palców przez ok. 1,2 s; pasek pod pytaniem pokazuje
  postęp. Dobra odpowiedź daje „Brawo!” i fajerwerki, a co 3. dobra odpowiedź także przelot sań. Po złej
  odpowiedzi pojawia się podpowiedź i można spróbować jeszcze raz.
- Co 2 dobre odpowiedzi rośnie poziom: **1** dodawanie → **2** odejmowanie → **3** mnożenie i dzielenie →
  **4** potęgi, pierwiastki i kolejność działań, np. `√49`, `2³ − 3`, `(5 + 7) : 3`.
- W trybie quizu efekty gestów są wstrzymane, żeby pokazywanie odpowiedzi ich nie uruchamiało.
- Palce są liczone tymi samymi regułami co przy rozpoznawaniu gestów. Najpewniej działa to, gdy dłonie
  są całe w kadrze, wnętrzem do kamery, a palce wyraźnie rozsunięte.

## Jak to działa? (w skrócie, do opowiedzenia przy stoisku)

1. **Kamera** wysyła ok. 30 obrazów na sekundę. Każdy obraz jest odbijany poziomo, żeby ekran
   zachowywał się jak lustro.
2. **Sztuczna inteligencja (MediaPipe Hands)** znajduje na obrazie dłonie i zaznacza na każdej
   **21 punktów**: nadgarstek, stawy i czubki palców.
3. **Geometria rozpoznaje gest.** Program mierzy odległości między punktami, na przykład:
   - palec jest **wyprostowany**, gdy jego czubek leży dalej od nadgarstka niż jego środkowy staw;
   - **pięść** to wszystkie palce zgięte i czubki blisko środka dłoni;
   - **znak V** to wyprostowany palec wskazujący i środkowy, rozchylone w literę V.

   Odległości są dzielone przez wielkość dłoni, więc działa to tak samo z bliska i z daleka.
4. **Druga sieć (MediaPipe Face Detection)** znajduje twarz i oczy. Z położenia oczu wynika, gdzie
   jest czoło, jak duża jest głowa i jak bardzo jest przechylona. Dzięki temu czapka siedzi na głowie
   i przechyla się razem z nią.
5. **Efekty** są nakładane na obraz z kamery z uwzględnieniem przezroczystości, tak jak warstwy
   w programie graficznym.

Klawisz `d` włącza **tryb podglądu**: na ekranie widać szkielet dłoni, nazwę rozpoznanego gestu
i prostokąt wokół twarzy. Najlepiej pokazuje on uczniom, co komputer „widzi”.

Cała analiza obrazu odbywa się **na miejscu, na laptopie**. Obraz nie jest nigdzie wysyłany ani
zapisywany. Jedyny wyjątek to zrzut ekranu wykonany ręcznie klawiszem `s`.

---

## Wersja w przeglądarce

Otwórz **https://mfiugsamorzad.github.io/mfikolajki-apka/** w Chrome, Edge lub Firefoksie. Kliknij **Uruchom kamerę** i zezwól na dostęp do kamery.

- Przy pierwszym otwarciu strona pobiera modele MediaPipe (ok. 20 MB), więc warto otworzyć ją przed
  stoiskiem, przy dobrym internecie. Potem przeglądarka trzyma je w pamięci podręcznej.
- Obraz z kamery jest analizowany wyłącznie w przeglądarce i nigdzie nie jest wysyłany.
- Klawisze: `m` quiz (albo przycisk „QUIZ” na ekranie), `d` tryb podglądu, `f` wykrywanie twarzy, `g` girlanda, `s` zapis zdjęcia (pobiera plik PNG),
  `Enter` lub podwójne kliknięcie to pełny ekran.

**Uruchomienie lokalne** (np. do zmian w kodzie): w głównym folderze projektu uruchom
`python -m http.server 8000` i otwórz http://localhost:8000/web/. Kamera działa tylko przez `https://`
albo `localhost`, dlatego samo otwarcie pliku `index.html` nie wystarczy.

**Publikacja:** po każdej zmianie w `web/`, `grafiki/` lub `czcionki/` na gałęzi `main` workflow
`.github/workflows/pages.yml` sam publikuje nową wersję strony (zakładka **Actions** na GitHubie).

---

## Uruchomienie krok po kroku (wersja w Pythonie)

### 1. Wymagania

- laptop z kamerą internetową (wbudowaną lub na USB),
- **Python 3.12** (działa też 3.9–3.11; **nie działa na 3.13 i nowszych**, bo potrzebna wersja
  MediaPipe nie jest dla nich wydana): https://www.python.org/downloads/.
  Na Windows zaznacz przy instalacji **„Add Python to PATH”**.

### 2. Pobranie kodu

```bash
git clone https://github.com/mfiugsamorzad/mfikolajki-apka.git
cd mfikolajki-apka
```

Bez gita: na stronie repozytorium kliknij **Code → Download ZIP** i rozpakuj archiwum.

Aktualizacja do najnowszej wersji (gdy kod był już wcześniej pobrany przez git):

```bash
git pull
```

### 3. Grafiki i czcionka

Wrzuć pliki do odpowiednich folderów. Nazwy muszą być **dokładnie takie**: małe litery, bez spacji.

```
mfikolajki-apka/
├── grafiki/
│   ├── logo_filterss.png logo aplikacji (prawy górny róg)
│   ├── logo_rss.png      logo samorządu (ekran startowy)
│   ├── santa.png         sanie Mikołaja
│   ├── hat.png           czapka Mikołaja
│   ├── tree.png          choinka
│   └── snowflake.png     śnieżynka
└── czcionki/
    ├── DMSans-Regular.ttf
    └── DMSans-SemiBold.ttf   (lub DMSans-Bold.ttf)
```

| Plik | Na co uważać |
|---|---|
| wszystkie `.png` | **przezroczyste tło**, inaczej grafika będzie miała wokół siebie prostokąt (puste marginesy program przytnie sam) |
| `santa.png` | sanie zwrócone **w prawo** (lecą od lewej do prawej), kadr podłużny ~3:1 |
| `hat.png` | biały otok przy **dolnej krawędzi** obrazka, bez pustych marginesów |
| `tree.png` | choinka ustawiona pionowo; lampki program dorysuje sam |
| `snowflake.png` | biała śnieżynka w kwadratowym kadrze |
| czcionka | z https://fonts.google.com/specimen/DM+Sans (**Get font → Download all**) weź z folderu `static/` plik `DMSans-Regular.ttf` oraz `DMSans-SemiBold.ttf` lub `DMSans-Bold.ttf` |

Jeśli którejś grafiki brakuje, aplikacja i tak się uruchomi i narysuje własną wersję zastępczą.

### 4. Instalacja bibliotek (jednorazowo)

W terminalu, w folderze projektu:

**Windows**
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Start

Przy każdym kolejnym uruchomieniu najpierw aktywuj środowisko (`.venv\Scripts\activate` na Windows,
`source .venv/bin/activate` na macOS i Linuksie), a potem:

```bash
python swiateczne_lustro.py --fullscreen
```

Bez `--fullscreen` aplikacja otworzy się w zwykłym oknie.

| Opcja | Działanie |
|---|---|
| `--fullscreen` | pełny ekran (polecane na stoisko) |
| `--camera 1` | inna kamera, np. zewnętrzna na USB (domyślnie `0`) |
| `--width 1280 --height 720` | rozdzielczość obrazu z kamery (domyślnie 960×540) |
| `--no-face` | bez wykrywania twarzy (czapka pojawi się wtedy nad palcami) |

### Klawisze w trakcie działania

| Klawisz | Działanie |
|---|---|
| `q` lub `Esc` | wyjście |
| `m` | włącz lub wyłącz tryb QUIZ (to samo robi kliknięcie przycisku „QUIZ” na ekranie) |
| `d` | tryb podglądu: szkielet dłoni, rozpoznany gest, wykryte twarze |
| `s` | zrzut ekranu (plik `lustro_…png` w folderze, z którego uruchomiono aplikację) |
| `g` | włącz lub wyłącz girlandę lampek u góry ekranu |
| `f` | włącz lub wyłącz wykrywanie twarzy |

---

## Wskazówki na stoisko

- **Światło:** twarz i dłonie powinny być dobrze oświetlone. Unikaj okna albo lampy za plecami
  osoby stojącej przed kamerą.
- **Odległość:** najlepiej 0,5–2 m od kamery. Dłoń powinna być cała w kadrze i skierowana
  wnętrzem do kamery.
- **Kamera:** na wysokości twarzy, najlepiej nad dużym monitorem lub telewizorem, który pokazuje obraz.
- **Kilka osób naraz:** czapkę dostaje każda wykryta twarz, a gesty rozpoznawane są z maksymalnie
  dwóch dłoni.
- **Przed otwarciem stoiska:** uruchom aplikację, sprawdź wszystkie 4 gesty i zrób zrzut klawiszem `s`.

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---|---|
| Czarny obraz | Daj programowi dostęp do kamery: **Windows:** Ustawienia → Prywatność → Kamera, **macOS:** Ustawienia systemowe → Prywatność i ochrona → Kamera (dla Terminala). Zamknij Teams, Zoom itp. Sprawdź zasłonkę kamery. |
| „nie można otworzyć kamery” | Spróbuj `--camera 1`. |
| „nie znaleziono mediapipe.solutions” | Zła wersja Pythona: potrzebny jest 3.12 (krok 1), a środowisko trzeba utworzyć od nowa. |
| Prostokąt wokół grafiki | PNG nie ma przezroczystego tła (w konsoli pojawi się `[uwaga]`). |
| `Brak pliku DMSans.ttf` | Czcionka nie leży w `czcionki/` albo ma podwójne rozszerzenie (np. `DMSans-Regular.ttf.ttf`). |
| Gest nie jest rozpoznawany | Włącz `d` i zobacz, czy dłoń ma narysowany szkielet. Popraw oświetlenie i pokaż całą dłoń wnętrzem do kamery. |
| Czapka źle dopasowana | Zmień stałe `HAT_WIDTH_SCALE` (rozmiar), `HAT_FOREHEAD_UP` (wysokość) i `HAT_ANCHOR_UV` (punkt obrazka, który trafia w czoło) na początku `swiateczne_lustro.py`. |
| Twarze daleko od kamery nie są wykrywane | Ustaw `FACE_MODEL = 1` w `swiateczne_lustro.py` (model dalekiego zasięgu, do ~5 m). |
| Ramki (kafelki, panel quizu, logo) za duże lub za małe, np. na dużym wyświetlaczu | Zmień stałą `UI_SCALE` (domyślnie 0,72) na początku `swiateczne_lustro.py` albo `web/js/main.js`. Mniejsza liczba oznacza mniejsze ramki. |
| Aplikacja się zacina | Uruchom z mniejszą rozdzielczością: `--width 640 --height 360`. |

---

## Dla programistów

**Technologie:** Python, OpenCV, MediaPipe (Hands, Face Detection), NumPy, Pillow.

```
swiateczne_lustro.py   pętla główna: kamera, MediaPipe, logika wyzwalania efektów, konfiguracja
lustro/
├── gesty.py           rozpoznawanie gestów z 21 punktów dłoni, wygładzanie, cooldown
├── twarz.py           wykrywanie twarzy, pozycja i przechylenie czapki
├── efekty.py          grafika: śnieg, sanie, czapka, choinka, iskry, girlanda, logo, pasek gestów
├── quiz.py            tryb QUIZ: pytania, liczenie palców, ocena odpowiedzi, panel i przełącznik
└── tekst.py           napisy czcionką DM Sans (Pillow → NumPy/OpenCV)
grafiki/               pliki PNG
czcionki/              DM Sans
```

- **Stabilność gestów:** gest liczy się dopiero, gdy pojawi się w co najmniej 60% z ostatnich 6 klatek.
  Sanie mają 3 s przerwy między wywołaniami, choinka 2 s.
- **Wydajność:** lżejszy model dłoni (`model_complexity=0`), mieszanie przezroczystości tylko w obszarze
  grafiki, pamięć podręczna przeskalowanych grafik i gotowych napisów. Efekty zajmują ok. 10 ms na
  klatkę, a całość działa płynnie na typowym laptopie.
- **Konfiguracja** (czasy efektów, cooldowny, dopasowanie czapki) jest w stałych na początku
  `swiateczne_lustro.py`. Każda ma komentarz.
- Szczegółowe komentarze o rozpoznawaniu gestów z punktów dłoni są w `lustro/gesty.py`.

**Wersja przeglądarkowa** (`web/`): czysty JavaScript bez budowania, MediaPipe Tasks Vision 0.10.21
(Hand Landmarker, Face Detector) ładowany z CDN jsDelivr, modele w `web/models/`, rysowanie na `<canvas>`.
Logika gestów w `web/js/gesty.js` to przeniesiony 1:1 kod `lustro/gesty.py`, z tymi samymi progami.
Zawartość folderu opisuje `web/README.md`.
