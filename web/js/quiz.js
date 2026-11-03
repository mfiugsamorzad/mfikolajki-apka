// Tryb QUIZ: zadanie matematyczne, na które odpowiada się liczbą palców.
// Port 1:1 z lustro/quiz.py (te same poziomy, czasy i zasady oceniania).
//
// Włączanie: klawisz `m` albo kliknięcie przycisku „QUIZ” w lewym górnym rogu.
// W trybie quizu efekty gestów są wstrzymane. Palce liczymy tymi samymi
// regułami co przy gestach (gesty.js – fingerStates), dwie dłonie = 1…10.

import { GOLD, WHITE, roundRectPath, fitFont } from "./efekty.js";

const HOLD_TIME = 1.2;    // tyle sekund trzeba trzymać tę samą liczbę palców
const SUCCESS_TIME = 2.5; // „Brawo!” przed kolejnym pytaniem
const WRONG_TIME = 1.8;   // podpowiedź po złej odpowiedzi
const MAX_LEVEL = 4;
const SUP = { 2: "²", 3: "³", 4: "⁴" };
const FONT = '"DM Sans", system-ui, sans-serif';

const randint = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const choice = (arr) => arr[Math.floor(Math.random() * arr.length)];

/** Losuje działanie dla poziomu. Wynik zawsze 1–10 (do pokazania na palcach). */
export function makeQuestion(level) {
  if (level <= 1) { // dodawanie
    const a = randint(1, 9), b = randint(1, 10 - a);
    return [`${a} + ${b}`, a + b];
  }
  if (level === 2) { // odejmowanie
    const ans = randint(1, 10), b = randint(1, 9);
    return [`${ans + b} − ${b}`, ans];
  }
  if (level === 3) { // mnożenie / dzielenie
    if (Math.random() < 0.5) {
      const [a, b] = choice([[2, 2], [2, 3], [3, 2], [2, 4], [4, 2], [2, 5], [5, 2], [3, 3], [1, 7], [1, 9]]);
      return [`${a} · ${b}`, a * b];
    }
    const ans = randint(2, 10), b = randint(2, 5);
    return [`${ans * b} : ${b}`, ans];
  }
  // poziom 4: potęgi, pierwiastki, kolejność działań
  const kind = randint(0, 3);
  if (kind === 0) {
    const n = randint(2, 10);
    return [`√${n * n}`, n];
  }
  if (kind === 1) {
    const [base, exp] = choice([[2, 2], [2, 3], [3, 2], [2, 4]]);
    const v = base ** exp;
    const c = randint(Math.max(0, v - 10), v - 1);
    return [c ? `${base}${SUP[exp]} − ${c}` : `${base}${SUP[exp]}`, v - c];
  }
  if (kind === 2) {
    const [a, b] = choice([[2, 2], [2, 3], [3, 2], [2, 4], [4, 2], [3, 3], [1, 5]]);
    return [`10 − ${a} · ${b}`, 10 - a * b];
  }
  const ans = randint(2, 10), c = randint(2, 4), total = ans * c, a = randint(1, total - 1);
  return [`(${a} + ${total - a}) : ${c}`, ans];
}

/** Suma wyprostowanych palców wszystkich dłoni; null, gdy w kadrze nie ma dłoni. */
export function countFingers(hands) {
  if (!hands || !hands.length) return null;
  return hands.reduce((s, h) => s + Object.values(h.extended).filter(Boolean).length, 0);
}

export class Quiz {
  constructor() {
    this.on = false;
    this.score = 0;
    this.question = "";
    this.answer = 0;
    this.state = "ask"; // ask | correct | wrong
    this.stateUntil = 0;
    this.counts = [];
    this.seen = null;
    this.holdValue = null;
    this.holdSince = 0;
    this.locked = null;
    this.wrongValue = null;
  }

  get level() {
    return Math.min(MAX_LEVEL, 1 + Math.floor(this.score / 2));
  }

  toggle(now) {
    this.on = !this.on;
    if (this.on) {
      this.score = 0;
      this.next(now);
    }
  }

  next() {
    const prev = this.question;
    for (let i = 0; i < 20; i++) { // unikamy powtórzenia tego samego pytania
      [this.question, this.answer] = makeQuestion(this.level);
      if (this.question !== prev) break;
    }
    this.state = "ask";
    this.holdValue = null;
    this.locked = null;
    this.counts = [];
  }

  holdProgress(now) {
    if (this.state !== "ask" || !this.seen || this.seen === this.locked) return 0;
    return Math.min(1, (now - this.holdSince) / HOLD_TIME);
  }

  /** Co klatkę; zwraca "correct" / "wrong" w chwili oceny odpowiedzi. */
  update(now, hands) {
    if (!this.on) return null;
    this.counts.push(countFingers(hands));
    if (this.counts.length > 8) this.counts.shift();
    // Wygładzanie: najczęstsza liczba palców z ostatnich klatek
    const visible = this.counts.filter((c) => c !== null);
    if (visible.length >= this.counts.length / 2) {
      const freq = new Map();
      for (const c of visible) freq.set(c, (freq.get(c) || 0) + 1);
      this.seen = [...freq.entries()].sort((a, b) => b[1] - a[1])[0][0];
    } else {
      this.seen = null;
    }

    if (this.state === "correct") {
      if (now >= this.stateUntil) this.next(now);
      return null;
    }
    if (this.state === "wrong" && now >= this.stateUntil) this.state = "ask";

    if (!this.seen) {
      this.holdValue = null;
      return null;
    }
    if (this.seen !== this.holdValue) {
      this.holdValue = this.seen;
      this.holdSince = now;
    }
    if (this.locked !== null && this.seen !== this.locked) this.locked = null;
    if (this.locked !== null || this.state !== "ask") return null;
    if (now - this.holdSince >= HOLD_TIME) {
      if (this.seen === this.answer) {
        this.score++;
        this.state = "correct";
        this.stateUntil = now + SUCCESS_TIME;
        return "correct";
      }
      this.state = "wrong";
      this.stateUntil = now + WRONG_TIME;
      this.locked = this.wrongValue = this.seen;
      return "wrong";
    }
    return null;
  }
}

/**
 * Panel z pytaniem i przełącznik QUIZ (lewy górny róg, pod girlandą).
 * Ekran poziomy: panel na środku u góry. Ekran pionowy (telefon): panel na całą
 * szerokość, pod przyciskiem i logo (`topLimit` – dolna krawędź logo).
 */
export class QuizView {
  constructor(w, h, ui, topLimit = 0) {
    const s = (this.s = ui);
    const portrait = (this.portrait = w < h);
    const garlandBottom = 0.07 * Math.min(h, w) + 6 * s;
    this.btn = { x: 12 * s, y: Math.max(58 * s, garlandBottom), w: 150 * s, h: 34 * s };
    const pw = portrait ? w - 24 * s : Math.min(w * 0.56, 520 * s);
    const py = portrait ? Math.max(this.btn.y + this.btn.h, topLimit) + 10 * s : this.btn.y;
    this.panel = { x: (w - pw) / 2, y: py, w: pw, h: 128 * s };
  }

  hitButton(x, y) {
    const b = this.btn;
    return x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h;
  }

  center() {
    const p = this.panel;
    return [p.x + p.w / 2, p.y + p.h / 2];
  }

  drawButton(ctx, on) {
    const { s, btn: b } = this;
    ctx.save();
    roundRectPath(ctx, b.x, b.y, b.w, b.h, b.h / 2);
    ctx.fillStyle = on ? "rgba(185,35,40,0.8)" : "rgba(20,24,30,0.8)";
    ctx.fill();
    ctx.lineWidth = Math.max(1, 1.5 * s);
    ctx.strokeStyle = on ? `rgb(${GOLD})` : "rgb(110,115,120)";
    ctx.stroke();
    ctx.fillStyle = "white";
    ctx.font = `600 ${14 * s}px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // na telefonie nie ma klawiatury – bez podpowiedzi „[m]”
    ctx.fillText(`QUIZ: ${on ? "wł." : "wył."}${this.portrait ? "" : "  [m]"}`, b.x + b.w / 2, b.y + b.h / 2 + 1);
    ctx.restore();
  }

  drawPanel(ctx, quiz, now) {
    if (!quiz.on) return;
    const { s, panel: p } = this;
    const cx = p.x + p.w / 2;
    ctx.save();
    roundRectPath(ctx, p.x, p.y, p.w, p.h, 18 * s);
    ctx.fillStyle = { ask: "rgba(20,24,30,0.78)", correct: "rgba(40,120,50,0.82)", wrong: "rgba(150,40,40,0.82)" }[quiz.state];
    ctx.fill();
    ctx.lineWidth = Math.max(1, 2 * s);
    ctx.strokeStyle = `rgb(${GOLD})`;
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = `rgb(${GOLD})`;
    const header = `QUIZ  ·  poziom ${quiz.level}  ·  wynik ${quiz.score}`;
    ctx.font = `600 ${fitFont(ctx, header, 600, 13 * s, p.w - 24 * s)}px ${FONT}`;
    ctx.fillText(header, cx, p.y + 18 * s);

    const q = quiz.state === "correct" ? `${quiz.question} = ${quiz.answer}` : `${quiz.question} = ?`;
    const size = fitFont(ctx, q, 600, 46 * s, p.w - 40 * s);
    ctx.font = `600 ${size}px ${FONT}`;
    ctx.fillStyle = `rgb(${WHITE})`;
    ctx.fillText(q, cx, p.y + 60 * s);

    let line;
    if (quiz.state === "correct") line = "Brawo! Świetna odpowiedź!";
    else if (quiz.state === "wrong") line = `Pokazujesz ${quiz.wrongValue} – spróbuj jeszcze raz`;
    else line = `Pokaż wynik na palcach   ·   widzę: ${quiz.seen || "–"}`;
    ctx.font = `400 ${fitFont(ctx, line, 400, 15 * s, p.w - 24 * s)}px ${FONT}`;
    ctx.fillStyle = "rgb(235,235,235)";
    ctx.fillText(line, cx, p.y + 98 * s);

    const prog = quiz.holdProgress(now);
    if (prog > 0) { // pasek „trzymania” odpowiedzi
      const bx0 = p.x + 24 * s, bx1 = p.x + p.w - 24 * s, by = p.y + p.h - 10 * s;
      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(2, 4 * s);
      ctx.strokeStyle = "rgb(60,65,70)";
      ctx.beginPath(); ctx.moveTo(bx0, by); ctx.lineTo(bx1, by); ctx.stroke();
      ctx.strokeStyle = `rgb(${GOLD})`;
      ctx.beginPath(); ctx.moveTo(bx0, by); ctx.lineTo(bx0 + (bx1 - bx0) * prog, by); ctx.stroke();
    }
    ctx.restore();
  }
}
