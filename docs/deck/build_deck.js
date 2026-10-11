// Builds docs/presentation.pptx (7 slides) from docs/results/results.json, docs/figures, docs/screenshots.
//   cd docs/deck && npm install && node build_deck.js && python ../../scripts/deck_to_pdf.py
// Candidate details come from docs/deck/candidate.json (edit it) or env TRIDENT_NAME / TRIDENT_COLLEGE / TRIDENT_REPO.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/results/results.json"), "utf8"));
const candFile = path.join(__dirname, "candidate.json");
const CAND = fs.existsSync(candFile) ? JSON.parse(fs.readFileSync(candFile, "utf8")) : {};
const NAME = process.env.TRIDENT_NAME || CAND.name || "[Your Full Name]";
const COLLEGE = process.env.TRIDENT_COLLEGE || CAND.college || "Indian Institute of Technology Madras";
const REPO = process.env.TRIDENT_REPO || CAND.repo || "github.com/<you>/iitm-<your-name>-hackathon";

const C = { ink: "0D1626", ink2: "18233A", accent: "B4540B", accentLt: "F6E7DA", muted: "556275", line: "D6DCE5", bg: "F4F6F9",
  white: "FFFFFF", pos: "0D7A5C", neg: "B3261E", blue: "1F4F8F", warn: "9A6700", posLt: "E3F3ED", negLt: "FBE9E7", warnLt: "FBF1DC" };
const FONT = "Calibri";
const icon = (n, c = "amber") => path.join(__dirname, "icons", `${n}_${c}.png`);
const img = (p) => path.join(ROOT, p);

// ---- numbers from results.json
const H = R.headline;
const lat = R.latency;
const s9 = R.stress.credit9;
const sent = R.nlp.sentiment[H.sentiment_eval_set];
const bt = R.backtest_real || R.backtest_synthetic;
const btReal = !!R.backtest_real;
const rob = R.robustness;
const svb = R.replay_svb;
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const f2 = (x) => Number(x).toFixed(2);
const sentProv = H.sentiment_eval_set === "gold_v0" ? "authored 180-headline set (optimistic)" : H.sentiment_eval_set.replace(/_/g, " ");
const mktProv = btReal ? "real prices + headlines" : "synthetic known-answer market";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = NAME;
pres.title = "TRIDENT - AI/NLP Risk Engine";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

pres.defineSlideMaster({
  title: "DARK", background: { color: C.ink },
  objects: [{ text: { text: "TRIDENT · S&P Global & Crisil Campus Hackathon 2026", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontFace: FONT, fontSize: 10, color: "8C98AB" } } }],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: "8C98AB", align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.white },
  objects: [{ text: { text: "TRIDENT · S&P Global & Crisil Campus Hackathon 2026", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted } } }],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: "right" },
});

function txt(slide, text, o) { slide.addText(text, { fontFace: FONT, isTextBox: true, margin: 0, color: C.ink, valign: "top", ...o }); }
function heading(slide, eyebrow, title) {
  txt(slide, eyebrow, { x: 0.6, y: 0.32, w: 9, h: 0.3, fontSize: 12, bold: true, color: C.accent, charSpacing: 2 });
  txt(slide, title, { x: 0.6, y: 0.62, w: 12.1, h: 0.75, fontSize: 30, bold: true, color: C.ink });
}
function iconCircle(slide, name, x, y, d = 0.6, fill = C.accentLt, col = "amber") {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  slide.addImage({ path: icon(name, col), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56 });
}
function card(slide, x, y, w, h, fill = C.bg) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
}
function stat(slide, x, y, w, big, label, sub, color = C.ink) {
  card(slide, x, y, w, 1.45);
  txt(slide, big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.62, fontSize: 32, bold: true, color });
  txt(slide, label, { x: x + 0.2, y: y + 0.74, w: w - 0.4, h: 0.3, fontSize: 14, bold: true });
  txt(slide, sub, { x: x + 0.2, y: y + 1.04, w: w - 0.4, h: 0.3, fontSize: 11, color: C.muted });
}

// =====================================================================================  1 TITLE
{
  const s = pres.addSlide({ masterName: "DARK" });
  txt(s, "S&P GLOBAL & CRISIL CAMPUS HACKATHON 2026 · CASE STUDY", { x: 0.6, y: 0.7, w: 7.4, h: 0.3, fontSize: 12, bold: true, color: "F0A347", charSpacing: 2 });
  txt(s, [{ text: "TRIDENT", options: { color: C.white } }, { text: ".", options: { color: "F0A347" } }], { x: 0.6, y: 1.25, w: 7.4, h: 1.3, fontSize: 80, bold: true });
  txt(s, "A real-time AI/NLP risk engine with a tactical index rebalancer and a wholesale-banking stress tester", { x: 0.6, y: 2.65, w: 7.0, h: 1.0, fontSize: 22, color: C.white });
  txt(s, "One engine · two strategies · three signals", { x: 0.6, y: 3.8, w: 7.0, h: 0.4, fontSize: 16, color: "F0A347", italic: true });
  txt(s, NAME, { x: 0.6, y: 4.85, w: 7.0, h: 0.4, fontSize: 20, bold: true, color: C.white });
  txt(s, COLLEGE, { x: 0.6, y: 5.25, w: 7.0, h: 0.35, fontSize: 16, color: "C9D1DC" });
  txt(s, REPO, { x: 0.6, y: 5.65, w: 7.0, h: 0.35, fontSize: 13, color: "8C98AB" });
  // signal card
  const x = 8.3, y = 0.95, w = 4.45;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 5.5, rectRadius: 0.1, fill: { color: C.ink2 }, line: { color: "2A3650" } });
  txt(s, "LIVE SIGNAL · SVB REPLAY, 8 MAR 2023", { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: 0.3, fontSize: 11, bold: true, color: "8C98AB", charSpacing: 1 });
  txt(s, "“SVB Financial announces emergency capital raise after $1.8 billion loss on bond portfolio sale”", { x: x + 0.3, y: y + 0.6, w: w - 0.6, h: 0.9, fontSize: 14, color: C.white, italic: true });
  const rows = [["Entity", "SIVB · SVB Financial · alias 0.94"], ["Event", "Credit Event · liquidity"], ["Sentiment", "−0.62"], ["Impact", "9 / 10"], ["Gate", "TRIGGERED · 2 sources"], ["Engine", `${f2(lat.engine_ms.p50)} ms p50`]];
  rows.forEach(([k, v], i) => {
    const yy = y + 1.65 + i * 0.6;
    txt(s, k, { x: x + 0.3, y: yy, w: 1.3, h: 0.35, fontSize: 13, color: "8C98AB" });
    txt(s, v, { x: x + 1.55, y: yy, w: w - 1.85, h: 0.35, fontSize: 14, bold: true, color: k === "Gate" ? "FF8A7A" : C.white });
  });
  // sentiment bar and impact dial
  const by = y + 1.65 + 2 * 0.6 + 0.38;
  s.addShape(pres.shapes.RECTANGLE, { x: x + 1.55, y: by, w: 2.5, h: 0.06, fill: { color: "2A3650" }, line: { color: "2A3650" } });
  s.addShape(pres.shapes.RECTANGLE, { x: x + 1.55 + 2.5 * 0.19, y: by - 0.05, w: 0.05, h: 0.16, fill: { color: "FF8A7A" }, line: { color: "FF8A7A" } });
  for (let k = 0; k < 10; k++) {
    s.addShape(pres.shapes.RECTANGLE, { x: x + 1.55 + k * 0.25, y: y + 1.65 + 3 * 0.6 + 0.38, w: 0.2, h: 0.08, fill: { color: k < 9 ? "FF8A7A" : "2A3650" }, line: { color: k < 9 ? "FF8A7A" : "2A3650" } });
  }
  s.addNotes(`Open with the story: in March 2023 a bank failed in under 48 hours, and the run spread through text first.
TRIDENT reads that text and turns it into a structured, audited risk signal in about ${f2(lat.engine_ms.p50)} milliseconds.
One engine, two strategies, three signals: sentiment, event class and impact. The card on the right is a real signal from our SVB replay.`);
}

// =====================================================================================  2 PROBLEM & APPROACH
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  heading(s, "PROBLEM & APPROACH", "Risk arrives as text hours before it arrives as data");
  txt(s, "The problem", { x: 0.6, y: 1.6, w: 5.6, h: 0.4, fontSize: 20, bold: true });
  txt(s, [
    { text: "Market-moving news, filings and posts are unstructured; risk systems read structured feeds.", options: { bullet: true, breakLine: true } },
    { text: "Free sources lag (GDELT 15 min, NewsAPI free 24 h) and are full of duplicates, rumors and ambiguous names.", options: { bullet: true, breakLine: true } },
    { text: "A useful signal must be fast, calibrated, explainable and tied to a decision.", options: { bullet: true } },
  ], { x: 0.6, y: 2.05, w: 5.7, h: 1.9, fontSize: 15, paraSpaceAfter: 6, color: C.ink });
  // SVB timeline
  txt(s, "SVB, March 2023 (UTC)", { x: 0.6, y: 4.15, w: 5.6, h: 0.35, fontSize: 14, bold: true, color: C.muted });
  const tl = [["8 Mar 21:15", "Capital raise after $1.8bn bond loss"], ["9 Mar", "$42bn withdrawal requests in one day"], ["10 Mar 16:20", "Regulators close the bank"]];
  s.addShape(pres.shapes.LINE, { x: 0.85, y: 4.75, w: 5.0, h: 0, line: { color: C.line, width: 2 } });
  tl.forEach(([d, t], i) => {
    const x = 0.75 + i * 2.15;
    s.addShape(pres.shapes.OVAL, { x, y: 4.65, w: 0.2, h: 0.2, fill: { color: i === 2 ? C.neg : C.accent }, line: { color: C.white, width: 2 } });
    txt(s, d, { x, y: 4.98, w: 1.95, h: 0.3, fontSize: 13, bold: true, color: i === 2 ? C.neg : C.accent });
    txt(s, t, { x, y: 5.28, w: 1.95, h: 0.8, fontSize: 12, color: C.ink });
  });
  // approach
  txt(s, "Our approach", { x: 6.9, y: 1.6, w: 5.8, h: 0.4, fontSize: 20, bold: true });
  const steps = [["resolve", "Resolve", "Map every item to a company, a sector or the market; reject ambiguous names with a logged reason."],
    ["score", "Score", "Target-aware sentiment, event class with abstain, impact calibrated to realized moves; confidence and evidence on each."],
    ["act", "Act", "Module A tilts a 15-stock index (minutes to days). Module B stress-tests a wholesale book and reports CET1 (capital)."]];
  steps.forEach(([ic, h, b], i) => {
    const y = 2.1 + i * 1.3;
    iconCircle(s, ic, 6.9, y, 0.7);
    txt(s, h, { x: 7.8, y: y - 0.02, w: 4.9, h: 0.35, fontSize: 17, bold: true });
    txt(s, b, { x: 7.8, y: y + 0.34, w: 4.9, h: 0.85, fontSize: 14, color: C.muted });
  });
  card(s, 6.9, 6.0, 5.85, 0.65, C.accentLt);
  txt(s, "Both modules built: A consumes sentiment, B consumes event + impact, so all three fields are exercised.", { x: 7.1, y: 6.05, w: 5.5, h: 0.55, fontSize: 13, color: C.ink, valign: "middle" });
  s.addNotes("Two decision horizons from the same signals: tactical position sizing and strategic capital adequacy. We built both modules because they consume different fields of the signal.");
}

// =====================================================================================  3 SYSTEM DESIGN
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  heading(s, "SYSTEM DESIGN", "Simple to run on a laptop, with a clear path to bank scale");
  s.addImage({ path: img("docs/architecture.png"), x: 0.6, y: 1.55, w: 8.75, h: 4.45 });
  const pts = [["clock", "Replay or live", "Deterministic replay of SVB-2023 and COVID-2020 means the demo needs no keys and no network; live mode polls RSS, SEC EDGAR, GDELT, NewsAPI and Reddit."],
    ["layers", "Pluggable bus", "In-process asyncio queue by default; Redis Streams with one setting; Kafka partitioned by ticker behind the same interface."],
    ["git", "Lineage on every signal", "Model versions, data hash and replay id travel with each signal; a champion/challenger registry logs every model change."]];
  pts.forEach(([ic, h, b], i) => {
    const y = 1.6 + i * 1.5;
    iconCircle(s, ic, 9.65, y, 0.55);
    txt(s, h, { x: 10.35, y: y - 0.02, w: 2.45, h: 0.35, fontSize: 15, bold: true });
    txt(s, b, { x: 10.35, y: y + 0.32, w: 2.45, h: 1.1, fontSize: 11.5, color: C.muted });
  });
  txt(s, `Measured: engine ${f2(lat.engine_ms.p50)} ms p50 / ${f2(lat.engine_ms.p95)} ms p95 per document · Module A rebalance ${f2(lat.module_a_rebalance_ms.p50)} ms · Module B full revaluation ${f2(lat.module_b_full_revaluation_ms.p50)} ms · ${Math.round(lat.throughput_docs_per_s)} docs/s on one CPU process`,
    { x: 0.6, y: 6.2, w: 12.1, h: 0.4, fontSize: 13, color: C.ink, bold: true });
  s.addNotes("Walk the critical path in amber: sources, adapters with circuit breakers, dedup, the five engine stages, the bus, FastAPI, the two modules. Kafka is the right answer at bank scale and the wrong one for a judge's laptop, so the bus is an interface.");
}

// =====================================================================================  4 IMPLEMENTATION HIGHLIGHTS
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  heading(s, "IMPLEMENTATION HIGHLIGHTS", "Four design choices that separate signal from noise");
  const tiles = [
    ["target", "Target-aware sentiment", "Finance lexicon with Loughran-McDonald words, negation and delta phrases, blended with FinBERT exported to ONNX INT8. Scored per entity: “Microsoft beats Google” is + for MSFT, − for GOOGL.", "VADER scores “Profit plunges after default” as +0.44. TRIDENT: −0.91."],
    ["shield", "Events with an abstain class", "8 classes plus Other, weak labels from keywords, 8-K Item codes and GDELT CAMEO, distilled into DeBERTa. A champion/challenger registry only promotes a model that beats the incumbent.", "Synthetic-only challengers were rejected (F1 0.48 vs 0.65)."],
    ["chart", "Impact calibrated to realized moves", "An ordinal model maps severity, |sentiment|, novelty, burst velocity, source reliability and magnitude cues to deciles of |CAR[0,+1]| from a market-model event study.", `Out-of-sample Spearman ${f2(R.impact_synthetic.spearman_default_impact_vs_abs_car)} (5-seed mean ${f2(rob.impact_spearman_oos.mean)}).`],
    ["bank", "Module B, the way a bank measures it", "Vasicek PD migration, duration-convexity, Greeks and Black-Scholes, Basel IRB RWA and CET1, stressed-correlation VaR, reverse stress test. A two-source gate stops single headlines.", `Credit I=9: CET1 ${pct(s9.cet1_before)} → ${pct(s9.cet1_after)}.`],
  ];
  tiles.forEach(([ic, h, b, k], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.95, 2.35);
    iconCircle(s, ic, x + 0.25, y + 0.25, 0.6, C.white);
    txt(s, h, { x: x + 1.05, y: y + 0.3, w: 4.7, h: 0.45, fontSize: 17, bold: true });
    txt(s, b, { x: x + 0.3, y: y + 0.95, w: 5.4, h: 0.95, fontSize: 12.5, color: C.ink });
    txt(s, k, { x: x + 0.3, y: y + 1.9, w: 5.4, h: 0.35, fontSize: 12.5, bold: true, color: C.accent });
  });
  txt(s, "Python 3.11 · FastAPI · Pydantic · scikit-learn · SciPy · cvxpy · transformers + onnxruntime · spaCy · ECharts · 55 tests incl. Hypothesis property tests", { x: 0.6, y: 6.68, w: 12.1, h: 0.3, fontSize: 11, color: C.muted });
  s.addNotes("Each tile is a tech choice and the reason for it. Emphasize the abstain class and the gate: an honest Other beats a confident wrong label that fires a capital alert.");
}

// =====================================================================================  5 KEY RESULTS
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  heading(s, "KEY RESULTS", "Measured against baselines, with the data source named");
  stat(s, 0.6, 1.55, 2.9, `${f2(lat.engine_ms.p50)} ms`, "Engine latency p50", `p95 ${f2(lat.engine_ms.p95)} ms · ${Math.round(lat.throughput_docs_per_s)} docs/s`, C.blue);
  stat(s, 3.67, 1.55, 2.9, `${f2(sent.trident_best)} vs ${f2(sent.vader_baseline)}`, "Sentiment F1 vs VADER", sentProv, C.accent);
  stat(s, 6.74, 1.55, 2.9, `${pct(s9.cet1_after)}`, "CET1 after Credit I=9", `from ${pct(s9.cet1_before)} · P&L ${s9.pnl.toFixed(0)}m`, C.neg);
  stat(s, 9.81, 1.55, 2.9, `${H.checklist_pass}`, "Behavioral tests passed", `entity precision ${f2(H.entity_precision)} · recall ${f2(H.entity_recall)}`, C.pos);
  // chart 1: sentiment baselines
  s.addChart(pres.charts.BAR, [{ name: "macro-F1", labels: ["Naive word count", "VADER", "TRIDENT"], values: [sent.naive_wordcount_baseline, sent.vader_baseline, sent.trident_best].map((v) => +v.toFixed(2)) }], {
    x: 0.6, y: 3.25, w: 3.9, h: 3.25, barDir: "col", chartColors: [C.line, C.muted, C.accent], showTitle: true, title: "Sentiment macro-F1", titleFontSize: 13, titleColor: C.ink, titleFontFace: FONT,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: C.ink, dataLabelFontFace: FONT, dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisHidden: true,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: C.muted, catAxisLabelFontSize: 11, catAxisLabelFontFace: FONT, showLegend: false,
  });
  // chart 2: CET1 by scenario
  const rows = R.stress.table.filter((t) => t.class === "preset" || t.impact === 9);
  const labels = rows.map((t) => t.scenario.replace(" (approximate moves)", "*").replace(", impact", "").replace(" I=9", " I=9"));
  s.addChart(pres.charts.BAR, [{ name: "CET1 after stress", labels, values: rows.map((t) => +(t.cet1_after * 100).toFixed(1)) }], {
    x: 4.7, y: 3.25, w: 4.8, h: 3.25, barDir: "bar", chartColors: [C.blue], showTitle: true, title: "CET1 ratio after stress, % (start 13.0)", titleFontSize: 13, titleColor: C.ink, titleFontFace: FONT,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: C.ink, dataLabelFontFace: FONT, dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 14, valAxisHidden: true,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelColor: C.muted, catAxisLabelFontSize: 10, catAxisLabelFontFace: FONT, showLegend: false, catAxisOrientation: "maxMin",
  });
  // Module A panel
  card(s, 9.7, 3.25, 3.0, 3.25);
  txt(s, "Module A, out of sample", { x: 9.9, y: 3.38, w: 2.7, h: 0.35, fontSize: 14, bold: true });
  txt(s, `${f2(bt.strategy.information_ratio)}`, { x: 9.9, y: 3.75, w: 2.7, h: 0.6, fontSize: 32, bold: true, color: C.accent });
  txt(s, `net information ratio (gross ${f2(bt.strategy_gross.information_ratio)}), ${bt.period[0].slice(0, 4)}-${bt.period[1].slice(0, 4)}, parameters chosen on earlier years`, { x: 9.9, y: 4.35, w: 2.65, h: 0.85, fontSize: 11.5, color: C.ink });
  txt(s, `Across 5 seeds: net IR ${f2(rob.net_ir_oos.mean)} ± ${f2(rob.net_ir_oos.sd)}, IC ${rob.ic.mean.toFixed(3)}. The signal is real; costs decide whether it pays.`, { x: 9.9, y: 5.2, w: 2.65, h: 1.15, fontSize: 11.5, color: C.muted });
  txt(s, `Data: sentiment and entities on ${sentProv}; impact and Module A on ${mktProv}; Module B on the synthetic wholesale book. Regenerate on real data with scripts/run_all_results.py. *approximate historical moves.`,
    { x: 0.6, y: 6.62, w: 12.1, h: 0.32, fontSize: 10, color: C.muted });
  s.addNotes(`Name one limitation honestly: the sentiment set is authored and optimistic, and the impact and Module A numbers come from a known-answer synthetic market that validates the pipeline. Real-data reruns are one command. Credibility is the asset.`);
}

// =====================================================================================  6 DOMAIN IMPACT
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  heading(s, "DOMAIN IMPACT", "From headline to capital decision in one replay");
  s.addImage({ path: img("docs/screenshots/03_module_b.png"), x: 0.6, y: 1.55, w: 7.3, h: 4.56, sizing: { type: "cover", w: 7.3, h: 4.56 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.55, w: 7.3, h: 4.56, fill: { type: "none" }, line: { color: C.line, width: 1 } });
  const fs1 = svb.first_stress;
  const tl = [["21:15", "First headline, impact 8, one source", "HELD", C.warn, C.warnLt], ["21:40", "Second wire confirms the story", "TRIGGERED", C.neg, C.negLt],
    ["→", `P&L ${fs1.pnl.toFixed(0)}m · CET1 ${pct(fs1.cet1_before)} → ${pct(fs1.cet1_after)}`, "CAPITAL VIEW", C.blue, "E6EEF8"]];
  txt(s, "SVB replay, 8 March 2023 (UTC)", { x: 8.2, y: 1.55, w: 4.5, h: 0.35, fontSize: 15, bold: true });
  tl.forEach(([t, d, tag, col, bg], i) => {
    const y = 2.0 + i * 0.78;
    txt(s, t, { x: 8.2, y, w: 0.7, h: 0.6, fontSize: 15, bold: true, color: col, valign: "middle" });
    txt(s, d, { x: 8.9, y, w: 2.55, h: 0.6, fontSize: 12.5, color: C.ink, valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 11.5, y: y + 0.13, w: 1.25, h: 0.34, rectRadius: 0.05, fill: { color: bg }, line: { color: bg } });
    txt(s, tag, { x: 11.5, y: y + 0.13, w: 1.25, h: 0.34, fontSize: 10, bold: true, color: col, align: "center", valign: "middle" });
  });
  const users = [["eye", "Credit surveillance", "Obligors flagged within minutes, ranked by calibrated impact, evidence attached."],
    ["chart", "Portfolio managers", "A cost-aware tilt with caps and bands, not headline chasing."],
    ["bank", "Risk & treasury", "Event-triggered CET1, VaR and reverse stress for the risk committee."]];
  users.forEach(([ic, h, b], i) => {
    const y = 4.4 + i * 0.82;
    iconCircle(s, ic, 8.2, y, 0.5);
    txt(s, h, { x: 8.85, y: y - 0.04, w: 3.9, h: 0.3, fontSize: 14, bold: true });
    txt(s, b, { x: 8.85, y: y + 0.26, w: 3.9, h: 0.45, fontSize: 11.5, color: C.muted });
  });
  s.addNotes("This is the screen to pause on in the demo: the status flips from monitoring to triggered. The first headline was held because only one source reported it; the trigger came on independent confirmation. Then the capital number a risk committee actually reads.");
}

// =====================================================================================  7 LIMITATIONS & NEXT STEPS
{
  const s = pres.addSlide({ masterName: "DARK" });
  txt(s, "LIMITATIONS & NEXT STEPS", { x: 0.6, y: 0.5, w: 9, h: 0.3, fontSize: 12, bold: true, color: "F0A347", charSpacing: 2 });
  txt(s, "What we know is weak, and what we would build next", { x: 0.6, y: 0.85, w: 12.1, h: 0.7, fontSize: 30, bold: true, color: C.white });
  const lim = [
    "Free sources lag by minutes to a day; engine speed is not end-to-end speed.",
    "Sentiment and entity scores use an authored set; real PhraseBank and labeled headlines are the numbers to report.",
    `Template synthetic text is easy to detect (AUC ${f2(R.adversarial.auc)}); it is capped at 30% of training and never used for evaluation.`,
    "Impact and Module A were validated on a known-answer market; real-price results depend on data we fetch locally.",
    "IRB without maturity adjustment; Greeks are linear except the options book.",
  ];
  const nxt = [
    "Fine-tune FinBERT and DeBERTa on real labels; serve ONNX INT8.",
    "Indian market: NSE universe, Indian regulatory disclosures, Hindi and regional-language news via GDELT.",
    "LLM-generated synthetic data that passes adversarial validation.",
    "Sector contagion through a learned correlation network; online recalibration of impact.",
    "Rating-surveillance view: map signals to obligors in a credit portfolio.",
  ];
  [["Limitations", lim, "FF8A7A"], ["Next steps", nxt, "7FD1B1"]].forEach(([h, items, col], k) => {
    const x = 0.6 + k * 6.25;
    txt(s, h, { x, y: 1.85, w: 5.8, h: 0.4, fontSize: 20, bold: true, color: col });
    txt(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })), { x, y: 2.35, w: 5.85, h: 3.6, fontSize: 14, color: C.white, paraSpaceAfter: 8 });
  });
  txt(s, "TRIDENT: text in, capital decision out, every step auditable.", { x: 0.6, y: 6.25, w: 12.1, h: 0.45, fontSize: 20, bold: true, color: "F0A347", italic: true });
  s.addNotes("State the limitations before the jury asks. Close with the one-line summary and invite questions.");
}

const out = path.join(ROOT, "docs", "presentation.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
