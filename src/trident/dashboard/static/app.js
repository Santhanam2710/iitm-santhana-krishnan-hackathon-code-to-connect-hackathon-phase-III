/* TRIDENT console client: WebSocket state + ECharts views. No build step. */
(function () {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (v, d = 1) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
  const pct = (v, d = 2) => (v * 100).toFixed(d) + '%';
  const STRESS = ['Geopolitical', 'Macroeconomic', 'Credit Event'];

  const S = { signals: [], decisions: [], weights: [], stress: null, stressHist: [], status: null, selected: null, filter: 'all', tickers: [] };
  const charts = {};
  let dirty = new Set();

  // ---------------------------------------------------------------- tabs
  let activeTab = 'feed';
  $$('nav.tabs button').forEach((b) => b.addEventListener('click', () => {
    activeTab = b.dataset.tab;
    $$('nav.tabs button').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    $$('.tab').forEach((t) => (t.hidden = t.id !== 'tab-' + activeTab));
    dirty.add(activeTab); Object.values(charts).forEach((c) => c && c.resize());
    if (activeTab === 'sys') loadSystem();
    schedule();
  }));

  // ---------------------------------------------------------------- api
  async function api(path, opts) {
    const r = await fetch(path, opts ? { ...opts, headers: { 'Content-Type': 'application/json' } } : undefined);
    if (!r.ok) throw new Error(path + ' ' + r.status);
    return r.json();
  }

  // ---------------------------------------------------------------- websocket
  function connect() {
    const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === 'hello') {
        S.status = m.data.status; S.signals = (m.data.recent || []).reverse();
        if (m.data.weights) S.weights = [m.data.weights];
        api('/api/stress/decisions?limit=60').then((d) => { S.decisions = d.slice().reverse(); dirty.add('gate'); dirty.add('feed'); schedule(); }).catch(() => {});
        api('/api/stress/history').then((h) => { S.stressHist = h.slice().reverse(); dirty.add('modb'); schedule(); }).catch(() => {});
        api('/api/rebalancer/history?limit=600').then((h) => { if (h.length) { S.weights = h; dirty.add('moda'); schedule(); } }).catch(() => {});
        if (m.data.stress) S.stress = m.data.stress;
        dirty = new Set(['feed', 'moda', 'modb', 'hdr']);
      } else if (m.type === 'signal') { S.signals.unshift(m.data); if (S.signals.length > 400) S.signals.pop(); dirty.add('feed'); dirty.add('hdr'); }
      else if (m.type === 'decision') { S.decisions.unshift(m.data); if (S.decisions.length > 60) S.decisions.pop(); dirty.add('gate'); }
      else if (m.type === 'weights') { S.weights.push(m.data); if (S.weights.length > 600) S.weights.shift(); dirty.add('moda'); }
      else if (m.type === 'stress') { S.stress = m.data; S.stressHist.unshift(m.data); dirty.add('modb'); }
      schedule();
    };
    ws.onclose = () => { setMode('disconnected'); setTimeout(connect, 1500); };
    ws.onopen = () => setInterval(() => ws.readyState === 1 && ws.send('ping'), 20000);
  }

  let raf = 0;
  function schedule() { if (!raf) raf = requestAnimationFrame(render); }
  function render() {
    raf = 0;
    if (dirty.has('hdr')) renderHeader();
    if (dirty.has('gate') || dirty.has('feed')) renderGate();
    if (activeTab === 'feed' && dirty.has('feed')) renderFeed();
    if (activeTab === 'moda' && dirty.has('moda')) renderModA();
    if (activeTab === 'modb' && dirty.has('modb')) renderModB();
    dirty.clear();
  }

  function setMode(txt, cls) { const c = $('#modeChip'); c.className = 'hstat ' + (cls || ''); c.querySelector('b').textContent = txt; }

  async function pollStatus() {
    try {
      S.status = await api('/api/status');
      renderHeader();
    } catch (e) { /* server restarting */ }
    setTimeout(pollStatus, 2000);
  }

  function renderHeader() {
    const st = S.status; if (!st) return;
    setMode(st.mode === 'replay' ? 'REPLAY · ' + (st.replay.name || '') : st.mode.toUpperCase(), st.mode);
    $('#simClock').textContent = st.sim_time ? st.sim_time.replace('T', ' ').slice(0, 16) + ' UTC' : '—';
    $('#replayProg').textContent = st.replay && st.replay.total ? `${st.replay.position}/${st.replay.total}${st.replay.running ? '' : ' ✓'}` : '—';
    $('#latency').textContent = `${st.latency_ms.p50.toFixed(1)} / ${st.latency_ms.p95.toFixed(1)} ms`;
    $('#sigCount').textContent = st.signals_stored;
    const sel = $('#replaySel');
    if (!sel.options.length && st.replays) { sel.innerHTML = st.replays.map((r) => `<option ${r === st.replay.name ? 'selected' : ''}>${esc(r)}</option>`).join(''); }
  }

  // ---------------------------------------------------------------- feed
  function polarity(s) { return s >= 0.1 ? 'pos' : (s <= -0.1 ? 'neg' : ''); }
  function filtered() {
    if (S.filter === 'stress') return S.signals.filter((s) => STRESS.includes(s.event_classification.label));
    if (S.filter === 'index') return S.signals.filter((s) => s.entity.in_index);
    return S.signals;
  }
  $$('[data-filter]').forEach((b) => b.addEventListener('click', () => {
    S.filter = b.dataset.filter; $$('[data-filter]').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); renderFeed();
  }));

  function renderFeed() {
    const list = filtered().slice(0, 150);
    $('#feed').innerHTML = list.map((s) => {
      const p = polarity(s.sentiment_score);
      const syn = s.lineage.is_synthetic ? '<span class="chip syn">synthetic</span>' : '';
      const flags = s.flags.filter((f) => !f.startsWith('synthetic')).map((f) => `<span class="chip">${esc(f)}</span>`).join('');
      return `<div class="card ${p} ${S.selected === s.signal_id ? 'sel' : ''}" data-id="${s.signal_id}">
        <div class="meta"><span class="chip tk">${esc(s.entity.ticker)}</span><span class="chip ev">${esc(s.event_classification.label)}</span>
        <span class="chip ${p}">s ${fmt(s.sentiment_score, 2)}</span>${syn}${flags}
        <span class="chip lat">${esc(s.source.name)} · ${s.latency_ms.engine_total.toFixed(1)} ms</span></div>
        <div class="imp ${s.impact_score > 7 ? 'hot' : ''}">${s.impact_score}<span class="muted small">/10</span></div>
        <div class="txt">${esc(s.text)}</div><div class="muted small mono">${s.ts_published.replace('T', ' ').slice(0, 16)}</div></div>`;
    }).join('') || '<div class="empty">Waiting for signals…</div>';
    $$('#feed .card').forEach((c) => c.addEventListener('click', () => { S.selected = c.dataset.id; renderFeed(); renderDetail(S.signals.find((x) => x.signal_id === c.dataset.id)); }));
    const n = S.signals.length;
    const stressN = S.signals.filter((s) => STRESS.includes(s.event_classification.label) && s.impact_score > 7).length;
    const trig = S.decisions.filter((d) => d.state === 'triggered').length, held = S.decisions.filter((d) => d.state === 'held').length;
    const neg = S.signals.filter((s) => s.sentiment_score < -0.1).length;
    $('#feedKpis').innerHTML = kpis([
      ['Signals in view', n, 'latest 400'], ['High-impact stress signals', stressN, 'class ∈ stress set, I > 7'],
      ['Gate: triggered / held', `${trig} / ${held}`, 'two-source rule'], ['Negative share', n ? pct(neg / n, 0) : '—', 'sentiment < −0.1'],
    ]);
    if (!S.selected && list[0]) renderDetail(list[0]);
  }

  function highlight(text, ev) {
    const spans = (ev || []).slice().sort((a, b) => a.start - b.start);
    let out = '', i = 0;
    for (const e of spans) { if (e.start < i) continue; out += esc(text.slice(i, e.start)) + `<span class="hl ${e.polarity}">${esc(text.slice(e.start, e.end))}</span>`; i = e.end; }
    return out + esc(text.slice(i));
  }

  function signalDetail(s) {
    const ev = s.event_classification;
    const probs = Object.entries(ev.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
    return `<div class="textview">${highlight(s.text, s.evidence)}</div>
      <div class="grid2" style="margin-top:12px">
        <div>
          <div class="ph"><h3>Sentiment ${fmt(s.sentiment_score, 2)}</h3><span class="muted small">conf ${s.sentiment_confidence.toFixed(2)}</span></div>
          <div class="gauge"><i style="left:${(s.sentiment_score + 1) * 50}%"></i></div>
          <div class="ph" style="margin-top:14px"><h3>Impact ${s.impact_score}/10</h3></div>
          <div class="dial">${Array.from({ length: 10 }, (_, i) => `<span class="${i < s.impact_score ? 'on' + (s.impact_score > 7 ? ' hot' : '') : ''}"></span>`).join('')}</div>
        </div>
        <div><div class="ph"><h3>${esc(ev.label)}${ev.subtype ? ' · ' + esc(ev.subtype) : ''}</h3><span class="muted small">conf ${ev.confidence.toFixed(2)}</span></div>
          ${probs.map(([k, v], i) => `<div class="pbar"><span>${esc(k)}</span><span class="b"><i class="${i === 0 ? 'top' : ''}" style="width:${(v * 100).toFixed(1)}%"></i></span><span class="mono">${(v * 100).toFixed(0)}%</span></div>`).join('')}
        </div>
      </div>
      <div class="callout">${esc(s.rationale.text)}</div>
      <dl class="dl" style="margin-top:12px">
        <dt>Entity</dt><dd>${esc(s.entity.ticker)} · ${esc(s.entity.name)} · ${esc(s.entity.match)} (${s.entity.confidence})</dd>
        <dt>Source</dt><dd>${esc(s.source.type)} · ${esc(s.source.name)} · reliability ${s.source.reliability} · novelty ${s.source.novelty}</dd>
        <dt>Latency</dt><dd>engine ${s.latency_ms.engine_total.toFixed(2)} ms · sentiment ${s.latency_ms.sentiment.toFixed(2)} ms · source lag ${s.latency_ms.source_lag_s ?? 0} s</dd>
        <dt>Models</dt><dd>${esc(Object.entries(s.lineage.models).map(([k, v]) => k + '=' + v).join(' · '))}</dd>
        <dt>Lineage</dt><dd>engine ${esc(s.lineage.engine_version)} · doc ${esc(s.lineage.doc_id)} · hash ${esc(s.lineage.doc_hash)}${s.lineage.replay_id ? ' · replay ' + esc(s.lineage.replay_id) : ''}</dd>
      </dl>
      <details style="margin-top:10px"><summary class="muted">Signal JSON</summary><pre>${esc(JSON.stringify(s, null, 2))}</pre></details>`;
  }
  function renderDetail(s) { if (s) $('#detail').innerHTML = signalDetail(s); }

  function renderGate() {
    const d = S.decisions.filter((x) => x.state !== 'monitored').slice(0, 25);
    if (!d.length) return;
    $('#gateBar').innerHTML = d.map((x) => `<span class="gate ${x.state}" title="${esc(x.reason)}">${x.ts.slice(5, 16).replace('T', ' ')} ${x.state.toUpperCase()} ${esc(x.label)} I=${x.impact} ${esc(x.ticker)}</span>`).join('');
  }

  function kpis(rows) { return rows.map((k) => `<div class="kpi"><span class="k">${k[0]}</span><span class="v">${k[1]}</span><span class="s">${k[2] || ''}</span></div>`).join(''); }

  // ---------------------------------------------------------------- module A
  function chart(id) { if (!charts[id]) { charts[id] = echarts.init(document.getElementById(id), null, { renderer: 'canvas' }); charts[id].setOption({ textStyle: { fontFamily: 'IBM Plex Sans, system-ui, sans-serif' } }); } return charts[id]; }
  window.addEventListener('resize', () => Object.values(charts).forEach((c) => c.resize()));
  const axisStyle = () => ({ axisLine: { lineStyle: { color: css('--line') } }, axisLabel: { color: css('--muted'), fontFamily: 'JetBrains Mono', fontSize: 10 }, splitLine: { lineStyle: { color: css('--line') } } });
  const palette = () => ['#1f4f8f', '#b4540b', '#0d7a5c', '#7a4fb3', '#b3261e', '#3a8fb7', '#9a6700', '#5b6b7f', '#c2185b', '#2e7d32', '#6d4c41', '#00838f', '#8e24aa', '#ef6c00', '#455a64'];

  function renderModA() {
    const W = S.weights; if (!W.length) return;
    const last = W[W.length - 1];
    const tickers = Object.keys(last.weights);
    S.tickers = tickers;
    const times = W.map((r) => r.ts.slice(5, 16).replace('T', ' '));
    const heat = [];
    W.forEach((r, x) => tickers.forEach((t, y) => heat.push([x, y, +((r.weights[t] - r.benchmark[t]) * 100).toFixed(3)])));
    const mx = Math.max(0.5, ...heat.map((h) => Math.abs(h[2])));
    chart('aHeat').setOption({
      animation: false, grid: { left: 54, right: 10, top: 10, bottom: 46 },
      tooltip: { formatter: (p) => `${tickers[p.value[1]]} · ${times[p.value[0]]}<br>active ${p.value[2].toFixed(2)} pp` },
      xAxis: { type: 'category', data: times, ...axisStyle(), axisLabel: { ...axisStyle().axisLabel, hideOverlap: true } },
      yAxis: { type: 'category', data: tickers, ...axisStyle() },
      visualMap: { min: -mx, max: mx, show: false, inRange: { color: [css('--neg'), css('--surface2'), css('--pos')] } },
      series: [{ type: 'heatmap', data: heat, progressive: 0 }],
    }, true);
    chart('aArea').setOption({
      animation: false, color: palette(), grid: { left: 44, right: 10, top: 30, bottom: 46 },
      tooltip: { trigger: 'axis', valueFormatter: (v) => v.toFixed(2) + '%' },
      legend: { type: 'scroll', top: 0, textStyle: { color: css('--muted'), fontSize: 10 } },
      xAxis: { type: 'category', data: times, boundaryGap: false, ...axisStyle(), axisLabel: { ...axisStyle().axisLabel, hideOverlap: true } },
      yAxis: { type: 'value', max: 100, ...axisStyle(), axisLabel: { ...axisStyle().axisLabel, formatter: '{value}%' } },
      series: tickers.map((t) => ({ name: t, type: 'line', stack: 'w', areaStyle: {}, symbol: 'none', lineStyle: { width: 0.5 }, data: W.map((r) => +(r.weights[t] * 100).toFixed(3)) })),
    }, true);
    const sc = last.scores;
    $('#aTable').innerHTML = `<thead><tr><th>Ticker</th><th class="n">Weight</th><th class="n">Benchmark</th><th class="n">Active pp</th><th class="n">S (total)</th><th class="n">own</th><th class="n">sector spill</th><th class="n">market</th></tr></thead><tbody>` +
      tickers.map((t) => { const a = (last.weights[t] - last.benchmark[t]) * 100; const s = sc[t];
        return `<tr><td>${t}</td><td class="n">${pct(last.weights[t])}</td><td class="n">${pct(last.benchmark[t])}</td><td class="n ${a >= 0 ? 'up' : 'dn'}">${fmt(a, 2)}</td><td class="n ${s.S >= 0 ? 'up' : 'dn'}">${fmt(s.S, 3)}</td><td class="n">${fmt(s.own, 3)}</td><td class="n">${fmt(s.spill, 3)}</td><td class="n">${fmt(s.market, 3)}</td></tr>`; }).join('') + '</tbody>';
    const supp = W.reduce((a, r) => a + r.suppressed, 0), cost = W.reduce((a, r) => a + r.cost_bps, 0);
    $('#aKpis').innerHTML = kpis([['Active share', pct(last.active_share, 1), '½ Σ |w − wᵇ|'], ['Last turnover', pct(last.turnover, 2), 'one-way'],
      ['Cost drag', cost.toFixed(1) + ' bps', 'linear + impact, cumulative'], ['Trades suppressed', supp, 'by no-trade band / min hold'], ['Method', last.method, 'A2 tilt or A4 optimizer']]);
  }

  function bindRange(id, fmtFn) { const el = $('#' + id), o = $('#' + id + 'O'); el.addEventListener('input', () => (o.textContent = fmtFn(+el.value))); }
  bindRange('pLam', (v) => v.toFixed(2)); bindRange('pMax', (v) => pct(v, 1)); bindRange('pBand', (v) => pct(v, 2)); bindRange('pTo', (v) => pct(v, 0));
  $('#pApply').addEventListener('click', async () => {
    const body = { lambda: +$('#pLam').value, w_max: +$('#pMax').value, band: +$('#pBand').value, turnover_max: +$('#pTo').value, method: $('#pMethod').value };
    const snap = await api('/api/rebalancer/params', { method: 'POST', body: JSON.stringify(body) });
    if (snap.last) { S.weights.push(snap.last); dirty.add('moda'); schedule(); }
  });

  // ---------------------------------------------------------------- module B
  function renderModB() {
    const r = S.stress;
    if (!r) { $('#bStatus').innerHTML = `<div class="status idle">MONITORING<small>No stress test has triggered yet. Credit, geopolitical or macro events with impact above 7 and two independent sources will trigger one.</small></div>`; return; }
    const d = r.decision;
    $('#bStatus').innerHTML = `<div class="status trig">${d ? 'STRESS TEST TRIGGERED' : 'MANUAL SCENARIO'}<small>${esc(r.label)}${d ? ' · ' + esc(d.reason) : ''}${r.trigger_text ? ' · “' + esc(r.trigger_text.slice(0, 110)) + '”' : ''}</small></div>`;
    $('#bKpis').innerHTML = kpis([
      ['Portfolio value', '$' + r.value_after.toFixed(0) + 'm', 'before $' + r.value_before.toFixed(0) + 'm'],
      ['Stress P&L', fmt(r.pnl, 1) + 'm', fmt(r.pnl_pct * 100, 2) + '% of book'],
      ['CET1 ratio', pct(r.cet1_after), 'before ' + pct(r.cet1_before) + ' · Δ ' + fmt((r.cet1_after - r.cet1_before) * 100, 2) + ' pp'],
      ['99% 10-day VaR', '$' + r.var_after.toFixed(1) + 'm', 'unstressed $' + r.var_before.toFixed(1) + 'm'],
      ['Expected shortfall', '$' + r.es_after.toFixed(1) + 'm', 'unstressed $' + r.es_before.toFixed(1) + 'm'],
    ]);
    $('#bLabel').textContent = r.label;
    const cls = Object.keys(r.by_class), vals = cls.map((c) => +r.by_class[c].dv.toFixed(2));
    let run = 0; const steps = vals.map((v) => { const a = run; run += v; return [a, run, v]; });
    steps.push([0, run, run]);
    const cats = [...cls, 'Total'];
    chart('bWater').setOption({
      animation: false, grid: { left: 50, right: 10, top: 24, bottom: 30 },
      tooltip: { formatter: (p) => `${cats[p.dataIndex]}: ${fmt(steps[p.dataIndex][2], 1)}m` },
      xAxis: { type: 'category', data: cats, ...axisStyle() },
      yAxis: { type: 'value', ...axisStyle(), scale: false },
      series: [{
        type: 'custom', data: steps.map((s, i) => [i, s[0], s[1], s[2]]), encode: { x: 0, y: [1, 2] },
        renderItem: (params, api) => {
          const i = api.value(0), a = api.value(1), b = api.value(2), v = api.value(3);
          const p0 = api.coord([i, Math.max(a, b)]), p1 = api.coord([i, Math.min(a, b)]);
          const w = api.size([1, 0])[0] * 0.6;
          const color = i === steps.length - 1 ? css('--ink') : (v >= 0 ? css('--pos') : css('--neg'));
          return { type: 'group', children: [
            { type: 'rect', shape: { x: p0[0] - w / 2, y: p0[1], width: w, height: Math.max(1, p1[1] - p0[1]) }, style: { fill: color } },
            { type: 'text', style: { text: fmt(v, 1), x: p0[0], y: (v >= 0 ? p0[1] - 12 : p1[1] + 4), textAlign: 'center', fill: css('--ink'), font: '600 11px JetBrains Mono' } }] };
        },
      }],
    }, true);
    chart('bBefore').setOption({
      animation: false, grid: { left: 50, right: 10, top: 30, bottom: 30 }, legend: { top: 0, textStyle: { color: css('--muted') } }, tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: cls, ...axisStyle() }, yAxis: { type: 'value', ...axisStyle(), name: '$m', nameTextStyle: { color: css('--muted') } },
      series: [{ name: 'Before', type: 'bar', itemStyle: { color: css('--muted') }, data: cls.map((c) => +r.by_class[c].v0.toFixed(1)) },
               { name: 'After', type: 'bar', itemStyle: { color: css('--accent') }, data: cls.map((c) => +r.by_class[c].v1.toFixed(1)) }],
    }, true);
    chart('bCet').setOption({
      animation: false, grid: { left: 20, right: 20, top: 20, bottom: 30 },
      xAxis: { type: 'value', min: 0, max: 15, ...axisStyle(), axisLabel: { ...axisStyle().axisLabel, formatter: '{value}%' } }, yAxis: { type: 'category', data: ['CET1'], show: false },
      series: [{ type: 'bar', data: [+(r.cet1_after * 100).toFixed(2)], itemStyle: { color: r.cet1_after < r.buffer_line ? css('--neg') : css('--ink') }, barWidth: 26,
        label: { show: true, position: 'right', color: css('--ink'), formatter: (p) => `after ${p.value}% (before ${(r.cet1_before * 100).toFixed(2)}%)` },
        markLine: { symbol: 'none', label: { color: css('--muted'), formatter: '{b}' }, data: [{ name: 'minimum 4.5%', xAxis: 4.5, lineStyle: { color: css('--neg') } }, { name: 'buffer 7%', xAxis: 7, lineStyle: { color: css('--warn') } }, { name: 'start', xAxis: +(r.cet1_before * 100).toFixed(2), lineStyle: { color: css('--muted'), type: 'dashed' } }] } }],
    }, true);
    const rv = r.reverse || {};
    $('#bReverse').innerHTML = rv.breach ? `<b>Reverse stress test.</b> CET1 falls through the ${pct(rv.line, 1)} line at <b>${rv.multiple.toFixed(2)}×</b> the full scenario${rv.implied_impact ? `, equivalent to an impact score of about <b>${rv.implied_impact.toFixed(1)}</b>` : ''}.` : '<b>Reverse stress test.</b> CET1 stays above the buffer line even at 3× the full scenario.';
    const sh = r.shock || {};
    $('#bShock').innerHTML = '<thead><tr><th>Factor</th><th class="n">Applied shock</th></tr></thead><tbody>' + [['Equities', 'eq', '%'], ['Rates', 'rates', 'bp'], ['IG spread', 'ig', 'bp'], ['HY spread', 'hy', 'bp'], ['EUR/USD', 'fx', '%'], ['Implied vol', 'vol', 'pts'], ['Credit factor Z', 'Z', 'σ'], ['LGD add-on', 'lgd_add', '']].map(([n, k, u]) => `<tr><td>${n}</td><td class="n">${sh[k] !== undefined ? fmt(+sh[k], k === 'Z' || k === 'lgd_add' ? 2 : 1) + ' ' + u : '—'}</td></tr>`).join('') + '</tbody>';
    $('#bBook').innerHTML = '<thead><tr><th>Position</th><th>Class</th><th>Method</th><th class="n">Before $m</th><th class="n">After $m</th><th class="n">ΔV $m</th><th class="n">Risk detail</th></tr></thead><tbody>' +
      r.positions.map((p) => { let det = '';
        if (p.kind === 'loan') det = `PD ${pct(p.pd)} → ${pct(p.pd_stressed)} · LGD ${pct(p.lgd, 0)} → ${pct(p.lgd_stressed, 0)}`;
        else if (p.kind === 'bond') det = `Δy ${fmt(p.dy_bp, 0)} bp`;
        return `<tr><td>${esc(p.name)}</td><td>${esc(p.asset_class)}</td><td>${esc(p.method || '')}</td><td class="n">${p.v0.toFixed(1)}</td><td class="n">${p.v1.toFixed(1)}</td><td class="n ${p.dv >= 0 ? 'up' : 'dn'}">${fmt(p.dv, 1)}</td><td class="n">${det}</td></tr>`; }).join('') + '</tbody>';
    $('#bHist').innerHTML = '<thead><tr><th>Time</th><th>Scenario</th><th class="n">P&L $m</th><th class="n">CET1 after</th></tr></thead><tbody>' + S.stressHist.slice(0, 20).map((h) => `<tr><td class="mono">${esc((h.ts || '').slice(0, 16).replace('T', ' '))}</td><td>${esc(h.label)}</td><td class="n dn">${fmt(h.pnl, 1)}</td><td class="n">${pct(h.cet1_after)}</td></tr>`).join('') + '</tbody>';
  }
  bindRange('sI', (v) => String(v));
  $('#sRun').addEventListener('click', async () => {
    const body = { event_class: $('#sCls').value, impact: +$('#sI').value, origin_sector: $('#sSector').value || null };
    S.stress = await api('/api/stress/run', { method: 'POST', body: JSON.stringify(body) }); S.stress.decision = null; S.stressHist.unshift(S.stress); dirty.add('modb'); schedule();
  });
  api('/api/stress/presets').then((p) => {
    $('#presetRow').innerHTML = Object.entries(p.presets).map(([k, v]) => `<button class="btn sm" data-preset="${k}">${esc(v)}</button>`).join('');
    $$('[data-preset]').forEach((b) => b.addEventListener('click', async () => {
      S.stress = await api('/api/stress/run', { method: 'POST', body: JSON.stringify({ preset: b.dataset.preset }) }); S.stress.decision = null; S.stressHist.unshift(S.stress); dirty.add('modb'); schedule();
    }));
  }).catch(() => {});

  // ---------------------------------------------------------------- engine lab
  const SAMPLES = ['JPMorgan downgraded two notches as commercial real estate losses mount', 'US imposes sweeping sanctions after missile strikes in the Strait of Hormuz; oil spikes 9%',
    'Nvidia unveils next-gen Blackwell Ultra GPUs, shares jump 6% in premarket', '$AAPL not looking good after iPhone sales miss lol', 'Microsoft beats Google to acquire gaming studio while Amazon slumps',
    'Pfizer losses narrowed in Q3 and the company raised guidance', 'Fifteen years after Lehman, analysts debate lessons from the 2008 crisis', 'Just baked an apple pie with my grandmother'];
  $('#labSamples').innerHTML = SAMPLES.map((s, i) => `<button class="btn" data-i="${i}">${esc(s)}</button>`).join('');
  $$('#labSamples button').forEach((b) => b.addEventListener('click', () => { $('#labIn').value = SAMPLES[+b.dataset.i]; runLab(); }));
  $('#labRun').addEventListener('click', runLab);
  async function runLab() {
    const res = await api('/api/analyze', { method: 'POST', body: JSON.stringify({ text: $('#labIn').value, source_type: $('#labSrc').value, publish: $('#labPub').checked }) });
    const rej = (res.rejected_entities || []).map((r) => `<div><span class="chip neg">rejected</span> “${esc(r.alias)}”: ${esc(r.reason)}</div>`).join('');
    const sigs = res.signals || [];
    $('#labOut').innerHTML = `<div class="ph"><h3>${sigs.length} signal(s)</h3><span class="muted small mono">${res.elapsed_ms} ms end to end</span></div>${rej}` +
      (sigs.length ? sigs.map((s) => `<div class="panel" style="margin-bottom:10px"><div class="ph"><span class="chip tk">${esc(s.entity.ticker)}</span></div>${signalDetail(s)}</div>`).join('') : `<div class="empty">${esc(res.dropped || 'No signal emitted.')}</div>`);
  }

  // ---------------------------------------------------------------- system
  async function loadSystem() {
    const st = await api('/api/status');
    $('#sysModels').innerHTML = `<dl class="dl">${Object.entries(st.models).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}${Object.entries(st.counters).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}<dt>queue depth</dt><dd>${st.queue_depth}</dd></dl>`;
    const lat = S.signals.map((s) => s.latency_ms.engine_total);
    const bins = Array.from({ length: 20 }, (_, i) => i * 1);
    const counts = bins.map((b) => lat.filter((v) => v >= b && v < b + 1).length);
    chart('sysLat').setOption({ animation: false, grid: { left: 40, right: 10, top: 10, bottom: 30 }, xAxis: { type: 'category', data: bins.map((b) => b + '–' + (b + 1)), ...axisStyle() }, yAxis: { type: 'value', ...axisStyle() },
      series: [{ type: 'bar', data: counts, itemStyle: { color: css('--blue') } }] }, true);
    const src = Object.values(st.sources || {});
    $('#sysSources').innerHTML = '<thead><tr><th>Source</th><th>State</th><th class="n">Docs</th><th class="n">Lag s</th><th>Last error / note</th></tr></thead><tbody>' +
      (src.length ? src.map((s) => `<tr><td>${esc(s.name)}</td><td>${esc(s.state)}</td><td class="n">${s.docs}</td><td class="n">${s.lag_seconds ?? '—'}</td><td class="small">${esc(s.last_error || s.note || '')}</td></tr>`).join('') : `<tr><td colspan="5" class="muted">Replay mode: sources are simulated from ${esc(st.replay.name || '')}. Start with <code>python -m trident serve --live</code> for live adapters.</td></tr>`) + '</tbody>';
    const rej = await api('/api/rejected');
    $('#sysRejected').innerHTML = rej.slice().reverse().map((r) => `<div><span class="chip">${esc(r.dropped || 'entity rejected')}</span> ${esc(r.text)} ${(r.rejected_entities || []).map((x) => `<span class="chip neg">${esc(x.alias)}: ${esc(x.reason)}</span>`).join('')}</div>`).join('') || '<span class="muted">Nothing dropped yet.</span>';
    const res = await api('/api/results');
    $('#sysResults').innerHTML = Object.keys(res).length ? `<pre>${esc(JSON.stringify(res.headline || res, null, 2))}</pre>` : '<span class="muted">Run python scripts/run_all_results.py to populate.</span>';
    const bt = res.backtest_synthetic || res.backtest_real;
    if (bt) $('#btCard').innerHTML = `<b>Backtest (${res.backtest_real ? 'real prices' : 'synthetic known-answer market'})</b><br>IR ${bt.strategy.information_ratio.toFixed(2)} · Sharpe ${bt.strategy.sharpe.toFixed(2)} vs cap-weight ${bt.cap_weight.sharpe.toFixed(2)} · cost ${bt.total_cost_bps.toFixed(0)} bps · ${bt.period.join(' → ')}`;
  }

  // ---------------------------------------------------------------- controls
  $('#replayBtn').addEventListener('click', async () => {
    S.signals = []; S.decisions = []; S.weights = []; S.stress = null; S.stressHist = []; S.selected = null;
    await api('/api/replay/start', { method: 'POST', body: JSON.stringify({ name: $('#replaySel').value, speed: +$('#speedSel').value }) });
    dirty = new Set(['feed', 'moda', 'modb']); schedule();
  });
  $('#injectBtn').addEventListener('click', () => api('/api/synthetic/inject?scenario=credit&n=10', { method: 'POST' }));

  connect(); pollStatus(); loadSystem().catch(() => {});
})();
