/* Editions browser: filters, list and A/B diff over showcase/data/publications/index.json (schema-rendered editions). */
(function () {
  "use strict";
  const K = window.AdamKit;
  const $ = (id) => document.getElementById(id);
  const rel = (p) => String(p || "").replace(/^showcase\//, "");
  const state = { editions: [], pick: [], cache: {} };

  function setStatus(msg) { $("status").textContent = msg; }

  async function loadJson(path) {
    if (!state.cache[path]) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(path + ": HTTP " + r.status);
      state.cache[path] = await r.json();
    }
    return state.cache[path];
  }

  function filters() {
    return { family: $("f-family").value, from: $("f-from").value, to: $("f-to").value, ticker: $("f-ticker").value, q: $("f-q").value.trim().toLowerCase() };
  }

  function filtered() {
    const f = filters();
    return state.editions.filter((e) => (!f.family || e.family === f.family) && (!f.from || e.date >= f.from) && (!f.to || e.date <= f.to)
      && (!f.ticker || (e.tickers || []).includes(f.ticker)) && (!f.q || e.title.toLowerCase().includes(f.q)))
      .sort((a, b) => b.date.localeCompare(a.date) || a.family.localeCompare(b.family));
  }

  function render() {
    const rows = filtered();
    $("count").textContent = rows.length + " of " + state.editions.length + " editions";
    $("rows").innerHTML = rows.map((e) => K.html`<tr data-adam-hook="edition-row" data-id="${e.id}">
      <td>${e.date}</td><td>${e.family}</td><td><a href="${K.safeUrl(rel(e.html))}">${e.title}</a></td>
      <td>${e.review_score === undefined ? "—" : Number(e.review_score).toFixed(2)} ${e.review_verdict || ""}</td>
      <td><button type="button" data-adam-hook="pick-a" data-pick="0" data-id="${e.id}" aria-pressed="${state.pick[0] === e.id}">A</button>
          <button type="button" data-adam-hook="pick-b" data-pick="1" data-id="${e.id}" aria-pressed="${state.pick[1] === e.id}">B</button></td></tr>`).join("");
    $("pick").textContent = "A: " + (state.pick[0] || "—") + " · B: " + (state.pick[1] || "—");
    K.publish("editions", { filters: filters(), count: rows.length, rows });
  }

  function diffRows(a, b, key, fields) {
    const ma = Object.fromEntries((a || []).map((x) => [x[key], x]));
    const mb = Object.fromEntries((b || []).map((x) => [x[key], x]));
    return Array.from(new Set([...Object.keys(ma), ...Object.keys(mb)])).sort().map((k) => {
      const row = { [key]: k };
      fields.forEach((f) => { row[f + "_a"] = ma[k] ? ma[k][f] : null; row[f + "_b"] = mb[k] ? mb[k][f] : null; });
      row.changed = fields.some((f) => row[f + "_a"] !== row[f + "_b"]);
      return row;
    });
  }

  async function renderDiff() {
    const box = $("diff");
    if (state.pick.filter(Boolean).length < 2) { box.hidden = true; K.publish("compare", null); return; }
    const [ea, eb] = state.pick.map((id) => state.editions.find((e) => e.id === id));
    const [a, b] = await Promise.all([loadJson(rel(ea.json)), loadJson(rel(eb.json))]);
    const pts = (e) => (e.price_targets || []).map((t) => Object.assign({}, t, { weighted: Math.round((t.base * t.weights.base + t.bull * t.weights.bull + t.bear * t.weights.bear) * 100) / 100 }));
    const diff = {
      a: ea.id, b: eb.id,
      calls: diffRows(a.calls, b.calls, "subject", ["stance", "conviction"]),
      targets: diffRows(pts(a), pts(b), "ticker", ["base", "weighted"]),
    };
    const fmt = (v) => (v === null || v === undefined ? "—" : v);
    box.hidden = false;
    box.innerHTML = K.html`<h2>Diff ${diff.a} → ${diff.b}</h2>` +
      '<h3>Calls</h3><table data-adam-hook="diff-calls"><thead><tr><th>Subject</th><th>A stance</th><th>B stance</th><th>A conv.</th><th>B conv.</th></tr></thead><tbody>' +
      diff.calls.map((r) => K.html`<tr class="${r.changed ? "chg" : ""}"><td>${r.subject}</td><td>${fmt(r.stance_a)}</td><td>${fmt(r.stance_b)}</td><td>${fmt(r.conviction_a)}</td><td>${fmt(r.conviction_b)}</td></tr>`).join("") +
      '</tbody></table><h3>Price targets</h3><table data-adam-hook="diff-targets"><thead><tr><th>Ticker</th><th>A base</th><th>B base</th><th>A weighted</th><th>B weighted</th></tr></thead><tbody>' +
      (diff.targets.length ? diff.targets.map((r) => K.html`<tr class="${r.changed ? "chg" : ""}"><td>${r.ticker}</td><td>${fmt(r.base_a)}</td><td>${fmt(r.base_b)}</td><td>${fmt(r.weighted_a)}</td><td>${fmt(r.weighted_b)}</td></tr>`).join("") : '<tr><td colspan="5">No price targets in either edition.</td></tr>') +
      "</tbody></table>";
    K.publish("compare", diff);
  }

  function options(id, values) {
    $(id).insertAdjacentHTML("beforeend", values.map((v) => K.html`<option value="${v}">${v}</option>`).join(""));
  }

  async function init() {
    try {
      const index = await loadJson("data/publications/index.json");
      state.editions = index.editions;
      options("f-family", Array.from(new Set(state.editions.map((e) => e.family))).sort());
      options("f-ticker", Array.from(new Set(state.editions.flatMap((e) => e.tickers || []))).sort());
      setStatus(index.editions.length + " editions · " + index.label);
    } catch (err) {
      setStatus("Could not load data/publications/index.json (" + err.message + "). Serve the repo over HTTP, e.g. python -m http.server.");
      return;
    }
    ["f-family", "f-from", "f-to", "f-ticker", "f-q"].forEach((id) => $(id).addEventListener("input", render));
    $("rows").addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-pick]");
      if (!btn) return;
      const slot = Number(btn.dataset.pick);
      state.pick[slot] = state.pick[slot] === btn.dataset.id ? undefined : btn.dataset.id;
      render();
      renderDiff().catch((err) => setStatus("Diff failed: " + err.message));
    });
    $("clear").addEventListener("click", () => { state.pick = []; render(); renderDiff(); });
    render();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
