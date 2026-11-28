/* Review panel: desk verdicts, red-team challenges and judge scores for every schema-rendered edition. */
(function () {
  "use strict";
  const K = window.AdamKit;
  const $ = (id) => document.getElementById(id);
  const rel = (p) => String(p || "").replace(/^showcase\//, "");
  const state = { editions: [], selected: null };

  function filtered() {
    const fam = $("f-family").value, verdict = $("f-verdict").value, flag = $("f-flag").value;
    return state.editions.filter((e) => (!fam || e.family === fam) && (!verdict || e.review_verdict === verdict) && (!flag || (e.flags || []).includes(flag)))
      .sort((a, b) => b.date.localeCompare(a.date) || a.family.localeCompare(b.family));
  }

  function render() {
    const rows = filtered();
    const mean = rows.length ? rows.reduce((s, e) => s + e.review_score, 0) / rows.length : 0;
    $("summary").textContent = rows.length + " editions · mean judge score " + mean.toFixed(3);
    $("rows").innerHTML = rows.map((e) => K.html`<tr data-adam-hook="review-row" data-id="${e.id}">
      <td>${e.date}</td><td>${e.family}</td><td>${Number(e.review_score).toFixed(3)}</td><td>${e.review_verdict}</td>
      <td>${(e.flags || []).join(", ") || "—"}</td><td><button type="button" data-adam-hook="open-review" data-id="${e.id}">Open</button></td></tr>`).join("");
    K.publish("review_panel", { filters: { family: $("f-family").value, verdict: $("f-verdict").value, flag: $("f-flag").value }, count: rows.length, mean_score: Number(mean.toFixed(4)), rows });
  }

  async function open(id) {
    const entry = state.editions.find((e) => e.id === id);
    const r = await fetch(rel(entry.json));
    const edition = await r.json();
    const rv = edition.review || {};
    const panel = rv.panel || {};
    const levels = (data) => Object.entries(Object.assign({}, data.targets || {}, data.price_targets || {}))
      .map(([k, v]) => K.html`<div>${k} ${v}</div>`).join("");
    const card = (side, data) => K.html`<article class="adam-term__card adam-term__card--${side}" data-adam-hook="review-${side}"><h3>${side.replace("_", " ")}</h3><p>${data.narrative || data.summary || "—"}</p>` + levels(data) + "</article>";
    const list = (items) => (items && items.length ? "<ul>" + items.map((x) => K.html`<li>${x}</li>`).join("") + "</ul>" : "<p>None.</p>");
    $("detail").hidden = false;
    $("detail").innerHTML = K.html`<h2>${edition.title} (${entry.id})</h2><p><a href="${K.safeUrl(rel(entry.html))}">Open edition</a> · judge ${rv.score} · desk ${(rv.desk || {}).verdict} · red-team impact ${(rv.red_team || {}).impact_score}</p>` +
      '<div class="adam-term__grid">' + card("bull", panel.bull || {}) + card("bear", panel.bear || {}) + card("red_team", panel.red_team || {}) + "</div>" +
      "<h3>Red-team challenges</h3>" + list((panel.red_team || {}).challenges) +
      "<h3>Flags</h3>" + list((rv.flags || []).map((f) => f.severity + " " + f.code + ": " + f.message)) +
      '<h3>Judge dimensions</h3><table data-adam-hook="judge-dimensions"><thead><tr><th>Dimension</th><th>Score</th><th>Weight</th><th>Reasoning</th></tr></thead><tbody>' +
      ((rv.judge || {}).dimensions || []).map((d) => K.html`<tr><td>${d.name}</td><td>${d.score}</td><td>${d.weight}</td><td>${d.reasoning}</td></tr>`).join("") + "</tbody></table>";
    K.publish("review_detail", { id: entry.id, review: rv });
  }

  function options(id, values) {
    $(id).insertAdjacentHTML("beforeend", values.map((v) => K.html`<option value="${v}">${v}</option>`).join(""));
  }

  async function init() {
    try {
      const r = await fetch("data/publications/index.json");
      if (!r.ok) throw new Error("HTTP " + r.status);
      state.editions = (await r.json()).editions.filter((e) => e.review_score !== undefined);
    } catch (err) {
      $("status").textContent = "Could not load data/publications/index.json (" + err.message + "). Serve the repo over HTTP.";
      return;
    }
    $("status").textContent = state.editions.length + " reviewed editions (offline deterministic review)";
    options("f-family", Array.from(new Set(state.editions.map((e) => e.family))).sort());
    options("f-verdict", Array.from(new Set(state.editions.map((e) => e.review_verdict))).sort());
    options("f-flag", Array.from(new Set(state.editions.flatMap((e) => e.flags || []))).sort());
    ["f-family", "f-verdict", "f-flag"].forEach((id) => $(id).addEventListener("input", render));
    $("rows").addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-id]");
      if (btn) open(btn.dataset.id).catch((err) => { $("status").textContent = "Load failed: " + err.message; });
    });
    render();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
