/* Price-target explorer: recompute probability-weighted targets client-side from published Base/Bull/Bear inputs. */
(function () {
  "use strict";
  const K = window.AdamKit;
  const $ = (id) => document.getElementById(id);
  const state = { data: null, published: null };
  const round2 = (v) => Number(Math.round(Number(v.toFixed(10) + "e2")) + "e-2"); // half-up, matches PriceTarget.weighted

  function weights() {
    const bull = Number($("w-bull").value), bear = Number($("w-bear").value);
    return { bull, bear, base: round2(Math.max(0, 1 - bull - bear)) };
  }

  function render() {
    let w = weights();
    if (w.bull + w.bear > 1) { $("w-bear").value = round2(1 - w.bull); w = weights(); }
    $("o-bull").textContent = w.bull.toFixed(2);
    $("o-bear").textContent = w.bear.toFixed(2);
    $("o-base").textContent = w.base.toFixed(2);
    const rows = state.data.targets.map((t) => {
      const mine = round2(t.base * w.base + t.bull * w.bull + t.bear * w.bear);
      const vsRef = t.reference_price ? mine / t.reference_price - 1 : null;
      return { ticker: t.ticker, method: t.method, bear: t.bear, base: t.base, bull: t.bull, published: t.weighted, weighted: mine, reference_price: t.reference_price, vs_reference: vsRef === null ? null : round2(vsRef * 100) };
    });
    $("rows").innerHTML = rows.map((r) => K.html`<tr data-adam-hook="target-row" data-ticker="${r.ticker}">
      <td>${r.ticker}</td><td>${r.method}</td><td>${r.bear}</td><td>${r.base}</td><td>${r.bull}</td><td>${r.published}</td>
      <td class="adam-term__big">${r.weighted}</td><td>${r.reference_price === null ? "—" : r.reference_price}</td>
      <td class="${r.vs_reference === null ? "" : r.vs_reference >= 0 ? "up" : "down"}">${r.vs_reference === null ? "—" : r.vs_reference.toFixed(1) + "%"}</td></tr>`).join("");
    K.publish("price_target_explorer", { as_of: state.data.as_of, weights: w, published_weights: state.published, inputs: "data/publications/price_targets_latest.json", rows });
  }

  function reset() {
    $("w-bull").value = state.published.bull;
    $("w-bear").value = state.published.bear;
    render();
  }

  async function init() {
    try {
      const r = await fetch("data/publications/price_targets_latest.json");
      if (!r.ok) throw new Error("HTTP " + r.status);
      state.data = await r.json();
    } catch (err) {
      $("status").textContent = "Could not load data/publications/price_targets_latest.json (" + err.message + "). Serve the repo over HTTP.";
      return;
    }
    state.published = state.data.targets[0].weights;
    $("status").textContent = state.data.targets.length + " tickers · as of " + state.data.as_of;
    $("meta").textContent = "Risk-free rate " + state.data.risk_free_rate_pct + "% (synthetic 10Y). " + state.data.label + ".";
    $("w-bull").addEventListener("input", render);
    $("w-bear").addEventListener("input", render);
    $("reset").addEventListener("click", reset);
    reset();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
