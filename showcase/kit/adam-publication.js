/* Publishes the schema-rendered edition and its review scores via AdamKit.publish (see docs/TEMPLATES.md §3a). */
(function () {
  "use strict";
  function init() {
    const el = document.getElementById("adam-operator-data");
    if (!el || !window.AdamKit) return;
    let record;
    try { record = JSON.parse(el.textContent); } catch (e) { return; }
    const review = record.review || {};
    window.AdamKit.publish("review_scores", {
      edition: record.family + "/" + record.date,
      score: review.score === undefined ? null : review.score,
      judge: review.judge || null,
      desk_verdict: (review.desk || {}).verdict || null,
      red_team_impact: (review.red_team || {}).impact_score === undefined ? null : review.red_team.impact_score,
      flags: (review.flags || []).map(function (f) { return f.code; }),
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
