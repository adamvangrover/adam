/* ADAM showcase kit: shared escaping helpers and the machine operator surface.
 * Load with <script src=".../kit/adam-kit.js" defer></script> (scripts/apply_showcase_kit.py does this).
 *   AdamKit.esc(value)             -> HTML-escaped string for interpolation into innerHTML
 *   AdamKit.html`<b>${value}</b>`  -> tagged template that escapes every interpolation
 *   AdamKit.publish(key, data)     -> add page data to the operator JSON view
 *   AdamKit.snapshot()             -> {page, role, family, canonical, hooks, data}
 * Operator view: append ?view=json (or press Alt+J) to see the snapshot as JSON.
 * Agents read <html data-adam-*>, [data-adam-hook] elements and window.AdamKit.snapshot().
 */
(function () {
  "use strict";
  const MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;", "`": "&#96;" };
  const esc = (v) => String(v === undefined || v === null ? "" : v).replace(/[&<>"'`]/g, (c) => MAP[c]);
  const html = (strings, ...values) => strings.reduce((out, s, i) => out + s + (i < values.length ? esc(values[i]) : ""), "");
  const safeUrl = (u) => (/^\s*(javascript|data|vbscript):/i.test(String(u)) ? "#" : String(u));
  const published = {};

  function publish(key, data) {
    published[key] = data;
    document.dispatchEvent(new CustomEvent("adam:data", { detail: { key } }));
  }

  function hooks() {
    return Array.from(document.querySelectorAll("[data-adam-hook]")).map((el) => ({
      hook: el.dataset.adamHook,
      text: (el.textContent || "").trim().slice(0, 500),
      ...Object.fromEntries(Object.entries(el.dataset).filter(([k]) => k !== "adamHook")),
    }));
  }

  function snapshot() {
    const d = document.documentElement.dataset;
    const embedded = document.getElementById("adam-operator-data");
    let data = Object.assign({}, published);
    if (embedded) {
      try { data = Object.assign(JSON.parse(embedded.textContent), data); } catch (e) { data._embedded_error = String(e); }
    }
    return {
      page: location.pathname.split("/").slice(-2).join("/"),
      title: document.title,
      role: d.adamRole || "unknown",
      family: d.adamFamily || "other",
      canonical: d.adamCanonical || null,
      hooks: hooks(),
      data,
    };
  }

  function showJson() {
    let panel = document.getElementById("adam-operator-view");
    if (panel) { panel.remove(); return; }
    panel = document.createElement("pre");
    panel.id = "adam-operator-view";
    panel.className = "adam-operator-view";
    panel.setAttribute("data-adam-hook", "operator-view");
    panel.textContent = JSON.stringify(snapshot(), null, 2);
    document.body.appendChild(panel);
  }

  function supersededBanner() {
    const canonical = document.documentElement.dataset.adamCanonical;
    if (!canonical || document.getElementById("adam-superseded")) return;
    const bar = document.createElement("div");
    bar.id = "adam-superseded";
    bar.className = "adam-superseded";
    bar.setAttribute("data-adam-hook", "superseded");
    bar.append("Earlier generation kept as a template. Current terminal: ");
    const a = document.createElement("a");
    a.href = safeUrl(canonical);
    a.textContent = canonical.split("/").pop();
    bar.appendChild(a);
    document.body.prepend(bar);
  }

  function init() {
    supersededBanner();
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "adam-operator-toggle";
    btn.title = "Machine operator view (Alt+J)";
    btn.setAttribute("aria-label", "Toggle machine operator JSON view");
    btn.setAttribute("data-adam-hook", "operator-toggle");
    btn.textContent = "{ }";
    btn.addEventListener("click", showJson);
    document.body.appendChild(btn);
    document.addEventListener("keydown", (e) => { if (e.altKey && (e.key === "j" || e.key === "J")) showJson(); });
    if (new URLSearchParams(location.search).get("view") === "json") showJson();
  }

  window.AdamKit = { esc, html, safeUrl, publish, snapshot, showJson };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
