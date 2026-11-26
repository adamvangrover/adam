/* ADAM showcase kit: shared escaping helpers and the machine operator surface.
 * Load with <script src=".../kit/adam-kit.js"></script> in <head> so inline scripts can call AdamKit.esc (scripts/apply_showcase_kit.py does this).
 *   AdamKit.esc(value)             -> HTML-escaped string for interpolation into innerHTML
 *   AdamKit.markdown(src)          -> marked.parse(src) passed through an allowlist HTML sanitizer
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
  // Read before inline scripts run: pages like the catalog rewrite the URL with history.replaceState.
  const wantJson = new URLSearchParams(location.search).get("view") === "json";

  function publish(key, data) {
    published[key] = data;
    const panel = document.getElementById("adam-operator-view");
    if (panel) panel.textContent = JSON.stringify(snapshot(), null, 2);
    document.dispatchEvent(new CustomEvent("adam:data", { detail: { key } }));
  }

  function hooks() {
    return Array.from(document.querySelectorAll("[data-adam-hook]:not(#adam-operator-view)")).map((el) => ({
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
    if (wantJson) showJson();
  }

  // Markdown via window.marked, then an allowlist sanitizer: unknown tags are unwrapped (script/style/etc. dropped),
  // only href/src/alt/title/class attributes survive and URLs go through safeUrl. Plain text without marked.
  const MD_TAGS = new Set("a b blockquote br code del div em h1 h2 h3 h4 h5 h6 hr i img li ol p pre s span strong sub sup table tbody td th thead tr ul".split(" "));
  const MD_DROP = new Set(["script", "style", "iframe", "object", "embed", "template", "noscript", "svg", "math", "form", "link", "meta", "base"]);
  const MD_ATTRS = new Set(["href", "src", "alt", "title", "class"]);
  const sanitize = (node) => {
    Array.from(node.childNodes).forEach((child) => {
      if (child.nodeType !== 1) {
        if (child.nodeType !== 3) child.remove();
        return;
      }
      const tag = child.tagName.toLowerCase();
      if (MD_DROP.has(tag)) return child.remove();
      sanitize(child);
      if (!MD_TAGS.has(tag)) return child.replaceWith(...child.childNodes);
      Array.from(child.attributes).forEach((at) => {
        const name = at.name.toLowerCase();
        if (!MD_ATTRS.has(name)) child.removeAttribute(at.name);
        else if (name === "href" || name === "src") child.setAttribute(name, safeUrl(at.value));
      });
    });
    return node;
  };
  const markdown = (src) => {
    const m = window.marked;
    const text = String(src === undefined || src === null ? "" : src);
    if (!m || typeof DOMParser === "undefined") return esc(text);
    const doc = new DOMParser().parseFromString("<body>" + (m.parse || m)(text) + "</body>", "text/html");
    return sanitize(doc.body).innerHTML;
  };

  window.AdamKit = { esc, html, safeUrl, markdown, publish, snapshot, showJson };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
