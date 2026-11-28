/* ADAM showcase kit: shared escaping helpers and the machine operator surface.
- Load with <script src=".../kit/adam-kit.js"></script> in <head> so inline scripts can call AdamKit.esc (scripts/apply_showcase_kit.py does this).
-   AdamKit.esc(value)             -> HTML-escaped string for interpolation into innerHTML
-   AdamKit.markdown(src)          -> marked.parse(src) passed through an allowlist HTML sanitizer
-   AdamKit.html`<b>${value}</b>`  -> tagged template that escapes every interpolation
-   AdamKit.publish(key, data)     -> add page data to the operator JSON view
-   AdamKit.snapshot()             -> {page, title, role, family, canonical, hooks, data}
- Operator view: append ?view=json (or press Alt+J) to see the snapshot as JSON.
- Agents read <html data-adam-*>, [data-adam-hook] elements and window.AdamKit.snapshot().
 */
(function () {
  "use strict";
  if (window.AdamKit) return;

  const MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;", "`": "&#96;" };
  const esc = (v) => String(v == null ? "" : v).replace(/[&<>"'`]/g, (c) => MAP[c]);
  const html = (strings, ...values) =>
    strings.reduce((out, s, i) => out + s + (i < values.length ? esc(values[i]) : ""), "");

  // Strip ASCII control characters before inspecting schemes so inputs like "java\tscript:" cannot bypass.
  const SAFE_HREF_SCHEMES = new Set(["http", "https", "mailto", "tel"]);
  const SAFE_SRC_SCHEMES = new Set(["http", "https"]);
  const cleanUrl = (u, allowedSchemes) => {
    if (u == null) return null;
    const cleaned = String(u).replace(/[\u0000-\u001F\u007F]+/g, "").trim();
    if (!cleaned) return null;
    const scheme = cleaned.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):/);
    if (scheme) return allowedSchemes.has(scheme[1].toLowerCase()) ? cleaned : null;
    return cleaned;
  };
  const safeUrl = (u) => (u === "" ? "" : cleanUrl(u, SAFE_HREF_SCHEMES) || "#");

  const published = Object.create(null);
  // Read before inline scripts run: pages like the catalog rewrite the URL with history.replaceState.
  const wantJson = new URLSearchParams(location.search).get("view") === "json";

  function safeJson(value) {
    try {
      const ancestors = [];
      return JSON.stringify(
        value,
        function (_k, v) {
          if (typeof v === "bigint") return v.toString();
          if (typeof v === "object" && v !== null) {
            while (ancestors.length > 0 && ancestors[ancestors.length - 1] !== this) {
              ancestors.pop();
            }
            if (ancestors.includes(v)) return "[Circular]";
            ancestors.push(v);
          }
          return v;
        },
        2
      );
    } catch (e) {
      return JSON.stringify({ _serialize_error: String(e) }, null, 2);
    }
  }

  function publish(key, data) {
    published[String(key)] = data;
    const panel = document.getElementById("adam-operator-view");
    if (panel) panel.textContent = safeJson(snapshot());
    document.dispatchEvent(new CustomEvent("adam:data", { detail: { key } }));
  }

  function hooks() {
    return Array.from(document.querySelectorAll("[data-adam-hook]:not(#adam-operator-view)")).map((el) => {
      const extra = {};
      for (const [k, v] of Object.entries(el.dataset)) {
        if (k !== "adamHook" && k !== "hook" && k !== "text") extra[k] = v;
      }
      return {
        hook: el.dataset.adamHook,
        text: (el.textContent || "").trim().slice(0, 500),
        ...extra,
      };
    });
  }

  function snapshot() {
    const d = document.documentElement.dataset;
    const embedded = document.getElementById("adam-operator-data");
    let data = Object.assign(Object.create(null), published);
    if (embedded) {
      try {
        const parsed = JSON.parse(embedded.textContent || "null");
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          data = Object.assign(Object.create(null), parsed, data);
        } else if (parsed !== null) {
          data._embedded = parsed;
        }
      } catch (e) {
        data._embedded_error = String(e);
      }
    }
    const segments = location.pathname.split("/").filter(Boolean);
    return {
      page: segments.length ? segments.slice(-2).join("/") : "/",
      title: document.title,
      role: d.adamRole || "unknown",
      family: d.adamFamily || "other",
      canonical: d.adamCanonical || null,
      hooks: hooks(),
      data,
    };
  }

  function showJson() {
    const btn = document.getElementById("adam-operator-toggle");
    let panel = document.getElementById("adam-operator-view");
    if (panel) {
      panel.remove();
      if (btn) btn.setAttribute("aria-expanded", "false");
      return;
    }
    if (!document.body) return;
    panel = document.createElement("pre");
    panel.id = "adam-operator-view";
    panel.className = "adam-operator-view";
    panel.setAttribute("data-adam-hook", "operator-view");
    panel.textContent = safeJson(snapshot());
    document.body.appendChild(panel);
    if (btn) btn.setAttribute("aria-expanded", "true");
  }

  function supersededBanner() {
    const canonical = document.documentElement.dataset.adamCanonical;
    if (!canonical || document.getElementById("adam-superseded") || !document.body) return;
    const bar = document.createElement("div");
    bar.id = "adam-superseded";
    bar.className = "adam-superseded";
    bar.setAttribute("data-adam-hook", "superseded");
    bar.append("Earlier generation kept as a template. Current terminal: ");
    const a = document.createElement("a");
    a.href = safeUrl(canonical);
    a.textContent = canonical.replace(/\/+$/, "").split("/").pop() || canonical;
    bar.appendChild(a);
    document.body.prepend(bar);
  }

  function init() {
    if (!document.body || document.getElementById("adam-operator-toggle")) return;
    supersededBanner();
    const btn = document.createElement("button");
    btn.id = "adam-operator-toggle";
    btn.type = "button";
    btn.className = "adam-operator-toggle";
    btn.title = "Machine operator view (Alt+J)";
    btn.setAttribute("aria-label", "Toggle machine operator JSON view");
    btn.setAttribute("aria-controls", "adam-operator-view");
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("data-adam-hook", "operator-toggle");
    btn.textContent = "{ }";
    btn.addEventListener("click", showJson);
    document.body.appendChild(btn);
    document.addEventListener("keydown", (e) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === "j" || e.key === "J" || (e.code === "KeyJ" && !/^[a-z]$/i.test(e.key))) {
        const t = e.target;
        if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
        e.preventDefault();
        showJson();
      }
    });
    if (wantJson) showJson();
  }

  // Markdown via window.marked, then an iterative allowlist sanitizer immune to DOM clobbering.
  const MD_TAGS = new Set(
    "a b blockquote br code del div em h1 h2 h3 h4 h5 h6 hr i img li ol p pre s span strong sub sup table tbody td th thead tr ul".split(" ")
  );
  const MD_DROP = new Set([
    "script", "style", "iframe", "object", "embed", "template", "noscript",
    "svg", "math", "form", "link", "meta", "base", "head",
    "title", "textarea", "xmp", "noembed", "noframes", "plaintext",
    "applet", "frame", "frameset", "input", "button", "select", "option",
    "audio", "video", "source", "track", "canvas", "portal",
  ]);
  const MD_GLOBAL_ATTRS = new Set(["title", "class"]);
  const MD_TAG_ATTRS = {
    a: new Set(["href"]),
    img: new Set(["src", "alt"]),
  };

  // Cached prototype accessors prevent HTMLFormElement named-property DOM clobbering.
  const getNodeType = Object.getOwnPropertyDescriptor(Node.prototype, "nodeType").get;
  const getNodeName = Object.getOwnPropertyDescriptor(Node.prototype, "nodeName").get;
  const getFirstChild = Object.getOwnPropertyDescriptor(Node.prototype, "firstChild").get;
  const getNextSibling = Object.getOwnPropertyDescriptor(Node.prototype, "nextSibling").get;
  const getAttributes = Object.getOwnPropertyDescriptor(Element.prototype, "attributes").get;
  const insertBeforeNode = Node.prototype.insertBefore;
  const removeChildNode = Node.prototype.removeChild;
  const removeAttr = Element.prototype.removeAttribute;
  const setAttr = Element.prototype.setAttribute;

  function sanitize(root) {
    const stack = [root];
    while (stack.length) {
      const parent = stack.pop();
      let child = getFirstChild.call(parent);
      while (child) {
        const type = getNodeType.call(child);
        if (type !== 1) {
          const next = getNextSibling.call(child);
          if (type !== 3) removeChildNode.call(parent, child);
          child = next;
          continue;
        }
        const tag = getNodeName.call(child).toLowerCase();
        if (MD_DROP.has(tag)) {
          const next = getNextSibling.call(child);
          removeChildNode.call(parent, child);
          child = next;
          continue;
        }
        if (!MD_TAGS.has(tag)) {
          const firstMoved = getFirstChild.call(child);
          const next = getNextSibling.call(child);
          let inner;
          while ((inner = getFirstChild.call(child))) {
            insertBeforeNode.call(parent, inner, child);
          }
          removeChildNode.call(parent, child);
          child = firstMoved || next;
          continue;
        }
        const tagAttrs = MD_TAG_ATTRS[tag];
        for (const at of Array.from(getAttributes.call(child))) {
          const name = at.name.toLowerCase();
          if (!MD_GLOBAL_ATTRS.has(name) && !(tagAttrs && tagAttrs.has(name))) {
            removeAttr.call(child, at.name);
          } else if (name === "href") {
            setAttr.call(child, at.name, safeUrl(at.value));
          } else if (name === "src") {
            const cleaned = cleanUrl(at.value, SAFE_SRC_SCHEMES);
            if (cleaned) setAttr.call(child, at.name, cleaned);
            else removeAttr.call(child, at.name);
          }
        }
        stack.push(child);
        child = getNextSibling.call(child);
      }
    }
    return root;
  }

  const markdown = (src) => {
    const m = window.marked;
    const text = String(src == null ? "" : src);
    if (!m || typeof DOMParser === "undefined") return esc(text);
    const rendered = typeof m.parse === "function" ? m.parse(text) : typeof m === "function" ? m(text) : null;
    if (typeof rendered !== "string") return esc(text);
    const doc = new DOMParser().parseFromString("<!DOCTYPE html><body>" + rendered + "</body>", "text/html");
    return sanitize(doc.body).innerHTML;
  };

  window.AdamKit = Object.freeze({ esc, html, safeUrl, markdown, publish, snapshot, showJson });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
