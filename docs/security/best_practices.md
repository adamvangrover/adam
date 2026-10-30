# Cybersecurity Engineering Best Practices & Defense-in-Depth

This document consolidates critical security vulnerabilities, architectural learnings, and defense-in-depth patterns captured across AFOS operations and Sentinel audits.

For portable operator prompts, runtime-agnostic minimums, and static code repository ingestion standards, see the [Operator Security Baseline & Governance Invariants](operator_security_baseline.md).

---

## 1. SSRF via Automated HTTP Redirects

* **Vulnerability:** Validating a URL's domain against an allowlist before making a request with `urllib.request.urlopen` is insufficient if the client follows redirects. An attacker can host an allowed URL that redirects to internal infrastructure (`127.0.0.1`, cloud metadata endpoints `169.254.169.254`).
* **Learning & Prevention:** Always disable automatic redirects when fetching untrusted URLs, or re-validate the target domain in a custom `urllib.request.HTTPRedirectHandler`. When using `urllib.request`, implement a custom `HTTPRedirectHandler` returning `None` for `redirect_request` and build an opener via `urllib.request.build_opener(NoRedirectHandler(), ...)`.

---

## 2. Command / Interpreter Option Injection via `--`

* **Vulnerability:** Evaluating scripts using `subprocess.run(['bash', '-lc', script])` or invoking `git clone <url>` without options delimiters allows payload strings starting with `-` (e.g. `-i` or `--upload-pack`) to be parsed as CLI flags rather than arguments.
* **Learning & Prevention:** Always insert the end-of-options separator `--` prior to untrusted command inputs (e.g., `['bash', '-lc', '--', script]` or `['git', 'clone', '--', url]`).

---

## 3. Dangerous Git URL Protocols (`ext::`, `file://`)

* **Vulnerability:** Passing an unvalidated URL to `git clone` can trigger arbitrary command execution via Git's native `ext::` transport or disclose sensitive local files via `file://`.
* **Learning & Prevention:** Strictly validate URL schemes, permitting only `https://` (and `http://` where explicitly needed), and reject unexpected protocol prefixes.

---

## 4. Jinja2 Cross-Site Scripting (XSS)

* **Vulnerability:** Instantiating Jinja2 `Environment()` without `autoescape=True` leaves generated HTML reports and dashboards open to template injection and XSS from untrusted model outputs or feeds.
* **Learning & Prevention:** Explicitly pass `autoescape=True` or `autoescape=select_autoescape(['html', 'xml'])` on all Jinja2 environment instantiations.

---

## 5. Unverified SSL Contexts

* **Vulnerability:** Using `ssl._create_unverified_context()` bypasses certificate validation and exposes network traffic to Man-in-the-Middle (MitM) attacks.
* **Learning & Prevention:** Always use `ssl.create_default_context()` for all network calls to enforce strict certificate and hostname verification.

---

## 6. Insecure Static Defaults for Secrets

* **Vulnerability:** Using fallback strings like `"default-insecure-key"` in configuration files risks accidental production exposure.
* **Learning & Prevention:** Dynamically generate cryptographically secure random secrets via `secrets.token_urlsafe(32)` if an environment variable is omitted, ensuring that any missing configuration defaults to an unguessable key.
