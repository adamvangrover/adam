# ADAM Template Library

> **These are starting templates — pick one, refine to taste.**

ADAM has accumulated many brands, products, newsletters, terminals and prompts. None of them is the
canonical voice. Each family below is a set of parallel defaults: copy the variant closest to your
audience, change what is listed under *Customize*, and leave the rest. Everything here is presentation or
prompt material — the execution contract (`AgentInput`/`AgentOutput`, `provenance_trace`) lives in
[`AGENTS.md`](../AGENTS.md) and does not change when you re-skin a page.

Related: [README § Templates & Brands](../README.md#-templates--brands) ·
[REPO_MAP.md](../REPO_MAP.md) · [showcase/README.md](../showcase/README.md) ·
[ARCHITECTURE.md](ARCHITECTURE.md)

| # | Family | Primary paths | Format |
|---|--------|---------------|--------|
| 1 | [Documentation templates](#1-documentation-templates) | `docs/templates/`, `prompt_library/templates/` | Markdown |
| 2 | [Brand / product pages](#2-brand--product-pages) | `docs/brands/`, `public/*/`, `showcase/dashboards/`, `showcase/analyst_os*.html`, `showcase/portal/` | Static HTML |
| 3 | [Newsletter / report families](#3-newsletter--report-families) | `showcase/Daily_Briefing_*`, `Market_Pulse_*`, `House_View_*`, `market-mayhem/templates/`, `newsletters/` | HTML / Markdown |
| 4 | [Terminal variants](#4-terminal-variants) | `showcase/dashboards/adam_v30_*`, `adam_v31_0_*`, `terminals/`, `public/adam-terminals/` | Static HTML |
| 5 | [Prompt templates](#5-prompt-templates) | `prompt_library/AOPL-v1.0/`, `prompt_library/AOPL-v2.0/` | Markdown / YAML |
| 6 | [Code templates](#6-code-templates) | `core/agents/templates/`, `market-mayhem/standalone/` | Python |

Version labels inside the files (`v23.5`, `v26`, `v30.1`, `v31.0`) are the release each variant was
written for. They are style lineage, not a statement about the current runtime.

---

## 1. Documentation templates

| Template | Path | Purpose | Copy | Customize |
|---|---|---|---|---|
| Agent doc | [`docs/templates/agent_documentation_template.md`](templates/agent_documentation_template.md) | Purpose, inputs, outputs, dependencies, config for one agent | Whole file → `docs/agents/<name>.md` or the agent's package README | Agent name, `config/*.yaml` keys, the `AgentOutput` fields and `provenance_trace` sources it emits |
| Data-source doc | [`docs/templates/data_source_documentation_template.md`](templates/data_source_documentation_template.md) | Provider, schema, refresh cadence, access | Whole file | Provider, credentials env var, schema, `config/data_sources.yaml` entry |
| Workflow doc | [`docs/templates/workflow_documentation_template.md`](templates/workflow_documentation_template.md) | Multi-agent workflow steps and hand-offs | Whole file | Graph (`core/engine/*_graph.py`), route (`Route` enum), confidence-tier behaviour |
| Prompt-authoring templates | [`prompt_library/templates/`](../prompt_library/templates/) | Reusable prompts for repo chores: `generating_readme.md`, `writing_unit_tests.md`, `strict_security_review.md`, `general_refactoring.md`, `root_cause_analysis.md`, … | Prompt body | Target paths, test command (`.venv/bin/python -m pytest tests/unit -q`), house rules from `AGENTS.md` |

Browse them rendered at [`docs/templates/index.html`](templates/index.html).

## 2. Brand / product pages

| Brand | Paths | Purpose | Copy | Customize |
|---|---|---|---|---|
| **Market Mayhem** (macro / credit newsletter) | [`docs/brands/market-mayhem.html`](brands/market-mayhem.html), [`public/market-mayhem/`](../public/market-mayhem/), [`showcase/dashboards/market_mayhem.html`](../showcase/dashboards/market_mayhem.html), `market_mayhem_dashboard.html`, `market_mayhem_terminal.html`, [`market-mayhem/`](../market-mayhem/) (skill, templates, standalone) | Tail-risk and contagion publication with dashboard and terminal skins | Landing page + one dashboard skin | Masthead, palette, edition cadence, data feeds, disclaimer |
| **Project Fortress** (credit underwriting) | [`docs/brands/fortress-hunt.html`](brands/fortress-hunt.html), [`public/fortress-hunt/`](../public/fortress-hunt/), `briefings/fortress_hunt*.md` | Defensive credit view; backed by `core/credit_sentinel/` | Brand page and one briefing | Sector focus, covenant set (`assets/` JSONLogic), rating scale |
| **Project Hunt** (alpha / catalyst) | Same Fortress-Hunt pages; prompt [`ADAM_v30_1_R_Hunt_and_Kill_Orchestrator_Prompt.md`](../prompt_library/AOPL-v2.0/professional_outcomes/ADAM_v30_1_R_Hunt_and_Kill_Orchestrator_Prompt.md); agents in `core/agents/` | Offensive 13F/13D catalyst scanning | Prompt + brand page | Universe, catalyst definitions, holding-period language |
| **Analyst OS** | [`showcase/analyst_os.html`](../showcase/analyst_os.html), `analyst_os_v5.html`, `analyst_os_vCredit.html`, `unified_analyst_os.html`, [`showcase/dashboards/analyst_os.html`](../showcase/dashboards/analyst_os.html) | Analyst workstation shells (general, v5 layout, credit-specialized) | One HTML file | Panels shown, default ticker list, colour tokens |
| **Portal variants** | [`showcase/portal/`](../showcase/portal/) (`index`, `analytics`, `intelligence`), [`showcase/dashboards/hub.html`](../showcase/dashboards/hub.html), `index_v30.html`, `nexus.html`, `system_brain.html`, [`index.html`](../index.html) | Entry hubs linking into the rest of the showcase | Directory (`portal/`) or one hub page | Link list, sections, hero copy |
| **Vertical showcases** | `showcase/dashboards/showcase_hedge_fund.html`, `showcase_private_equity.html`, `unified_banking.html`, `vertical.html` | Audience-specific demos | One file | Audience vocabulary, KPIs, sample portfolio |
| **Other brand pages** | `docs/brands/adam-core.html`, `adam-governance.html`, `adam-institutional.html`, `adam-research.html`, `adam-terminals.html`, `exiled-spark.html`, `logic-lab.html`; `public/adam-*`, `public/architect-infinite/` | One landing page per product line | One file + `public/styles.css` | Name, tagline, links, palette |

The full dashboard list with classifications is generated in
[`showcase/dashboards/manifest.json`](../showcase/dashboards/manifest.json) — see
[showcase/README.md](../showcase/README.md#template-families--manifest-classification).

## 3. Newsletter / report families

| Family | Paths | Count (approx.) | Purpose | Copy | Customize |
|---|---|---|---|---|---|
| **Daily Briefing** | `showcase/Daily_Briefing_YYYY_MM_DD.html` | ~100 | One-screen daily market summary | Most recent file | Date, headline, market table, sentiment line; `<title>` still says `ADAM v23.5` — update the prefix |
| **Market Pulse** | `showcase/Market_Pulse_YYYY_MM_DD.html` | ~65 | Weekly pulse / regime read | Most recent file | Regime label, cross-asset table, outlook |
| **House View** | `showcase/House_View_YYYYMMDD.html` | ~15 | Monthly house view | Most recent file | Allocation calls, conviction scale, risks |
| **Market Mayhem newsletter** | `showcase/newsletter_market_mayhem_*.html`, `showcase/Market_Mayhem_*.html`, `newsletters/market_mayhem_*.{html,md}`, `core/libraries_and_archives/newsletters/` | 60+ | Long-form macro/credit edition, including historical crisis replays (1929, 1987, 2008, 2020) | One edition HTML, or its `.md` source in `newsletters/` | Edition date, thesis, charts, sign-off |
| **Market Mayhem report templates** | [`market-mayhem/templates/`](../market-mayhem/templates/): `daily_brief.md`, `cio_brief.md`, `executive_summary.md`, `institutional_report.md`, `portfolio_risk_note.md` | 5 | Markdown skeletons for different audiences (desk, CIO, exec, institution, risk) | Whichever audience fits | Section order, length, house style; references in `market-mayhem/references/` (playbooks, glossary) |
| **Briefings** | `briefings/` | — | Fortress/Hunt and Market Mayhem briefs | One brief | Subject, data date |

Generators that produce or update these: `scripts/update_daily_briefings.py`, `scripts/generate_newsletters.py`,
`scripts/add_new_briefings.py`, `scripts/ingest_briefings.py`,
`market-mayhem/standalone/generate_standalone_publication.py`. Generated HTML should be regenerated, not
hand-edited, when a generator exists for it.

### 3a. Schema-rendered publication templates (current)

Every edition is first an `Edition` record ([`core/schemas/publication.py`](../core/schemas/publication.py),
Pydantic) written as JSON, then rendered to Markdown and HTML by
[`core/publication/render.py`](../core/publication/render.py). The HTML pages above (3) are kept as copyable
earlier generations; new editions use these templates.

| Family | Templates (`showcase/templates/publication/`) | Cadence |
|---|---|---|
| Market Mayhem | `market_mayhem.{md,html}.j2` | Weekly |
| Daily Briefing | `daily_briefing.{md,html}.j2` | Business-daily |
| Market Pulse | `market_pulse.{md,html}.j2` | Weekly |
| House View | `house_view.{md,html}.j2` | Monthly |
| Equity Research | `equity_research.{md,html}.j2` | Monthly / on event |
| Sector Outlook | `sector_outlook.{md,html}.j2` | Quarterly |
| Credit Memo | `credit_memo.{md,html}.j2` | On event |
| Deep Dive | `deep_dive.{md,html}.j2` | On event |

* Shared layout: `_base.{md,html}.j2`. Family files extend it and override blocks `kicker`,
  `snapshot_title`, `calls_title`, `before_sections`, `after_sections`, `review`.
* HTML pages load `showcase/kit/adam-kit.{css,js}` (inside the `<!-- adam-kit -->` block, so
  `scripts/apply_showcase_kit.py` stays idempotent) plus `showcase/kit/adam-publication.css`; every section
  has a `data-adam-hook`; the full record is embedded as `<script id="adam-operator-data" type="application/json">`
  so `?view=json` / Alt+J return the rendered data.
* Jinja autoescape is on; Markdown output escapes raw `<`/`>`. Editions with any `_estimated` part show the
  "Illustrative scenario — not investment advice" label.
* Regenerate the Apr–Nov 2026 backfill (synthetic, `_estimated`; see `docs/PROVENANCE.md` §2a):
  `python scripts/publications/build_scenario_path.py && python scripts/publications/backfill_editions.py`,
  then rebuild the catalogs (`showcase/README.md`). Records land in `showcase/data/publications/<family>/`,
  HTML at the family paths (`showcase/Daily_Briefing_YYYY_MM_DD.html`, `Market_Mayhem_…`, `Market_Pulse_…`,
  `House_View_…`, `Equity_Research_…`, `Sector_Outlook_…`); earlier hand-made pages are never overwritten.
* Review layer (`core/publication/review/`, thresholds in `config/publication_review.yaml`): every
  backfilled edition carries `review` = deterministic critics (missing sources, unlabelled `_estimated`,
  unsourced values, targets vs. reference / prior, reversals, unsupported calls, stale data, scenario
  probabilities), an expert-desk verdict, a red-team pass through `core/system/red_teaming_framework.py`
  (flipped scenario, fragile targets) and a weighted `CleanRoomLLMJudge` score. Pages render a
  "Bull vs. Bear vs. Red Team" panel and `showcase/kit/adam-publication.js` publishes the scores via
  `AdamKit.publish("review_scores", …)`. LLM prompts: `prompt_library/publication_review/`; the LLM path is
  off by default (`llm.enabled: false`). Seeded flawed editions: `tests/evals/datasets/publication_review.json`.
* To add a family: add it to `Family` in the schema, copy one `<family>.{md,html}.j2` pair, and extend
  `tests/unit/test_publication_schema.py` (it renders the fixture through every family).

## 4. Terminal variants

| Variant | Path | Style | Copy | Customize |
|---|---|---|---|---|
| v30.1 Institutional Financial OS | [`showcase/dashboards/adam_v30_1_institutional_financial_os.html`](../showcase/dashboards/adam_v30_1_institutional_financial_os.html) | Dense institutional | One file | Panels, data stubs |
| v30.3 Institutional Financial OS | [`showcase/dashboards/adam_v30_3_institutional_financial_os.html`](../showcase/dashboards/adam_v30_3_institutional_financial_os.html) | Refined v30.1 layout | One file | Same |
| v30.4 Precision Institutional Terminal | [`showcase/dashboards/adam_v30_4_precision_institutional_terminal.html`](../showcase/dashboards/adam_v30_4_precision_institutional_terminal.html) | Precision / minimal | One file | Same |
| v31.0 Cyber-Retro Institutional OS | [`showcase/dashboards/adam_v31_0_cyber_retro_institutional_os.html`](../showcase/dashboards/adam_v31_0_cyber_retro_institutional_os.html) | Retro CRT aesthetic | One file | Palette, fonts, scanline effects |
| Market Mayhem terminal | `showcase/dashboards/market_mayhem_terminal.html`, `terminals/market_mayhem_20260829.html` | Newsletter-branded terminal | One file | Brand, feeds |
| Dated terminal snapshots | [`terminals/`](../terminals/): `adam_terminal_20260829.html`, `main_dashboard.html`, `edgar_pipeline.html`, `global_systemic_risk.html`, `knowledge_graph.html`, `supply_chain_graph.html` | Single-purpose terminals | One file | Data date, panel focus |
| Terminal landing | [`public/adam-terminals/`](../public/adam-terminals/), [`docs/brands/adam-terminals.html`](brands/adam-terminals.html) | Index of terminal skins | Directory | Link list |

## 5. Prompt templates

| Library | Path | Status | Purpose | Copy | Customize |
|---|---|---|---|---|---|
| **AOPL v2.0** | [`prompt_library/AOPL-v2.0/`](../prompt_library/AOPL-v2.0/) | Current | `professional_outcomes/` (credit valuation, macro PD stress, Market Mayhem synthesis, v30.1-R orchestrators), `orchestration/` (consensus dispatcher, swarm roll-up), `swarm/`, `tasks/`, `workflows/`, `encyclopedic_knowledge/`, `fields_of_study/`, `independent_research/` | One prompt file | Entity, sector, horizon, output schema; keep the provenance / confidence instructions |
| **AOPL v1.0** | [`prompt_library/AOPL-v1.0/`](../prompt_library/AOPL-v1.0/) | Legacy | `analyst_os/`, `snc/`, `quantitative_analysis/`, `simulation/`, `ingestion/`, `system_architecture/`, `EACI.yaml` | As reference or for older flows | Same; prefer the v2.0 equivalent when one exists |
| Prompt viewer | [`showcase/dashboards/prompt_library_viewer.html`](../showcase/dashboards/prompt_library_viewer.html) | — | Browse prompts in a browser | — | — |

Persona prompts that match these libraries are catalogued in [`AGENTS.md` §8–§9](../AGENTS.md#8-afos-agent-system-instructions--extended-persona-catalog).

## 6. Code templates

| Template | Path | Purpose | Copy | Customize |
|---|---|---|---|---|
| System 2 agent | [`core/agents/templates/v30_template_agent.py`](../core/agents/templates/v30_template_agent.py) | `TemplateAgentV30` base: provenance header, confidence tier, JSONLogic guardrails | Subclass it; override `analyze()` | Domain logic only (see [AGENTS.md §5](../AGENTS.md#5-standard-interface)) |
| Standalone publication | [`market-mayhem/standalone/`](../market-mayhem/standalone/) | Config-driven single-file publication generator (`config.json` → HTML/PDF) | Directory | `config.json` content and assets |
| Custom build bundle | `scripts/build_adam.py` ([docs](custom_builds.md)) | Export a subset of modules as a portable bundle | Run it | Module selection |

---

## How to fork a template

1. Pick the variant closest to your audience from the tables above.
2. Copy it next to the original with a new suffix (`_<yourbrand>` or a date); don't edit shared originals
   in place if other pages link to them.
3. Change the *Customize* items. Keep data-provenance cues (sources, `_estimated` markers, disclaimers).
4. For anything under `showcase/dashboards/`, regenerate the catalog:
   `python scripts/build_dashboard_manifest.py`.
5. If the page is linked from root `index.html`, run
   `.venv/bin/python -m pytest tests/unit/test_standalone_distribution.py -q`.
