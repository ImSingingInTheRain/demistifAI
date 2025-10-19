# demistifAI Repository Structure

This document provides a guided tour of the repository so contributors can quickly locate UI components, data/model helpers, and supporting configuration.

## Root directory
- `AGENTS.md` – Contributor playbook describing architectural conventions, UI practices, data/model guidelines, testing expectations, and documentation norms for the whole codebase.【F:AGENTS.md†L1-L42】
- `LICENSE` – Apache 2.0 license text covering the terms for use, distribution, and contribution.【F:LICENSE†L1-L201】
- `README.md` – Narrative overview of the demistifAI Streamlit lab, including the learning goals, stage-by-stage walkthrough, and setup instructions.【F:README.md†L1-L18】
- `.gitignore` – Standard Python ignore patterns for build artefacts, environments, IDE caches, and tooling directories.【F:.gitignore†L1-L164】
- `.pre-commit-config.yaml` – Registers the CSS usage audit so pre-commit hooks can flag unused theme selectors before shipping.【F:.pre-commit-config.yaml†L1-L8】
- `.git/` – Git metadata (objects, refs, hooks) that tracks version history; normally left untouched.
- `.github/` – GitHub service configuration (see below).
- `.devcontainer/` – Development container configuration for GitHub Codespaces or VS Code remote environments (see below).
- `requirements.txt` – Application dependencies spanning Streamlit, scikit-learn, transformers, visualization, and language-detection libraries.【F:requirements.txt†L1-L14】
- `streamlit_app.py` – Streamlit entry point that imports constants, core helpers, datasets, and modeling utilities to render each lifecycle stage of the lab.【F:streamlit_app.py†L1-L160】
- `demistifai/` – Primary Python package with constants, dataset/model logic, high-level UI components, and nested subpackages for core utilities and styles (detailed below).
- `demistifai/ui/` – Unified UI toolkit housing reusable components, primitive widgets, and static assets (detailed below).
- `docs/` – Project documentation, including stage/component references and this structure map.【F:docs/stage_component_reference.md†L1-L40】
- `scripts/` – Developer tooling such as the unused CSS selector scanner used by the pre-commit hook.【F:scripts/find_unused_css.py†L1-L160】
- `pages/` – Streamlit pages that house the stage-specific flows plugged into the main app (detailed below).
- `tests/` – Lightweight regression suite that guards against import regressions in configuration and style modules.【F:tests/test_import_smoke.py†L1-L38】

## Hidden configuration
### `.github/`
- `CODEOWNERS` – Routes pull requests to the Streamlit Community Cloud team by default.【F:.github/CODEOWNERS†L1-L1】

### `.devcontainer/`
- `devcontainer.json` – Defines the Python base image, recommended VS Code extensions, automated dependency installation, and the Streamlit command executed after attaching to the container, along with forwarded port metadata.【F:.devcontainer/devcontainer.json†L1-L33】

## UI toolkit (`demistifai/ui/`)
### Components (`demistifai/ui/components/`)
- `__init__.py` – Aggregates hero, training, terminal, guardrail, and mission components so pages can import them from a single namespace.【F:demistifai/ui/components/__init__.py†L1-L87】
- `arch_demai.py` – Dataclass-driven architecture cards and styling that render the demAI system diagram within Streamlit.【F:demistifai/ui/components/arch_demai.py†L1-L146】
- `data_review.py` – Dataset balance bars, sample cards, and scoped CSS helpers for the Prepare stage review panels.【F:demistifai/ui/components/data_review.py†L1-L74】
- `guardrail_panel.py` – Streams charts and guardrail cards within a styled panel, injecting CSS once per session.【F:demistifai/ui/components/guardrail_panel.py†L1-L48】
- `intro_hero.py` – Welcome-stage hero layout, lifecycle diagram markup, and scoped quote wrappers for the EU AI Act narrative.【F:demistifai/ui/components/intro_hero.py†L1-L112】
- `mac_window.py` – Utility for generating scoped macOS-style window shells with configurable columns and theming.【F:demistifai/ui/components/mac_window.py†L1-L158】
- `overview_mission.py` – Mission briefing markup and mailbox preview helpers that style the overview stage briefing cards.【F:demistifai/ui/components/overview_mission.py†L1-L129】
- `stage_navigation.py` – Stage top grid renderer, highlight card, and previous/next CTA controls shared across lifecycle pages.【F:demistifai/ui/components/stage_navigation.py†L1-L179】
- `pii_indicators.py` – Displays responsive PII indicator tiles summarising detected token counts.【F:demistifai/ui/components/pii_indicators.py†L1-L25】
- `train_animation.py` – Plotly-powered training map animation with brand token bridging, HTML wrappers, and graceful fallbacks when the optional Plotly dependency is missing.【F:demistifai/ui/components/train_animation.py†L9-L277】
- `train_intro.py` – Training stage launchpad cards, inline notes, and CSS builders that accompany the animation panel.【F:demistifai/ui/components/train_intro.py†L1-L143】
- `terminal/` – Animated terminal namespace with scenario-specific scripts powering stage introductions and walkthroughs:
  - `boot_sequence.py` – Alternate terminal script that frames the "demAI machine" boot sequence with staged prompts and Nerd Mode hints.【F:demistifai/ui/components/terminal/boot_sequence.py†L1-L228】
  - `article3.py` – Article 3-focused terminal variation highlighting the mission pillars of demAI.【F:demistifai/ui/components/terminal/article3.py†L1-L225】
  - `data_prep.py` – Terminal storyline guiding users through dataset creation, linting, and schema expectations.【F:demistifai/ui/components/terminal/data_prep.py†L1-L209】
  - `train.py` – Stage-specific animation narrating model training, feature importance, and calibration steps.【F:demistifai/ui/components/terminal/train.py†L1-L214】
  - `evaluate.py` – Evaluation-focused script covering metrics, curves, and Nerd Mode guidance.【F:demistifai/ui/components/terminal/evaluate.py†L1-L210】
  - `use.py` – Inference stage terminal outlining routing decisions, oversight hooks, and operational prompts.【F:demistifai/ui/components/terminal/use.py†L1-L213】

### Primitives (`demistifai/ui/primitives/`)
- `__init__.py` – Re-exports frequently used primitives including EU AI quotes, mailbox tables, Nerd Mode toggles, and text helpers for easy import in the app shell.【F:demistifai/ui/primitives/__init__.py†L1-L24】
- `mailbox.py` – Mailbox table and panel renderers that keep inbox/spam displays consistent across stages.【F:demistifai/ui/primitives/mailbox.py†L1-L68】
- `popovers.py` – Lightweight guidance popover component used for inline tooltips and walkthrough callouts.【F:demistifai/ui/primitives/popovers.py†L1-L12】
- `quotes.py` – EU AI Act quote helpers powering the welcome stage hero copy.【F:demistifai/ui/primitives/quotes.py†L1-L32】
- `sections.py` – Section surface and Nerd Mode toggle primitives reused to style stage panels.【F:demistifai/ui/primitives/sections.py†L1-L64】
- `text.py` – Shared text utilities such as `shorten_text` for trimming long strings in tables.【F:demistifai/ui/primitives/text.py†L1-L11】
- `typing_quote.py` – Inline typing effect that animates and highlights Article 3 language for the welcome panels.【F:demistifai/ui/primitives/typing_quote.py†L1-L677】

### Assets (`demistifai/ui/assets/`)
- `__init__.py` – Placeholder package for icons, SVGs, or CSS fragments as they are added.【F:demistifai/ui/assets/__init__.py†L1-L3】

## Core package (`demistifai/`)
### Package root
- `__init__.py` – Re-exports constants, dataset utilities, modeling helpers, and core utility functions for convenient imports throughout the app.【F:demistifai/__init__.py†L1-L161】
- `constants.py` – Centralised application metadata, CSS themes, stage descriptors, and lifecycle assets referenced by the UI.【F:demistifai/constants.py†L1-L60】
- `dataset.py` – Drives dataset synthesis (spam/safe templates, attachments, edge cases), TypedDict-backed configs, linting/PII scanners, provenance hashing, summaries, and narrative deltas that explain configuration shifts.【F:demistifai/dataset.py†L1-L856】
- `incoming_generator.py` – Produces unlabeled inbox batches by blending spam/safe archetypes, urgency cues, links, and attachments for evaluation or live-routing demos.【F:demistifai/incoming_generator.py†L1-L185】
- `modeling.py` – Full modeling stack covering feature extraction, embedding backends, hybrid logistic pipeline, calibrators, threshold analytics, interpretability stories, and batch probability helpers used across stages.【F:demistifai/modeling.py†L1-L1002】

### `demistifai/config/`
- `__init__.py` – Re-exports stage metadata, theme CSS (via `demistifai.styles.APP_THEME_CSS`), and token policies so callers can continue importing from `demistifai.config`.【F:demistifai/config/__init__.py†L1-L16】
- `app.py` – Provides stage listings, lookup tables, and urgency terminology for navigation-aware helpers.【F:demistifai/config/app.py†L1-L8】
- `tokens.py` – Centralises PII replacement policies and chip display labels consumed by linting helpers.【F:demistifai/config/tokens.py†L1-L7】

### `demistifai/core/`
- `dataset.py` – Evaluates dataset health metrics and produces badges summarising row counts, spam ratios, and lint status for governance surfaces.【F:demistifai/core/dataset.py†L1-L52】
- `audit.py` – Appends timestamped audit entries to the session log during the Use stage.【F:demistifai/core/audit.py†L1-L11】
- `downloads.py` – Provides a Streamlit download link helper for exporting text artefacts.【F:demistifai/core/downloads.py†L1-L7】
- `language.py` – Handles optional language-detection fallbacks, aggregates language mix stats, and renders chip/caption summaries for train/test splits within Streamlit containers.【F:demistifai/core/language.py†L1-L102】
- `nav.py` – Deprecated shim that re-exports the stage navigation grid from the UI layer; import `demistifai.ui.components.stage_navigation` instead.【F:demistifai/core/nav.py†L1-L27】
- `navigation.py` – Synchronises active stage selection between query params, session state, and the renderer map and exposes the `activate_stage` helper used in `streamlit_app.py`.【F:demistifai/core/navigation.py†L1-L108】
- `routing.py` – Computes recommended or automatic routing decisions based on autonomy level, predictions, and thresholds.【F:demistifai/core/routing.py†L1-L15】
- `export.py` – Normalises batch processing logs into a pandas DataFrame ready for CSV/JSON export.【F:demistifai/core/export.py†L1-L13】
- `validation.py` – Normalises dataset labels and verifies CSV schema requirements before imports.【F:demistifai/core/validation.py†L1-L20】
- `pii.py` – Aggregates detected PII spans, formats chips/badges, applies policy-based replacements, and renders clean-up banners.【F:demistifai/core/pii.py†L1-L60】
- `guardrails.py` – Detects guardrail signals (links, caps, money, urgency), renders badge markup, and exposes helper regexes for suspicious content.【F:demistifai/core/guardrails.py†L1-L80】
- `embeddings.py` – Caches MiniLM embeddings for dataset texts via Streamlit caching to accelerate retraining flows.【F:demistifai/core/embeddings.py†L1-L11】
- `cache.py` – Streamlit cache wrappers for dataset preparation, feature extraction, and model training payloads.【F:demistifai/core/cache.py†L1-L143】
- `state.py` – Synchronises Nerd Mode "advanced knob" widget state, queues flash messages, and records pending updates in Streamlit session state.【F:demistifai/core/state.py†L1-L80】
- `session_defaults.py` – Populates Streamlit session state with baseline datasets, guardrail knobs, and adaptiveness settings on first load.【F:demistifai/core/session_defaults.py†L1-L118】
- `utils.py` – Houses rerun helper, suspicious link detection, money cue counters, caps ratio computation, and TLD checks shared across the app.【F:demistifai/core/utils.py†L1-L58】

### `demistifai/styles/`
- `css_blocks.py` – Reusable CSS snippets for PII indicators and guardrail panels.【F:demistifai/styles/css_blocks.py†L1-L61】
- `inject.py` – Deduplicated CSS injector that hashes style blocks to avoid repeated insertion.【F:demistifai/styles/inject.py†L1-L35】
- `theme.py` – Primary app theme stylesheet defining global tokens, layout spacing, and responsive section styling.【F:demistifai/styles/theme.py†L1-L80】
- `__init__.py` – Re-exports the shared `APP_THEME_CSS` string so callers can `from demistifai.styles import APP_THEME_CSS`.【F:demistifai/styles/__init__.py†L1-L6】

## Documentation (`docs/`)
- `stage_component_reference.md` – Maintains stage line ranges in `streamlit_app.py`, maps supporting modules, and sets expectations for keeping the reference current.【F:docs/stage_component_reference.md†L1-L39】
- `streamlit_nav_bar.md` – Explains how to embed the `streamlit-navigation-bar` component, covering API usage, styling hooks, optional behaviors, and recommended project structure for multi-page apps.【F:docs/streamlit_nav_bar.md†L1-L128】【F:docs/streamlit_nav_bar.md†L130-L201】
- `repository_structure.md` – (This file) describes the repository layout for new contributors.

## Stage pages (`pages/`)
- `welcome.py` – Renders the intro stage terminal, lifecycle hero window, and EU AI Act quote wrapper while wiring the next-stage handoff.【F:pages/welcome.py†L1-L66】
- `overview.py` – Sets up the overview stage grid, mission briefing panels, Nerd Mode toggle, and system snapshot window with mailbox previews.【F:pages/overview.py†L1-L203】
- `data/` – Prepare stage package with the orchestrator (`page.py`), dataset builder flows, review dashboards, PII scrubbing, session-state helpers, and HTML/markup utilities.【F:pages/data/page.py†L1-L194】【F:pages/data/builder.py†L1-L508】【F:pages/data/review.py†L1-L865】【F:pages/data/pii.py†L1-L201】【F:pages/data/dataset_io.py†L1-L118】【F:pages/data/ui.py†L1-L104】
- `train_stage/page.py` – Orchestrates the Train stage, preparing session state, launching runs, and delegating guardrail controls, storyboard rendering, and results panels to the supporting modules below.【F:pages/train_stage/page.py†L1-L307】
- `train_stage/guardrails.py` – Provides the launchpad guardrail controls and context preview meter for the numeric assist window.【F:pages/train_stage/guardrails.py†L1-L75】
- `train_stage/panels.py` – Builds the launchpad cards, dataset readiness checks, Nerd Mode controls, and advanced guardrail configuration surfaces.【F:pages/train_stage/panels.py†L1-L299】
- `train_stage/results.py` – Drives post-training storytelling, embeddings diagnostics, calibration workflows, and interpretability surfaces.【F:pages/train_stage/results.py†L1-L552】
- `train_stage/state.py` – Defines training dataclasses, executes the training pipeline, applies numeric adjustments, and persists refreshed guardrail settings.【F:pages/train_stage/state.py†L1-L323】
- `train_stage/visualizations.py` – Altair calibration chart builder layered with the reliability diagonal.【F:pages/train_stage/visualizations.py†L1-L39】
- `train_stage/` – Package consolidating shared callbacks, navigation hooks, and the entry-point re-export for the stage renderer.【F:pages/train_stage/__init__.py†L1-L5】【F:pages/train_stage/callbacks.py†L1-L69】【F:pages/train_stage/navigation.py†L1-L32】
- `train_stage/helpers/` – Modularised training helpers covering meaning map preparation and storyboard assembly shared across the stage.【F:pages/train_stage/helpers/__init__.py†L1-L23】【F:pages/train_stage/helpers/meaning_map.py†L1-L1196】【F:pages/train_stage/helpers/storyboard.py†L1-L261】
- `train_stage/helpers/guardrails.py` – Computes guardrail window bounds and constructs placeholder charts/labels for numeric assist visuals.【F:pages/train_stage/helpers/guardrails.py†L1-L91】
- `train_stage/helpers/numeric_clues.py` – Formats numeric guardrail reasons, preview chips, and post-run cards highlighting structured signals.【F:pages/train_stage/helpers/numeric_clues.py†L1-L231】
- `train_stage/helpers/sampling.py` – Utility for label-aware sampling when limiting example counts in previews.【F:pages/train_stage/helpers/sampling.py†L1-L31】
- `evaluate.py` – Evaluation dashboards for metrics, threshold management, and governance summaries.【F:pages/evaluate.py†L44-L566】
- `use.py` – Live classification console with autonomy toggles, adaptiveness, and routing diagnostics.【F:pages/use.py†L37-L489】
- `model_card.py` – Transparency surface that renders the downloadable model card and supporting dataset context.【F:pages/model_card.py†L21-L141】
- `__init__.py` – Package marker for stage modules.【F:pages/__init__.py†L1-L1】

### Shared helpers
- `animated_logo.py` – Renders the animated "demAI" hero logo with scripted typing/erasing of the mission pillars.【F:demistifai/ui/animated_logo.py†L1-L80】
- `custom_header.py` – Fixed header shell that mirrors stage navigation, bridges HTML buttons to Streamlit callbacks, and embeds the animated logo.【F:demistifai/ui/custom_header.py†L1-L80】
- `__init__.py` – Declares the UI helper namespace and re-exports the hero logo renderer alongside aggregate UI subpackages.【F:demistifai/ui/__init__.py†L1-L11】
