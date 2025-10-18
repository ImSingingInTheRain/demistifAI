# demistifAI Repository Structure

This document provides a guided tour of the repository so contributors can quickly locate UI components, data/model helpers, and supporting configuration.

## Root directory
- `AGENTS.md` – Contributor playbook describing architectural conventions, UI practices, data/model guidelines, testing expectations, and documentation norms for the whole codebase.【F:AGENTS.md†L1-L42】
- `LICENSE` – Apache 2.0 license text covering the terms for use, distribution, and contribution.【F:LICENSE†L1-L201】
- `README.md` – Narrative overview of the demistifAI Streamlit lab, including the learning goals, stage-by-stage walkthrough, and setup instructions.【F:README.md†L1-L18】
- `.gitignore` – Standard Python ignore patterns for build artefacts, environments, IDE caches, and tooling directories.【F:.gitignore†L1-L164】
- `.git/` – Git metadata (objects, refs, hooks) that tracks version history; normally left untouched.
- `.github/` – GitHub service configuration (see below).
- `.devcontainer/` – Development container configuration for GitHub Codespaces or VS Code remote environments (see below).
- `requirements.txt` – Application dependencies spanning Streamlit, scikit-learn, transformers, visualization, and language-detection libraries.【F:requirements.txt†L1-L14】
- `streamlit_app.py` – Streamlit entry point that imports constants, core helpers, datasets, and modeling utilities to render each lifecycle stage of the lab.【F:streamlit_app.py†L1-L160】
- `demistifai/` – Primary Python package with constants, dataset/model logic, high-level UI components, and nested subpackages for core utilities and styles (detailed below).
- `demistifai/ui/` – Unified UI toolkit housing reusable components, primitive widgets, and static assets (detailed below).
- `docs/` – Project documentation, including stage/component references and this structure map.【F:docs/stage_component_reference.md†L1-L40】
- `pages/` – Streamlit pages that house the stage-specific flows plugged into the main app (detailed below).

## Hidden configuration
### `.github/`
- `CODEOWNERS` – Routes pull requests to the Streamlit Community Cloud team by default.【F:.github/CODEOWNERS†L1-L1】

### `.devcontainer/`
- `devcontainer.json` – Defines the Python base image, recommended VS Code extensions, automated dependency installation, and the Streamlit command executed after attaching to the container, along with forwarded port metadata.【F:.devcontainer/devcontainer.json†L1-L33】

## UI toolkit (`demistifai/ui/`)
### Components (`demistifai/ui/components/`)
- `arch_demai.py` – Dataclass-driven architecture cards and styling that render the demAI system diagram within Streamlit.【F:demistifai/ui/components/arch_demai.py†L1-L210】
- `control_room.py` – Stage Control Room surface that standardises headers, Nerd Mode toggles, and navigation CTAs across stages.【F:demistifai/ui/components/control_room.py†L1-L230】
- `mac_window.py` – Utility for generating scoped macOS-style window shells with configurable columns and theming.【F:demistifai/ui/components/mac_window.py†L1-L224】
- `train_animation.py` – Plotly-powered training map animation with brand token bridging, HTML wrappers, and graceful fallbacks when the optional Plotly dependency is missing.【F:demistifai/ui/components/train_animation.py†L1-L120】【F:demistifai/ui/components/train_animation.py†L205-L471】
- `terminal/` – Animated terminal namespace with shared helpers and scenario-specific scripts:
  - `terminal_base.py` – Shared helpers for rendering terminal sequences, including CSS injection, typing operations, and highlight logic.【F:demistifai/ui/components/terminal/terminal_base.py†L1-L200】
  - `classic.py` – Command-line style welcome terminal typing the EU AI Act definition and demAI guidance for the classic intro experience.【F:demistifai/ui/components/terminal/classic.py†L1-L200】
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
- `dataset.py` – Synthetic dataset generator, linting, provenance tracking, and summary utilities used in data preparation and classification stages.【F:demistifai/dataset.py†L1-L60】
- `incoming_generator.py` – Synthesises unlabeled incoming email batches and spam archetypes for the Use stage.【F:demistifai/incoming_generator.py†L1-L40】
- `modeling.py` – End-to-end modeling toolkit providing feature engineering, embedding support, hybrid logistic model definitions, evaluation helpers, and interpretability utilities.【F:demistifai/modeling.py†L1-L40】

### `demistifai/config/`
- `__init__.py` – Re-exports stage metadata, theme CSS, and token policies sourced from the canonical constants module.【F:demistifai/config/__init__.py†L1-L15】
- `app.py` – Provides stage listings, lookup tables, and urgency terminology for navigation-aware helpers.【F:demistifai/config/app.py†L1-L8】
- `styles.py` – Exposes the shared Streamlit theme CSS token for reuse in UI components.【F:demistifai/config/styles.py†L1-L7】
- `tokens.py` – Centralises PII replacement policies and chip display labels consumed by linting helpers.【F:demistifai/config/tokens.py†L1-L7】

### `demistifai/core/`
- `dataset.py` – Evaluates dataset health metrics and produces badges summarising row counts, spam ratios, and lint status for governance surfaces.【F:demistifai/core/dataset.py†L1-L52】
- `audit.py` – Appends timestamped audit entries to the session log during the Use stage.【F:demistifai/core/audit.py†L1-L11】
- `downloads.py` – Provides a Streamlit download link helper for exporting text artefacts.【F:demistifai/core/downloads.py†L1-L7】
- `language.py` – Wraps optional language-detection, summarises language mix statistics, and renders chip-based summaries for train/test splits.【F:demistifai/core/language.py†L1-L80】
- `nav.py` – Renders the stage top grid, manages previous/next navigation buttons, and keeps query parameters in sync with stage selection.【F:demistifai/core/nav.py†L1-L160】
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

### `demistifai/components/`
- `guardrail_panel.py` – Streams charts and guardrail cards within a styled panel, injecting CSS once per session.【F:demistifai/components/guardrail_panel.py†L1-L48】
- `pii_indicators.py` – Displays responsive PII indicator tiles summarising detected token counts.【F:demistifai/components/pii_indicators.py†L1-L25】

### `demistifai/styles/`
- `css_blocks.py` – Reusable CSS snippets for PII indicators and guardrail panels.【F:demistifai/styles/css_blocks.py†L1-L61】
- `inject.py` – Deduplicated CSS injector that hashes style blocks to avoid repeated insertion.【F:demistifai/styles/inject.py†L1-L36】
- `__init__.py` – Empty module placeholder to mark the package namespace.【F:demistifai/styles/__init__.py†L1-L1】

## Documentation (`docs/`)
- `stage_component_reference.md` – Maintains stage line ranges in `streamlit_app.py`, maps supporting modules, and sets expectations for keeping the reference current.【F:docs/stage_component_reference.md†L1-L39】
- `repository_structure.md` – (This file) describes the repository layout for new contributors.

## Stage pages (`pages/`)
- `welcome.py` – Intro stage UI with lifecycle hero surfaces, EU AI Act framing, and launch controls.【F:pages/welcome.py†L22-L337】
- `overview.py` – Stage control room summarising system components, mission steps, and Nerd Mode insights.【F:pages/overview.py†L25-L755】
- `data.py` – Prepare stage workflow covering dataset generation, linting, cleanup, and provenance snapshots.【F:pages/data.py†L84-L1849】
- `train_stage.py` – Full training stage implementation covering dataset prep callbacks, modeling workflows, interpretability panels, and narrative storytelling.【F:pages/train_stage.py†L117-L2215】
- `train_helpers.py` – Shared utilities for the training stage, including feature explanations, meaning maps, and Altair selection helpers.【F:pages/train_helpers.py†L1-L1857】
- `evaluate.py` – Evaluation dashboards for metrics, threshold management, and governance summaries.【F:pages/evaluate.py†L44-L566】
- `use.py` – Live classification console with autonomy toggles, adaptiveness, and routing diagnostics.【F:pages/use.py†L37-L489】
- `model_card.py` – Transparency surface that renders the downloadable model card and supporting dataset context.【F:pages/model_card.py†L21-L141】
- `__init__.py` – Package marker for stage modules.【F:pages/__init__.py†L1-L1】

### Shared helpers
- `animated_logo.py` – Renders the animated "demAI" hero logo with scripted typing/erasing of the mission pillars.【F:demistifai/ui/animated_logo.py†L1-L80】
- `custom_header.py` – Fixed header shell that mirrors stage navigation, bridges HTML buttons to Streamlit callbacks, and embeds the animated logo.【F:demistifai/ui/custom_header.py†L1-L80】
- `__init__.py` – Declares the UI helper namespace and re-exports the hero logo renderer alongside aggregate UI subpackages.【F:demistifai/ui/__init__.py†L1-L11】
