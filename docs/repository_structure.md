# demistifAI Repository Structure

This document provides a guided tour of the repository so contributors can quickly locate UI components, data/model helpers, and supporting configuration.

## Root directory
- `AGENTS.md` – Contributor playbook describing architectural conventions, UI practices, data/model guidelines, testing expectations, and documentation norms for the whole codebase.【F:AGENTS.md†L1-L41】
- `LICENSE` – Apache 2.0 license text covering the terms for use, distribution, and contribution.【F:LICENSE†L1-L40】
- `README.md` – Narrative overview of the demistifAI Streamlit lab, including the learning goals, stage-by-stage walkthrough, and setup instructions.【F:README.md†L1-L78】
- `.gitignore` – Standard Python ignore patterns for build artefacts, environments, IDE caches, and tooling directories.【F:.gitignore†L1-L124】
- `.git/` – Git metadata (objects, refs, hooks) that tracks version history; normally left untouched.
- `.github/` – GitHub service configuration (see below).
- `.devcontainer/` – Development container configuration for GitHub Codespaces or VS Code remote environments (see below).
- `requirements.txt` – Application dependencies spanning Streamlit, scikit-learn, transformers, visualization, and language-detection libraries.【F:requirements.txt†L1-L12】
- `streamlit_app.py` – Streamlit entry point that imports constants, core helpers, datasets, and modeling utilities to render each lifecycle stage of the lab.【F:streamlit_app.py†L1-L80】
- `demistifai/` – Primary Python package with constants, dataset/model logic, high-level UI components, and nested subpackages for core utilities and styles (detailed below).
- `demistifai/ui_components/` – Shared presentation components for the welcome experience and Stage Control Room (detailed below).
- `demistifai/ui/` – Lightweight UI widgets that complement the main components (detailed below).
- `docs/` – Project documentation, including stage/component references and this structure map.【F:docs/stage_component_reference.md†L1-L39】
- `stages/` – Stage-specific Streamlit flows that plug into the main app (detailed below).

## Hidden configuration
### `.github/`
- `CODEOWNERS` – Routes pull requests to the Streamlit Community Cloud team by default.【F:.github/CODEOWNERS†L1-L1】

### `.devcontainer/`
- `devcontainer.json` – Defines the Python base image, recommended VS Code extensions, automated dependency installation, and the Streamlit command executed after attaching to the container, along with forwarded port metadata.【F:.devcontainer/devcontainer.json†L1-L26】

## Application components (`demistifai/ui_components/`)
- `arch_demai.py` – Dataclass-driven architecture cards and styling that render the demAI system diagram within Streamlit.【F:demistifai/ui_components/arch_demai.py†L1-L75】
- `cmd_overview.py` – Animated command-line style welcome terminal that types the EU AI Act definition and demAI guidance for the classic intro experience.【F:demistifai/ui_components/cmd_overview.py†L1-L76】
- `cmd_overview_new.py` – Alternate terminal script that frames the "demAI machine" boot sequence with staged prompts and Nerd Mode hints.【F:demistifai/ui_components/cmd_overview_new.py†L1-L79】
- `cmd_welcome.py` – Variation of the animated terminal emphasising the Article 3 definition and demAI's mission pillars.【F:demistifai/ui_components/cmd_welcome.py†L1-L79】
- `components_cmd.py` – Shared helpers for rendering terminal sequences, including CSS injection, typing operations, and highlight logic.【F:demistifai/ui_components/components_cmd.py†L1-L118】
- `components_mac.py` – Utility for generating scoped macOS-style window shells with configurable columns and theming.【F:demistifai/ui_components/components_mac.py†L1-L80】
- `stage_control_room.py` – Reusable Stage Control Room surface that standardises headers, Nerd Mode toggles, and navigation CTAs across stages.【F:demistifai/ui_components/stage_control_room.py†L1-L120】
- `ui_command_grid.py` – Welcome-stage layout combining the animated terminal with a typing quote panel and responsive styling.【F:demistifai/ui_components/ui_command_grid.py†L1-L80】
- `ui_typing_quote.py` – Inline typing effect that animates and highlights a key EU AI Act sentence for the welcome panels.【F:demistifai/ui_components/ui_typing_quote.py†L1-L120】

## Core package (`demistifai/`)
### Package root
- `__init__.py` – Re-exports constants, dataset utilities, modeling helpers, and core utility functions for convenient imports throughout the app.【F:demistifai/__init__.py†L1-L80】
- `constants.py` – Centralised application metadata, CSS themes, stage descriptors, and lifecycle assets referenced by the UI.【F:demistifai/constants.py†L1-L80】
- `dataset.py` – Synthetic dataset generator, linting, provenance tracking, and summary utilities used in data preparation and classification stages.【F:demistifai/dataset.py†L1-L58】
- `incoming_generator.py` – Synthesises unlabeled incoming email batches and spam archetypes for the Use stage.【F:demistifai/incoming_generator.py†L1-L80】
- `modeling.py` – End-to-end modeling toolkit providing feature engineering, embedding support, hybrid logistic model definitions, evaluation helpers, and interpretability utilities.【F:demistifai/modeling.py†L1-L80】

### `demistifai/core/`
- `constants.py` – Mirrors key stage and urgency metadata plus token replacement policies and PII display configuration for core helpers.【F:demistifai/core/constants.py†L1-L51】
- `dataset.py` – Evaluates dataset health metrics and produces badges summarising row counts, spam ratios, and lint status for governance surfaces.【F:demistifai/core/dataset.py†L1-L52】
- `audit.py` – Appends timestamped audit entries to the session log during the Use stage.【F:demistifai/core/audit.py†L1-L11】
- `downloads.py` – Provides a Streamlit download link helper for exporting text artefacts.【F:demistifai/core/downloads.py†L1-L7】
- `language.py` – Wraps optional language-detection, summarises language mix statistics, and renders chip-based summaries for train/test splits.【F:demistifai/core/language.py†L1-L102】
- `nav.py` – Renders the stage top grid, manages previous/next navigation buttons, and keeps query parameters in sync with stage selection.【F:demistifai/core/nav.py†L1-L112】
- `routing.py` – Computes recommended or automatic routing decisions based on autonomy level, predictions, and thresholds.【F:demistifai/core/routing.py†L1-L15】
- `export.py` – Normalises batch processing logs into a pandas DataFrame ready for CSV/JSON export.【F:demistifai/core/export.py†L1-L13】
- `validation.py` – Normalises dataset labels and verifies CSV schema requirements before imports.【F:demistifai/core/validation.py†L1-L20】
- `pii.py` – Aggregates detected PII spans, formats chips/badges, applies policy-based replacements, and renders clean-up banners.【F:demistifai/core/pii.py†L1-L80】
- `guardrails.py` – Detects guardrail signals (links, caps, money, urgency), renders badge markup, and exposes helper regexes for suspicious content.【F:demistifai/core/guardrails.py†L1-L80】
- `embeddings.py` – Caches MiniLM embeddings for dataset texts via Streamlit caching to accelerate retraining flows.【F:demistifai/core/embeddings.py†L1-L11】
- `state.py` – Synchronises Nerd Mode "advanced knob" widget state, queues flash messages, and records pending updates in Streamlit session state.【F:demistifai/core/state.py†L1-L89】
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

## Stages (`stages/`)
- `train_stage.py` – Full training stage implementation covering dataset prep callbacks, modeling workflows, interpretability panels, and narrative storytelling.【F:stages/train_stage.py†L1-L75】
- `train_helpers.py` – Shared utilities for the training stage, including feature explanations, meaning maps, and Altair selection helpers.【F:stages/train_helpers.py†L1-L80】
- `__init__.py` – Package marker for stage modules.【F:stages/__init__.py†L1-L1】

## UI helpers (`demistifai/ui/`)
- `animated_logo.py` – Renders the animated "demAI" hero logo with scripted typing/erasing of the mission pillars.【F:demistifai/ui/animated_logo.py†L1-L80】
- `__init__.py` – Declares the UI helper namespace and re-exports the hero logo renderer.【F:demistifai/ui/__init__.py†L1-L5】
