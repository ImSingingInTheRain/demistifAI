"""Smoke tests to guard against circular imports in configuration modules."""

from __future__ import annotations

import importlib
import inspect
import unittest
import warnings


class ConfigImportSmokeTests(unittest.TestCase):
    """Ensure key configuration modules remain importable."""

    def test_styles_theme_exposes_css(self) -> None:
        theme = importlib.import_module("demistifai.styles.theme")
        css = getattr(theme, "APP_THEME_CSS", "")
        self.assertIn("<style>", css)
        self.assertIn(".stApp", css)

    def test_tokens_exports_are_shared(self) -> None:
        tokens = importlib.import_module("demistifai.config.tokens")
        config_pkg = importlib.import_module("demistifai.config")

        self.assertEqual(tokens.TOKEN_POLICY["email"], "{{EMAIL}}")
        self.assertIs(tokens.TOKEN_POLICY, config_pkg.TOKEN_POLICY)
        self.assertEqual(tokens.PII_DISPLAY_LABELS, config_pkg.PII_DISPLAY_LABELS)
        self.assertEqual(tokens.PII_CHIP_CONFIG, config_pkg.PII_CHIP_CONFIG)

    def test_pii_helpers_split_between_core_and_ui(self) -> None:
        pii_core = importlib.import_module("demistifai.core.pii")
        self.assertTrue(hasattr(pii_core, "summarize_pii_counts"))

        pii_ui = importlib.import_module("demistifai.ui.components.pii")
        self.assertTrue(hasattr(pii_ui, "render_pii_cleanup_banner"))

    def test_stage_navigation_component_exports_stage_grid(self) -> None:
        nav_component = importlib.import_module("demistifai.ui.components.stage_navigation")
        self.assertTrue(hasattr(nav_component, "render_stage_top_grid"))
        render_fn = getattr(nav_component, "render_stage_top_grid")
        sig = inspect.signature(render_fn)
        self.assertIn(
            sig.return_annotation,
            (inspect.Signature.empty, None, "None"),
        )

    def test_core_nav_reexports_with_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            nav_module = importlib.import_module("demistifai.core.nav")

        self.assertTrue(hasattr(nav_module, "render_stage_top_grid"))
        self.assertTrue(
            any(isinstance(w.message, DeprecationWarning) for w in caught),
            "Expected DeprecationWarning when importing demistifai.core.nav",
        )


if __name__ == "__main__":  # pragma: no cover - convenience for direct execution
    unittest.main()
