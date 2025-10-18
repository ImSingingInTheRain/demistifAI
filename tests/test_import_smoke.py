"""Smoke tests to guard against circular imports in configuration modules."""

from __future__ import annotations

import importlib
import unittest


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

    def test_core_pii_imports_tokens_without_cycle(self) -> None:
        pii_module = importlib.import_module("demistifai.core.pii")
        self.assertTrue(hasattr(pii_module, "render_pii_cleanup_banner"))

    def test_core_nav_exports_stage_grid(self) -> None:
        nav_module = importlib.import_module("demistifai.core.nav")
        self.assertTrue(hasattr(nav_module, "render_stage_top_grid"))
        self.assertTrue(hasattr(nav_module, "StageTopGridSlots"))


if __name__ == "__main__":  # pragma: no cover - convenience for direct execution
    unittest.main()
