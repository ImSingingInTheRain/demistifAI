#!/usr/bin/env python3
"""Find unused CSS class selectors defined in APP_THEME_CSS."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demistifai.styles.theme import APP_THEME_CSS

CLASS_PATTERN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_-]*)")
DEFAULT_SEARCH_PATHS: Sequence[str] = ("demistifai", "pages", "streamlit_app.py")
SKIP_DIRS = {".git", "__pycache__"}
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
}
ALWAYS_USED_CLASSES = {
    # Streamlit emits these classes at runtime even though they do not appear in
    # the repository's markup.
    "stApp",
    "stTabs",
    "stDataFrame",
    "stAlert",
    "stButton",
    "stSelectbox",
    "stMultiSelect",
    "stRadio",
    "stCheckbox",
    "stDateInput",
    "stTextInput",
    "stNumberInput",
    "block-container",
    # Dataset health badges are assembled dynamically in pages/data/review.py.
    "dataset-health-status--good",
    "dataset-health-status--warn",
    "dataset-health-status--risk",
    "dataset-health-status--neutral",
}


def extract_class_names(css: str) -> list[str]:
    """Return unique class selectors defined in the CSS snippet."""

    classes = OrderedDict.fromkeys(CLASS_PATTERN.findall(css))
    return list(classes.keys())


THEME_PATH = (REPO_ROOT / "demistifai" / "styles" / "theme.py").resolve()


def iter_search_files(root: Path, search_paths: Sequence[str]) -> Iterable[Path]:
    """Yield candidate files that may reference CSS classes."""

    for rel in search_paths:
        path = (root / rel).resolve()
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_dir():
                    if candidate.name in SKIP_DIRS:
                        continue
                    # Skip nested directories handled by rglob automatically
                    continue
                if candidate.suffix.lower() in SKIP_SUFFIXES:
                    continue
                yield candidate
        elif path.is_file():
            if path.suffix.lower() not in SKIP_SUFFIXES:
                yield path


def class_is_used(class_name: str, files: Iterable[Path]) -> bool:
    """Return True if *class_name* is referenced in any of the files."""

    pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(class_name)}(?![A-Za-z0-9_-])")
    for candidate in files:
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        if not pattern.search(text):
            continue
        # Avoid counting CSS references embedded in APP_THEME_CSS itself
        if candidate.resolve() == THEME_PATH and "APP_THEME_CSS" in text:
            continue
        return True
    return False


def find_unused_classes(root: Path, search_paths: Sequence[str]) -> dict[str, list[str]]:
    """Return mapping with the unused classes and those that are referenced."""

    classes = extract_class_names(APP_THEME_CSS)
    unused: list[str] = []
    used: list[str] = []

    # Materialise search file list once so we can iterate multiple times
    files = list(iter_search_files(root, search_paths))

    for cls in classes:
        if cls in ALWAYS_USED_CLASSES:
            used.append(cls)
            continue
        if class_is_used(cls, files):
            used.append(cls)
        else:
            unused.append(cls)

    return {"used": used, "unused": unused}


def format_report(result: dict[str, list[str]], *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(result, indent=2, sort_keys=True)

    lines = [
        "CSS selector usage report",
        f"  Total selectors: {len(result['used']) + len(result['unused'])}",
        f"  Referenced selectors: {len(result['used'])}",
        f"  Unused selectors: {len(result['unused'])}",
    ]
    if result["unused"]:
        lines.append("\nUnused selectors:")
        lines.extend(f"  - {cls}" for cls in result["unused"])
    else:
        lines.append("\nNo unused selectors detected. ✅")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-path",
        action="append",
        default=list(DEFAULT_SEARCH_PATHS),
        help="Relative path to scan for selector usage (defaults may be specified multiple times).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON.",
    )
    parser.add_argument(
        "--fail-on-unused",
        action="store_true",
        help="Return a non-zero exit code when unused selectors are present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    result = find_unused_classes(repo_root, tuple(args.search_path))
    print(format_report(result, as_json=args.json))

    if args.fail_on_unused and result["unused"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
