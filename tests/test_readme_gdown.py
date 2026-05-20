"""README gdown command guardrail (Phase 7.1).

Live VPS test surfaced that the installed gdown does not support
--remaining-ok. The README must not ship a command that crashes on
the first run."""

from __future__ import annotations

from pathlib import Path


_README = Path(__file__).resolve().parents[1] / "README.md"


def _gdown_lines() -> list[str]:
    text = _README.read_text(encoding="utf-8")
    return [
        line for line in text.splitlines()
        if line.strip().startswith("gdown ")
    ]


def test_readme_has_a_gdown_command():
    """Sanity check that the docs still tell the user how to pull the folder."""
    lines = _gdown_lines()
    assert lines, "README is missing a `gdown` command line"


def test_readme_gdown_command_does_not_require_remaining_ok():
    """Phase 7.1: the active example command must not pass
    --remaining-ok unconditionally. It can be mentioned as an
    optional flag for newer gdown forks, but the executable line
    itself must work with the default pypi gdown."""
    for line in _gdown_lines():
        assert "--remaining-ok" not in line, (
            "README gdown command still passes --remaining-ok; the "
            "installed gdown does not support it. Make it optional in "
            "prose only."
        )
