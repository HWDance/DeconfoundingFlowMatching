from pathlib import Path

import deconfoundingfm
from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig


def test_public_imports_and_version():
    assert deconfoundingfm.__version__ == "0.3.3"
    assert DeconfoundingFM is not None
    assert DeconfoundingFMConfig is not None


def test_no_legacy_package_references_in_installed_source():
    root = Path(deconfoundingfm.__file__).parent
    forbidden = ("doflow", "deconfoundingfm.backends")
    for path in root.rglob("*.py"):
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"legacy reference {token!r} in {path}"
