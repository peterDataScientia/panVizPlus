from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_editor_asset_exists():
    path = ROOT / "editor.html"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "__PANVIZ_SCENE__" in text
    assert "panVizPlus" in text


def test_compatibility_modules_import():
    sys.path.insert(0, str(ROOT))
    try:
        import scientific_records  # noqa: F401
        import interactive_engine  # noqa: F401
        import utils  # noqa: F401
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)
