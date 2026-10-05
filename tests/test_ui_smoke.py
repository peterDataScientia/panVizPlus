from pathlib import Path


def test_editor_asset_exists():
    path = Path(__file__).resolve().parents[1] / "editor.html"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "__PANVIZ_SCENE__" in text
    assert "panVizPlus" in text


def test_compatibility_modules_import():
    import scientific_records  # noqa: F401
    import interactive_engine  # noqa: F401
    import utils  # noqa: F401
