from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_is_native_entrypoint():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "analyze_structure" in text
    assert "panVizPlus-native engine" in text


def test_removed_legacy_modules_are_not_required():
    assert not (ROOT / "interactive_engine.py").exists()
    assert not (ROOT / "scientific_records.py").exists()
    assert not (ROOT / "utils.py").exists()
    assert not (ROOT / "editor.html").exists()
