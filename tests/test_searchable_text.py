import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "searchable_text", Path(__file__).resolve().parents[1] / "searchable_text.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_multiline_text_returns_complete_unmodified_string():
    text = "Erste Zeile\nzweite Zeile mit Umlauten äöü\n[.*+?] und Leerzeichen  "
    assert module.SearchableMultilineText().output_text(text) == (text,)
    assert module.SearchableMultilineText.RETURN_TYPES == ("STRING",)
    assert module.SearchableMultilineText.INPUT_TYPES()["required"]["text"][1]["multiline"] is True


def test_search_node_is_registered_as_string_source():
    assert module.NODE_CLASS_MAPPINGS["SearchableMultilineText"] is module.SearchableMultilineText
