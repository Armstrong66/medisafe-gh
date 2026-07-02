"""
Unit tests for the Khaya/GhanaNLP translation bridge.
"""

import json
import sys
import types

from translation import khaya


def test_ghanaian_english_does_not_require_translation():
    assert "ghanaian_en" not in khaya.TRANSLATION_REQUIRED_LANGUAGES
    assert khaya.translate_response_to_english("Please go to hospital.", "ghanaian_en") == "Please go to hospital."


def test_loads_api_key_from_credential_json(tmp_path, monkeypatch):
    credentials = tmp_path / "khaya.json"
    credentials.write_text(json.dumps({"api_key": "secret-key"}), encoding="utf-8")

    monkeypatch.delenv("KHAYA_API_KEY", raising=False)
    monkeypatch.setenv("KHAYA_CREDENTIALS_PATH", str(credentials))

    assert khaya._load_khaya_api_key() == "secret-key"


def test_extract_translation_from_common_response_shapes():
    assert khaya._extract_translation({"translatedText": "Hello"}) == "Hello"
    assert khaya._extract_translation({"data": {"translation": "Hello"}}) == "Hello"
    assert khaya._extract_translation({"translations": [{"translated_text": "Hello"}]}) == "Hello"


def test_local_fallback_uses_offline_model_loading(monkeypatch):
    calls = {}

    def fake_pipeline(task, **kwargs):
        calls["task"] = task
        calls["load_kwargs"] = kwargs

        def fake_translator(text, **translation_kwargs):
            calls["translation_text"] = text
            calls["translation_kwargs"] = translation_kwargs
            return [{"translation_text": "Go to a doctor quickly."}]

        return fake_translator

    fake_transformers = types.SimpleNamespace(pipeline=fake_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(khaya, "_local_translator", None)
    monkeypatch.setattr(khaya, "NLLB_MODEL", "local/nllb")
    monkeypatch.setattr(khaya, "NLLB_SOURCE_LANG", "aka_Latn")
    monkeypatch.setattr(khaya, "NLLB_TARGET_LANG", "eng_Latn")
    monkeypatch.setattr(khaya, "NLLB_MAX_NEW_TOKENS", 64)

    assert khaya._translate_via_local_model("Ko dokita ntɛm.") == "Go to a doctor quickly."
    assert calls["task"] == "translation"
    assert calls["load_kwargs"]["model"] == "local/nllb"
    assert calls["load_kwargs"]["local_files_only"] is True
    assert calls["translation_text"] == "Ko dokita ntɛm."
    assert calls["translation_kwargs"] == {
        "src_lang": "aka_Latn",
        "tgt_lang": "eng_Latn",
        "max_new_tokens": 64,
    }
