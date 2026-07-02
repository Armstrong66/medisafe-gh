"""
tests/test_language_id.py — Unit tests for scorer/language_id.py
Owner: D  |  MediSafe-GH · Africa AI Safety Prize 2026

These tests mock fasttext's model loading entirely — no network access,
no real model download, no dependency on fasttext's actual accuracy.
They test OUR wrapper logic (label parsing, caching, empty-text handling),
not fasttext's underlying classification quality.

Run with: pytest tests/test_language_id.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

import scorer.language_id as language_id


class FakeFasttextModel:
    """Stand-in for a loaded fasttext model — avoids any real model file."""

    def __init__(self, label: str, confidence: float = 0.95):
        self._label = label
        self._confidence = confidence

    def predict(self, text, k=1):
        return ([f"__label__{self._label}"], [self._confidence])


class FakeFasttextModelWithNumpy2Bug:
    """Mimics fasttext-wheel 0.9.2's NumPy 2 copy=False failure."""

    class Binding:
        def predict(self, text, k, threshold, on_unicode_error):
            return [(0.91, "__label__en")]

    f = Binding()

    def predict(self, text, k=1):
        raise ValueError("Unable to avoid copy while creating an array as requested.")


class TestDetectResponseLanguage:

    def setup_method(self):
        # Reset the module-level singleton between tests so mocks don't leak
        language_id._model = None

    def teardown_method(self):
        language_id._model = None

    @patch.object(language_id, "_load_model")
    def test_detects_twi(self, mock_load):
        mock_load.return_value = FakeFasttextModel("tw")
        result = language_id.detect_response_language("Wo ho yɛ den paa.")
        assert result == "twi"

    @patch.object(language_id, "_load_model")
    def test_detects_english(self, mock_load):
        mock_load.return_value = FakeFasttextModel("en")
        result = language_id.detect_response_language("You should see a doctor.")
        assert result == "english"

    @patch.object(language_id, "_load_model")
    def test_other_languages_collapse_to_english(self, mock_load):
        """
        Per module docstring: anything that isn't Twi collapses to "english"
        — this project only distinguishes twi vs. not-twi today. A French
        or Ewe detection should still route as "english" (not crash, not
        a third bucket that downstream code doesn't expect).
        """
        mock_load.return_value = FakeFasttextModel("fr")
        result = language_id.detect_response_language("Bonjour, allez voir un médecin.")
        assert result == "english"

    def test_empty_text_returns_english_without_loading_model(self):
        # Should short-circuit before ever calling _load_model — no model
        # load needed for an empty string, and no risk of a network call
        # in a context where empty responses are common (e.g. API errors).
        with patch.object(language_id, "_load_model") as mock_load:
            result = language_id.detect_response_language("")
            assert result == "english"
            mock_load.assert_not_called()

    @patch.object(language_id, "_load_model")
    def test_whitespace_only_text_returns_english(self, mock_load):
        result = language_id.detect_response_language("   \n\n  ")
        assert result == "english"
        mock_load.assert_not_called()

    @patch.object(language_id, "_load_model")
    def test_newlines_are_replaced_before_prediction(self, mock_load):
        """fasttext's predict() cannot handle embedded newlines — confirm we strip them."""
        fake_model = FakeFasttextModel("en")
        fake_model.predict = MagicMock(return_value=(["__label__en"], [0.9]))
        mock_load.return_value = fake_model

        language_id.detect_response_language("Line one.\nLine two.")

        call_args = fake_model.predict.call_args[0]
        assert "\n" not in call_args[0]


class TestDetectResponseLanguageWithConfidence:

    def setup_method(self):
        language_id._model = None

    def teardown_method(self):
        language_id._model = None

    @patch.object(language_id, "_load_model")
    def test_returns_label_and_confidence(self, mock_load):
        mock_load.return_value = FakeFasttextModel("tw", confidence=0.87)
        result, confidence = language_id.detect_response_language_with_confidence(
            "Kɔ dokita ntɛm."
        )
        assert result == "twi"
        assert confidence == pytest.approx(0.87)

    def test_empty_text_returns_full_confidence(self):
        result, confidence = language_id.detect_response_language_with_confidence("")
        assert result == "english"
        assert confidence == 1.0

    @patch.object(language_id, "_load_model")
    def test_numpy2_fasttext_copy_false_fallback(self, mock_load):
        mock_load.return_value = FakeFasttextModelWithNumpy2Bug()
        result, confidence = language_id.detect_response_language_with_confidence(
            "You should see a doctor."
        )
        assert result == "english"
        assert confidence == pytest.approx(0.91)


class TestModelDownload:
    """
    Confirm the download-on-first-use logic behaves correctly without
    actually hitting the network — these test OUR retry/error-wrapping
    logic, not the real download.
    """

    def test_skips_download_if_file_exists(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "lid.176.ftz"
        fake_path.write_text("fake model content")
        monkeypatch.setattr(language_id, "DEFAULT_MODEL_PATH", fake_path)

        with patch("urllib.request.urlretrieve") as mock_download:
            result = language_id._ensure_model_downloaded()
            mock_download.assert_not_called()
            assert result == fake_path

    def test_raises_clear_error_on_download_failure(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "missing" / "lid.176.ftz"
        monkeypatch.setattr(language_id, "DEFAULT_MODEL_PATH", fake_path)
        monkeypatch.setattr(language_id, "DEFAULT_MODEL_DIR", fake_path.parent)

        with patch("urllib.request.urlretrieve", side_effect=OSError("network blocked")):
            with pytest.raises(RuntimeError, match="Could not download fasttext"):
                language_id._ensure_model_downloaded()
