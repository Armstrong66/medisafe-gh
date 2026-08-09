from models.router import (
    DEFAULT_GEMINI_FALLBACK_MODELS,
    DEFAULT_GEMINI_MODEL,
    _is_retryable_gemini_error,
    build_prompt_with_language_instruction,
)


def test_gemini_defaults_use_current_flash_target():
    default_fallbacks = DEFAULT_GEMINI_FALLBACK_MODELS.split(",")

    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"
    assert "gemini-1.5-flash" not in [DEFAULT_GEMINI_MODEL, *default_fallbacks]
    assert "gemini-2.0-flash" not in default_fallbacks


class TestGeminiRetryClassification:
    def test_hard_free_tier_quota_is_not_retryable(self):
        error = RuntimeError(
            "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            "limit: 0, model: gemini-2.0-flash. Please check your plan and billing details."
        )

        assert _is_retryable_gemini_error(error) is False

    def test_temporary_resource_exhausted_remains_retryable(self):
        error = RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 28s.")

        assert _is_retryable_gemini_error(error) is True


class TestLanguageInstruction:
    def test_twi_instruction_explicitly_blocks_english(self):
        prompt = build_prompt_with_language_instruction("Wo ho te sen?", "twi")

        assert "Respond only in Twi/Akan" in prompt
        assert "Do not answer in English" in prompt

    def test_english_prompt_is_unchanged(self):
        prompt = "How are you?"

        assert build_prompt_with_language_instruction(prompt, "english") == prompt
