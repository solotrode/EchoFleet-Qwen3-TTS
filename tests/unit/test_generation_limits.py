import pytest

from inference.generation_limits import estimate_max_new_tokens


class TestEstimateMaxNewTokens:
    def test_short_sentence_produces_small_limit(self) -> None:
        text = "Ever heard of the Double Irish with a Dutch Sandwich?"
        tokens = estimate_max_new_tokens(
            text,
            tokens_per_second=12.0,
            words_per_second=2.6,
            max_output_seconds=20,
            max_new_tokens=512,
            min_new_tokens=96,
        )
        assert 96 <= tokens <= 512
        # Should not default to a multi-minute cap.
        assert tokens < 256

    def test_long_text_clamps_to_max(self) -> None:
        text = "word " * 5000
        tokens = estimate_max_new_tokens(
            text,
            tokens_per_second=12.0,
            words_per_second=2.6,
            max_output_seconds=60,
            max_new_tokens=256,
            min_new_tokens=96,
        )
        assert tokens == 256

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_text_uses_floor(self, text) -> None:
        tokens = estimate_max_new_tokens(
            text,  # type: ignore[arg-type]
            tokens_per_second=12.0,
            words_per_second=2.6,
            max_output_seconds=20,
            max_new_tokens=512,
            min_new_tokens=96,
        )
        assert tokens == 96
