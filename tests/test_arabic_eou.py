"""
Unit tests for ArabicTurnDetector — mocked model, no GPU required.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from livekit_plugins_arabic_turn_detector.arabic_eou import (
    ArabicTurnDetector,
    load,
)
from livekit_plugins_arabic_turn_detector.models import (
    DEFAULT_THRESHOLD,
    SUPPORTED_LANGUAGES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(mock_model, mock_tokenizer, threshold=None):
    """Instantiate ArabicTurnDetector with mocked HF model."""
    with (
        patch(
            "livekit_plugins_arabic_turn_detector.arabic_eou.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
        patch(
            "livekit_plugins_arabic_turn_detector.arabic_eou.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "livekit_plugins_arabic_turn_detector.arabic_eou.torch.cuda.is_available",
            return_value=False,
        ),
    ):
        if threshold is not None:
            return ArabicTurnDetector(unlikely_threshold=threshold)
        return ArabicTurnDetector()


def _set_eou_probability(mock_model, prob: float, eou_token_id: int = 151643):
    """Configure mock model to return a specific EOU probability."""
    vocab_size = 151936
    logits = torch.zeros(1, 5, vocab_size)
    # Set logit so softmax yields approximately `prob` for eou token
    # softmax([x, 0, 0, ...]) ≈ exp(x) / (exp(x) + vocab_size - 1)
    if prob > 0.99:
        logits[0, -1, eou_token_id] = 20.0
    elif prob < 0.01:
        logits[0, -1, eou_token_id] = -20.0
    else:
        # Rough approximation
        import math

        logits[0, -1, eou_token_id] = math.log(prob / (1 - prob) * (vocab_size - 1))
    mock_model.return_value.logits = logits


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestDetectorInit:
    def test_creates_with_defaults(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        assert detector.model == "arabic"
        assert detector.provider == "hf"
        assert detector._default_threshold == DEFAULT_THRESHOLD

    def test_creates_with_custom_threshold(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer, threshold=0.98)
        assert detector._default_threshold == 0.98

    def test_languages_populated(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        for lang in SUPPORTED_LANGUAGES:
            assert lang in detector._languages

    def test_uses_cpu_when_no_cuda(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        assert detector._device == "cpu"


# ---------------------------------------------------------------------------
# Language support tests
# ---------------------------------------------------------------------------


class TestLanguageSupport:
    @pytest.mark.asyncio
    async def test_supports_generic_arabic(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        threshold = await detector.unlikely_threshold("ar")
        assert threshold is not None

    @pytest.mark.asyncio
    async def test_supports_saudi_arabic(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        threshold = await detector.unlikely_threshold("ar-SA")
        assert threshold is not None

    @pytest.mark.asyncio
    async def test_rejects_english(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        threshold = await detector.unlikely_threshold("en")
        assert threshold is None

    @pytest.mark.asyncio
    async def test_rejects_none_language(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        threshold = await detector.unlikely_threshold(None)
        assert threshold is None

    @pytest.mark.asyncio
    async def test_fallback_to_base_language(self, mock_model, mock_tokenizer):
        """ar-XX should fall back to 'ar' base language."""
        detector = _make_detector(mock_model, mock_tokenizer)
        threshold = await detector.unlikely_threshold("ar-YE")  # Yemen not in list
        # Falls back to "ar" which IS in the list
        assert threshold is not None

    @pytest.mark.asyncio
    async def test_custom_threshold_overrides(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer, threshold=0.95)
        threshold = await detector.unlikely_threshold("ar-SA")
        assert threshold == 0.95


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_predict_eou_high_probability(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        _set_eou_probability(mock_model, 0.95)
        prob = detector._predict_eou("السلام عليكم")
        assert 0.0 <= prob <= 1.0
        assert prob > 0.5

    def test_predict_eou_low_probability(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        _set_eou_probability(mock_model, 0.05)
        prob = detector._predict_eou("أبغى")
        assert 0.0 <= prob <= 1.0
        assert prob < 0.5

    def test_predict_eou_returns_float(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        prob = detector._predict_eou("مرحبا")
        assert isinstance(prob, float)

    def test_predict_eou_in_valid_range(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        for text in ["مرحبا", "كيف الحال", "أبغى أسأل"]:
            prob = detector._predict_eou(text)
            assert 0.0 <= prob <= 1.0

    @pytest.mark.asyncio
    async def test_predict_end_of_turn_empty_context(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        ctx = MagicMock()
        ctx.items = []
        prob = await detector.predict_end_of_turn(ctx)
        assert prob == 0.0

    @pytest.mark.asyncio
    async def test_predict_end_of_turn_short_text(self, mock_model, mock_tokenizer):
        """Text shorter than 2 chars should return 0.0."""
        detector = _make_detector(mock_model, mock_tokenizer)
        item = MagicMock()
        item.type = "message"
        item.role = "user"
        item.text_content = "ا"  # Single char
        ctx = MagicMock()
        ctx.items = [item]
        prob = await detector.predict_end_of_turn(ctx)
        assert prob == 0.0

    @pytest.mark.asyncio
    async def test_predict_end_of_turn_valid_message(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        _set_eou_probability(mock_model, 0.9)
        item = MagicMock()
        item.type = "message"
        item.role = "user"
        item.text_content = "السلام عليكم كيف حالك"
        ctx = MagicMock()
        ctx.items = [item]
        prob = await detector.predict_end_of_turn(ctx)
        assert 0.0 <= prob <= 1.0

    @pytest.mark.asyncio
    async def test_predict_end_of_turn_ignores_system_messages(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        system_item = MagicMock()
        system_item.type = "message"
        system_item.role = "system"
        system_item.text_content = "You are a helpful assistant"
        ctx = MagicMock()
        ctx.items = [system_item]
        prob = await detector.predict_end_of_turn(ctx)
        assert prob == 0.0

    @pytest.mark.asyncio
    async def test_predict_end_of_turn_error_returns_safe_default(self, mock_model, mock_tokenizer):
        detector = _make_detector(mock_model, mock_tokenizer)
        # Make model raise an error
        mock_model.side_effect = RuntimeError("CUDA OOM")
        item = MagicMock()
        item.type = "message"
        item.role = "user"
        item.text_content = "مرحبا كيف حالك"
        ctx = MagicMock()
        ctx.items = [item]
        prob = await detector.predict_end_of_turn(ctx)
        assert prob == 0.5  # Safe default on error


# ---------------------------------------------------------------------------
# load() factory tests
# ---------------------------------------------------------------------------


class TestLoadFactory:
    def test_load_returns_detector(self, mock_model, mock_tokenizer):
        with (
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            detector = load()
            assert isinstance(detector, ArabicTurnDetector)

    def test_load_with_threshold(self, mock_model, mock_tokenizer):
        with (
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            detector = load(threshold=0.98)
            assert detector._default_threshold == 0.98

    def test_load_backward_compat_threshold(self, mock_model, mock_tokenizer):
        with (
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "livekit_plugins_arabic_turn_detector.arabic_eou.torch.cuda.is_available",
                return_value=False,
            ),
        ):
            detector = load(unlikely_threshold=0.97)
            assert detector._default_threshold == 0.97
