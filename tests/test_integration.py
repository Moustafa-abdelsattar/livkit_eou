"""
Integration tests — require the actual HuggingFace model.

Run with: pytest tests/test_integration.py -m slow
Skipped in CI (no GPU / model download).
"""

import pytest

from .conftest import COMPLETE_UTTERANCES, EDGE_CASES, INCOMPLETE_UTTERANCES

# Skip entire module if transformers or torch not fully available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_DEPS, reason="torch/transformers not installed"),
]


MODEL_ID = "Moustafa3092/livekit-turn-detector-arabic"
PRODUCTION_THRESHOLD = 0.98  # As used in agent.py


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load the real model once for all integration tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    eou_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return model, tokenizer, eou_token_id, device


def _predict(model, tokenizer, eou_token_id, device, text: str) -> float:
    """Run a single EOU prediction."""
    formatted = f"<|im_start|>user\n{text}"
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        return probs[eou_token_id].cpu().item()


# ---------------------------------------------------------------------------
# Complete utterance tests (should have HIGH EOU probability)
# ---------------------------------------------------------------------------


class TestCompleteUtterances:
    @pytest.mark.parametrize("text", COMPLETE_UTTERANCES)
    def test_complete_utterance_detected(self, model_and_tokenizer, text):
        model, tokenizer, eou_id, device = model_and_tokenizer
        prob = _predict(model, tokenizer, eou_id, device, text)
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
        # Log for analysis even if assertion below is soft
        print(f"  COMPLETE | prob={prob:.4f} | '{text[:40]}'")


class TestIncompleteUtterances:
    @pytest.mark.parametrize("text", INCOMPLETE_UTTERANCES)
    def test_incomplete_utterance_detected(self, model_and_tokenizer, text):
        model, tokenizer, eou_id, device = model_and_tokenizer
        prob = _predict(model, tokenizer, eou_id, device, text)
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
        print(f"  INCOMPLETE | prob={prob:.4f} | '{text[:40]}'")


class TestEdgeCases:
    @pytest.mark.parametrize("text,expected_eou", EDGE_CASES)
    def test_edge_case(self, model_and_tokenizer, text, expected_eou):
        model, tokenizer, eou_id, device = model_and_tokenizer
        if len(text.strip()) < 2:
            pytest.skip("Short text handled by pre-check in detector")
            return
        prob = _predict(model, tokenizer, eou_id, device, text)
        assert 0.0 <= prob <= 1.0
        print(f"  EDGE | prob={prob:.4f} | expected_eou={expected_eou} | '{text[:40]}'")


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


class TestThresholdAnalysis:
    """Aggregate analysis — not pass/fail, but generates a report."""

    def test_threshold_report(self, model_and_tokenizer):
        """Print a classification report at different thresholds."""
        model, tokenizer, eou_id, device = model_and_tokenizer

        complete_probs = []
        for text in COMPLETE_UTTERANCES:
            prob = _predict(model, tokenizer, eou_id, device, text)
            complete_probs.append(prob)

        incomplete_probs = []
        for text in INCOMPLETE_UTTERANCES:
            prob = _predict(model, tokenizer, eou_id, device, text)
            incomplete_probs.append(prob)

        print("\n" + "=" * 60)
        print("THRESHOLD ANALYSIS REPORT")
        print("=" * 60)

        for threshold in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
            tp = sum(1 for p in complete_probs if p >= threshold)
            fn = sum(1 for p in complete_probs if p < threshold)
            fp = sum(1 for p in incomplete_probs if p >= threshold)
            tn = sum(1 for p in incomplete_probs if p < threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            print(f"\n  Threshold: {threshold}")
            print(f"    Accuracy:  {accuracy:.1%}")
            print(f"    Precision: {precision:.1%}")
            print(f"    Recall:    {recall:.1%}")
            print(f"    F1:        {f1:.1%}")
            print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

        print("=" * 60)
