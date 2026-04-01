"""
Model Evaluation Script for Arabic Turn Detector

Evaluates the fine-tuned EOU model on Saudi dialect test data
with proper ML metrics: accuracy, precision, recall, F1, confusion matrix.

Usage:
    python evaluate.py                    # Full evaluation
    python evaluate.py --threshold 0.98   # Evaluate at specific threshold
    python evaluate.py --sweep            # Sweep thresholds 0.1-0.99
"""

import argparse
import json
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Moustafa3092/livekit-turn-detector-arabic"

# ---------------------------------------------------------------------------
# Test dataset — Saudi dialect utterances with ground truth labels
# ---------------------------------------------------------------------------

# label: True = complete utterance (EOU), False = incomplete (not EOU)
EVAL_DATASET = [
    # --- Complete utterances (should predict EOU) ---
    {"text": "السلام عليكم", "label": True, "category": "greeting"},
    {"text": "كيف حالك اليوم؟", "label": True, "category": "question"},
    {"text": "أبغى أحجز موعد يوم الخميس", "label": True, "category": "request"},
    {"text": "شكراً جزيلاً على مساعدتك", "label": True, "category": "closing"},
    {"text": "لا ما أبغى شي ثاني", "label": True, "category": "closing"},
    {"text": "وش سعر الباقة الشهرية؟", "label": True, "category": "question"},
    {"text": "خلاص تمام", "label": True, "category": "closing"},
    {"text": "الله يعطيك العافية", "label": True, "category": "closing"},
    {"text": "أبغى أكلم المدير لو سمحت", "label": True, "category": "request"},
    {"text": "ما عندي أي سؤال ثاني", "label": True, "category": "closing"},
    {"text": "إيه صحيح كلامك", "label": True, "category": "confirmation"},
    {"text": "طيب خلنا نكمل بكرة", "label": True, "category": "closing"},
    {"text": "وين أقدر ألاقي الفرع الثاني؟", "label": True, "category": "question"},
    {"text": "تمام فهمت الحين", "label": True, "category": "confirmation"},
    {"text": "ما عليك أمر", "label": True, "category": "closing"},
    {"text": "أنا موافق على العرض", "label": True, "category": "confirmation"},
    {"text": "متى يفتح المحل؟", "label": True, "category": "question"},
    {"text": "خلاص ما أبغى أكمل", "label": True, "category": "closing"},
    {"text": "الحمد لله تمام", "label": True, "category": "confirmation"},
    {"text": "أبغى أغير الباقة حقتي", "label": True, "category": "request"},
    # --- Incomplete utterances (should NOT predict EOU) ---
    {"text": "أبغى", "label": False, "category": "incomplete"},
    {"text": "يعني أنا", "label": False, "category": "trailing"},
    {"text": "السعر هو", "label": False, "category": "incomplete"},
    {"text": "بس المشكلة إن", "label": False, "category": "trailing"},
    {"text": "لأن", "label": False, "category": "incomplete"},
    {"text": "ممم", "label": False, "category": "hesitation"},
    {"text": "وبعدين", "label": False, "category": "incomplete"},
    {"text": "الحين أنا أبغى أسأل عن", "label": False, "category": "trailing"},
    {"text": "يعني", "label": False, "category": "filler"},
    {"text": "اللي قلته هو", "label": False, "category": "incomplete"},
    {"text": "عندي سؤال بخصوص", "label": False, "category": "trailing"},
    {"text": "أنا شفت إن", "label": False, "category": "trailing"},
    {"text": "المشكلة اللي واجهتني", "label": False, "category": "trailing"},
    {"text": "بس أبغى أوضح إن", "label": False, "category": "trailing"},
    {"text": "لأن الموضوع", "label": False, "category": "incomplete"},
    {"text": "هل يمكن", "label": False, "category": "incomplete"},
    {"text": "طيب بس", "label": False, "category": "filler"},
    {"text": "أها يعني", "label": False, "category": "filler"},
    {"text": "والله أنا", "label": False, "category": "trailing"},
    {"text": "السالفة إن", "label": False, "category": "incomplete"},
]


@dataclass
class EvalResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    avg_latency_ms: float
    predictions: list


def load_model():
    """Load model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    eou_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"Model loaded. EOU token ID: {eou_token_id}")
    return model, tokenizer, eou_token_id, device


def predict_eou(model, tokenizer, eou_token_id, device, text: str) -> tuple[float, float]:
    """Run EOU prediction, return (probability, latency_ms)."""
    formatted = f"<|im_start|>user\n{text}"
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        eou_prob = probs[eou_token_id].cpu().item()
    latency = (time.perf_counter() - start) * 1000

    return eou_prob, latency


def evaluate_at_threshold(predictions: list, threshold: float) -> EvalResult:
    """Compute metrics at a given threshold."""
    tp = fp = tn = fn = 0
    total_latency = 0.0

    for pred in predictions:
        predicted_eou = pred["prob"] >= threshold
        actual_eou = pred["label"]
        total_latency += pred["latency_ms"]

        if predicted_eou and actual_eou:
            tp += 1
        elif predicted_eou and not actual_eou:
            fp += 1
        elif not predicted_eou and actual_eou:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return EvalResult(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        avg_latency_ms=total_latency / len(predictions) if predictions else 0,
        predictions=predictions,
    )


def print_report(result: EvalResult):
    """Print a formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION REPORT — Threshold: {result.threshold}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {result.accuracy:.1%}")
    print(f"  Precision: {result.precision:.1%}")
    print(f"  Recall:    {result.recall:.1%}")
    print(f"  F1 Score:  {result.f1:.1%}")
    print(f"  Avg Latency: {result.avg_latency_ms:.1f}ms")
    print()
    print("  Confusion Matrix:")
    print("                Predicted EOU    Predicted ~EOU")
    print(f"  Actual EOU      TP={result.tp:<4}          FN={result.fn:<4}")
    print(f"  Actual ~EOU     FP={result.fp:<4}          TN={result.tn:<4}")
    print(f"{'=' * 60}")


def print_detailed_predictions(predictions: list, threshold: float):
    """Print per-sample predictions."""
    print(f"\n{'─' * 70}")
    print(f"  DETAILED PREDICTIONS (threshold={threshold})")
    print(f"{'─' * 70}")

    correct = 0
    for p in predictions:
        predicted = p["prob"] >= threshold
        actual = p["label"]
        is_correct = predicted == actual
        if is_correct:
            correct += 1
        status = "OK" if is_correct else "WRONG"
        arrow = "EOU" if predicted else "~EOU"
        print(
            f"  [{status:>5}] prob={p['prob']:.4f} → {arrow:<4} "
            f"(actual={'EOU' if actual else '~EOU'}) "
            f"[{p['category']:>12}] {p['text'][:35]}"
        )

    print(f"\n  {correct}/{len(predictions)} correct ({correct / len(predictions):.1%})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic EOU Model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds 0.1-0.99")
    parser.add_argument("--detailed", action="store_true", help="Show per-sample predictions")
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    model, tokenizer, eou_token_id, device = load_model()

    # Run predictions
    predictions = []
    print(f"\nRunning predictions on {len(EVAL_DATASET)} samples...")
    for sample in EVAL_DATASET:
        prob, latency = predict_eou(model, tokenizer, eou_token_id, device, sample["text"])
        predictions.append(
            {
                "text": sample["text"],
                "label": sample["label"],
                "category": sample["category"],
                "prob": prob,
                "latency_ms": latency,
            }
        )

    if args.sweep:
        print("\n" + "=" * 60)
        print("  THRESHOLD SWEEP")
        print("=" * 60)
        best_f1 = 0
        best_threshold = 0
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]:
            result = evaluate_at_threshold(predictions, t)
            print(
                f"  t={t:<5} | Acc={result.accuracy:.1%} | "
                f"P={result.precision:.1%} | R={result.recall:.1%} | "
                f"F1={result.f1:.1%} | TP={result.tp} FP={result.fp} TN={result.tn} FN={result.fn}"
            )
            if result.f1 > best_f1:
                best_f1 = result.f1
                best_threshold = t

        print(f"\n  Best threshold: {best_threshold} (F1={best_f1:.1%})")
        print_report(evaluate_at_threshold(predictions, best_threshold))
    else:
        result = evaluate_at_threshold(predictions, args.threshold)
        print_report(result)

    if args.detailed:
        print_detailed_predictions(
            predictions, args.threshold if not args.sweep else best_threshold
        )

    if args.json:
        output = {
            "model": MODEL_ID,
            "device": device,
            "num_samples": len(predictions),
            "threshold": args.threshold,
            "predictions": predictions,
        }
        if args.sweep:
            output["best_threshold"] = best_threshold
            output["best_f1"] = best_f1
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
