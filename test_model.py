"""
Test Arabic Turn Detector Performance

Test the EOU model against various Arabic inputs to evaluate performance.
"""

import sys
import io

# Fix Windows UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, 'livekit-plugins-arabic-turn-detector')

from livekit_plugins_arabic_turn_detector.arabic_eou import ArabicTurnDetector


def test_model():
    """Test the Arabic Turn Detector with various inputs"""

    print("="*70)
    print("Loading Arabic Turn Detector...")
    print("="*70)

    # Initialize detector
    detector = ArabicTurnDetector(
        model_id="Moustafa3092/livekit-turn-detector-arabic",
        unlikely_threshold=0.5
    )

    print(f"Model loaded successfully on {detector._device}\n")

    # Test cases: (text, expected_eou, category)
    test_cases = [
        # === COMPLETE UTTERANCES (Should be HIGH probability) ===
        ("شكرا جزيلا", True, "Closure - Thanks"),
        ("تمام", True, "Closure - Perfect"),
        ("نعم فهمت", True, "Closure - Yes understood"),
        ("لا شكرا", True, "Closure - No thanks"),
        ("مع السلامة", True, "Closure - Goodbye"),
        ("الله يسلمك", True, "Closure - Bless you"),
        ("ان شاء الله", True, "Closure - God willing"),
        ("حاضر", True, "Closure - Okay"),
        ("ماشي", True, "Closure - Alright"),
        ("أوكي فهمت", True, "Closure - OK understood"),
        ("كيف حالك اليوم؟", True, "Question - Complete"),
        ("أنت تسمعني؟", True, "Question - Can you hear me"),
        ("ما اسمك؟", True, "Question - What's your name"),
        ("أريد أن أعرف الوقت الآن", True, "Statement - Complete"),
        ("أنا بخير والحمد لله", True, "Statement - I'm fine"),

        # === INCOMPLETE UTTERANCES (Should be LOW probability) ===
        ("اممممممم", False, "Hesitation - Ummmm"),
        ("امم يعني", False, "Hesitation - Umm like"),
        ("يعني", False, "Filler - You know"),
        ("خلاص بس", False, "Incomplete - Okay but"),
        ("طيب و", False, "Incomplete - Okay and"),
        ("انا كنت", False, "Incomplete - I was"),
        ("المشكلة هي", False, "Incomplete - The problem is"),
        ("بس", False, "Incomplete - But"),
        ("و", False, "Incomplete - And"),
        ("لأن", False, "Incomplete - Because"),
        ("إذا", False, "Incomplete - If"),
        ("كان يريد", False, "Incomplete - He wanted"),
        ("عندما", False, "Incomplete - When"),

        # === EDGE CASES ===
        ("اه", False, "Hesitation - Ahh"),
        ("هممم", False, "Hesitation - Hmmm"),
        ("ايه", True, "Affirmation - Yeah"),
        ("اي نعم", True, "Affirmation - Yes yes"),
        ("لا لا", True, "Negation - No no"),
        ("والله", False, "Filler - I swear"),
        ("يا اخي", False, "Vocative - My brother"),
    ]

    # Run tests
    print("="*70)
    print(f"{'TEXT':<30} | {'EXPECTED':<8} | {'PROB':>6} | {'PRED':<6} | {'STATUS':<6} | CATEGORY")
    print("="*70)

    correct = 0
    total = len(test_cases)

    results_by_category = {}

    for text, expected_eou, category in test_cases:
        # Get probability from model
        prob = detector._predict_eou(text)

        # Predict based on threshold
        predicted_eou = prob > 0.5

        # Check if correct
        is_correct = predicted_eou == expected_eou
        correct += is_correct

        # Track by category
        if category not in results_by_category:
            results_by_category[category] = {"correct": 0, "total": 0}
        results_by_category[category]["total"] += 1
        if is_correct:
            results_by_category[category]["correct"] += 1

        # Status symbol
        status = "✓" if is_correct else "✗"

        # Expected and predicted labels
        exp_label = "EOU" if expected_eou else "CONT"
        pred_label = "EOU" if predicted_eou else "CONT"

        # Print result
        print(f"{text:<30} | {exp_label:<8} | {prob:>6.3f} | {pred_label:<6} | {status:<6} | {category}")

    print("="*70)

    # Overall statistics
    accuracy = 100 * correct / total
    print(f"\nOVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")

    # Statistics by category
    print("\n" + "="*70)
    print("PERFORMANCE BY CATEGORY")
    print("="*70)

    for category in sorted(results_by_category.keys()):
        stats = results_by_category[category]
        cat_accuracy = 100 * stats["correct"] / stats["total"]
        print(f"{category:<30} | {stats['correct']}/{stats['total']} ({cat_accuracy:.1f}%)")

    # Threshold analysis
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS")
    print("="*70)

    thresholds = [0.3, 0.5, 0.7, 0.9, 0.95, 0.98]
    print(f"{'Threshold':<12} | {'Accuracy':<10}")
    print("-" * 25)

    for threshold in thresholds:
        correct_at_threshold = 0
        for text, expected_eou, _ in test_cases:
            prob = detector._predict_eou(text)
            predicted_eou = prob > threshold
            if predicted_eou == expected_eou:
                correct_at_threshold += 1

        acc = 100 * correct_at_threshold / total
        marker = " ← BEST" if acc == max([100 * sum(1 for text, exp, _ in test_cases if (detector._predict_eou(text) > t) == exp) / total for t in thresholds]) else ""
        print(f"{threshold:<12.2f} | {acc:<10.1f}%{marker}")

    print("="*70)
    print("\nTesting complete!")


if __name__ == "__main__":
    test_model()
