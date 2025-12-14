"""
Quick Test - Arabic Turn Detector

Fast performance evaluation without interactive mode.
"""

import sys
import io

# Fix Windows UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, 'livekit-plugins-arabic-turn-detector')

from livekit_plugins_arabic_turn_detector.arabic_eou import ArabicTurnDetector


print("Loading model...")
detector = ArabicTurnDetector(
    model_id="Moustafa3092/livekit-turn-detector-arabic",
    unlikely_threshold=0.5
)
print(f"Loaded on {detector._device}\n")

# Test cases: (text, expected_complete, category)
tests = [
    # Complete utterances
    ("شكرا جزيلا", True, "Thanks"),
    ("تمام", True, "Perfect"),
    ("نعم", True, "Yes"),
    ("مع السلامة", True, "Goodbye"),
    ("كيف حالك؟", True, "Question"),
    ("أنت تسمعني؟", True, "Question"),

    # Incomplete utterances
    ("اممممممم", False, "Hesitation"),
    ("يعني", False, "Filler"),
    ("خلاص بس", False, "But"),
    ("انا كنت", False, "I was"),
    ("بس", False, "But"),
    ("إذا", False, "If"),
]

print("="*60)
print(f"{'TEXT':<20} | {'EXP':<4} | {'PROB':>6} | {'PRED':<4} | {'OK':<3}")
print("="*60)

correct = 0
for text, expected, category in tests:
    prob = detector._predict_eou(text)
    pred = prob > 0.5
    ok = pred == expected
    correct += ok

    exp_s = "EOU" if expected else "CONT"
    pred_s = "EOU" if pred else "CONT"
    ok_s = "Y" if ok else "N"

    print(f"{text:<20} | {exp_s:<4} | {prob:>6.3f} | {pred_s:<4} | {ok_s:<3}")

print("="*60)
print(f"Accuracy: {correct}/{len(tests)} ({100*correct/len(tests):.1f}%)")
print("="*60)
