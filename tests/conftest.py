"""
Shared test fixtures for Arabic Turn Detector tests.
"""

from unittest.mock import MagicMock

import pytest
import torch

# -- Saudi dialect test samples --------------------------------------------------
# Each tuple: (text, expected_eou: True=complete, False=incomplete)

COMPLETE_UTTERANCES = [
    "السلام عليكم",  # Greeting (complete)
    "كيف حالك اليوم؟",  # How are you today?
    "أبغى أحجز موعد يوم الخميس",  # I want to book Thursday
    "شكراً جزيلاً على مساعدتك",  # Thank you for your help
    "لا ما أبغى شي ثاني",  # No I don't want anything else
    "وش سعر الباقة الشهرية؟",  # What's the monthly package price?
    "خلاص تمام",  # OK done
    "الله يعطيك العافية",  # God bless you (closing)
    "أبغى أكلم المدير لو سمحت",  # I want to speak to the manager
    "ما عندي أي سؤال ثاني",  # I have no other questions
    "إيه صحيح كلامك",  # Yes you're right
    "طيب خلنا نكمل بكرة",  # OK let's continue tomorrow
]

INCOMPLETE_UTTERANCES = [
    "أبغى",  # I want... (incomplete)
    "يعني أنا",  # I mean I... (trailing)
    "السعر هو",  # The price is... (incomplete)
    "بس المشكلة إن",  # But the problem is... (trailing)
    "لأن",  # Because... (incomplete)
    "ممم",  # Hesitation
    "وبعدين",  # And then... (incomplete)
    "الحين أنا أبغى أسأل عن",  # Now I want to ask about... (trailing)
    "يعني",  # I mean... (filler)
    "اللي قلته هو",  # What you said is... (incomplete)
]

EDGE_CASES = [
    ("مم", False),  # Very short hesitation
    (".", False),  # Just punctuation
    ("", False),  # Empty
    (" ", False),  # Whitespace
    ("ا", False),  # Single letter
]


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.return_value = 151643  # <|im_end|> token ID
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock HuggingFace causal LM that returns controllable logits."""
    model = MagicMock()
    # Default: return logits where <|im_end|> has high probability
    vocab_size = 151936  # Qwen2 vocab size
    logits = torch.zeros(1, 5, vocab_size)
    logits[0, -1, 151643] = 5.0  # High logit for <|im_end|>
    model.return_value.logits = logits
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_chat_context():
    """Create a mock LiveKit ChatContext."""
    ctx = MagicMock()

    def make_item(role, text):
        item = MagicMock()
        item.type = "message"
        item.role = role
        item.text_content = text
        return item

    ctx._make_item = make_item
    return ctx
