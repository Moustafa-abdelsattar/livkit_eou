"""
Tests for models.py — constants and configuration.
"""

from livekit_plugins_arabic_turn_detector.models import (
    ARABIC_MODEL_ID,
    DEFAULT_THRESHOLD,
    SUPPORTED_LANGUAGES,
)


def test_model_id_points_to_huggingface():
    assert "Moustafa3092" in ARABIC_MODEL_ID
    assert "arabic" in ARABIC_MODEL_ID.lower()


def test_default_threshold_is_valid():
    assert 0.0 < DEFAULT_THRESHOLD < 1.0


def test_supported_languages_include_saudi():
    assert "ar-SA" in SUPPORTED_LANGUAGES


def test_supported_languages_include_generic_arabic():
    assert "ar" in SUPPORTED_LANGUAGES


def test_supported_languages_include_gulf_dialects():
    gulf = {"ar-SA", "ar-AE", "ar-KW", "ar-QA", "ar-BH", "ar-OM"}
    assert gulf.issubset(set(SUPPORTED_LANGUAGES))


def test_supported_languages_no_duplicates():
    assert len(SUPPORTED_LANGUAGES) == len(set(SUPPORTED_LANGUAGES))
