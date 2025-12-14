"""
LiveKit Plugins - Arabic Turn Detector

Fine-tuned Arabic End-of-Utterance detection for LiveKit voice agents.
Optimized for Modern Standard Arabic and Gulf dialects.

Model: https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic
"""

from .arabic_eou import ArabicTurnDetector, load
from .version import __version__
from .models import ARABIC_MODEL_ID, DEFAULT_THRESHOLD, SUPPORTED_LANGUAGES

__all__ = [
    "ArabicTurnDetector",
    "load",
    "__version__",
    "ARABIC_MODEL_ID",
    "DEFAULT_THRESHOLD",
    "SUPPORTED_LANGUAGES",
]
