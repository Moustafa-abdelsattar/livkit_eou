"""
Arabic Turn Detector - Following LiveKit's exact architecture
"""

from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Any

from livekit.agents import llm

from .models import ARABIC_MODEL_ID, DEFAULT_THRESHOLD, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6


class ArabicTurnDetector:
    """
    Arabic Turn Detector following LiveKit's architecture

    Simplified version without EOUModelBase inheritance to avoid
    inference executor complexity while maintaining API compatibility.
    """

    def __init__(
        self,
        *,
        model_id: str = ARABIC_MODEL_ID,
        unlikely_threshold: float | None = None,
    ):
        """
        Initialize Arabic turn detector

        Args:
            model_id: HuggingFace model ID
            unlikely_threshold: Optional threshold override for all languages
        """
        self._model_id = model_id
        self._unlikely_threshold = unlikely_threshold
        self._languages: dict[str, Any] = {}

        # Default threshold
        self._default_threshold = unlikely_threshold or DEFAULT_THRESHOLD

        # Auto-detect device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Arabic EOU model: {model_id} on {self._device}")

        # Load model and tokenizer
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
        )

        if self._device == "cpu":
            self._model = self._model.to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # Get <|im_end|> token ID for EOU prediction
        self._eou_token_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Set thresholds for all supported Arabic variants
        for lang in SUPPORTED_LANGUAGES:
            self._languages[lang] = {"threshold": self._default_threshold}

        logger.info(f"Arabic EOU model loaded (threshold={self._default_threshold})")

    @property
    def model(self) -> str:
        """Return model type identifier"""
        return "arabic"

    @property
    def provider(self) -> str:
        """Return provider identifier"""
        return "hf"

    def _inference_method(self) -> str:
        """Return inference method identifier"""
        return "arabic_eou_pytorch"

    async def supports_language(self, language: str | None) -> bool:
        """
        Check if this detector supports a given language

        Args:
            language: Language code (e.g., "ar", "ar-SA")

        Returns:
            True if language is Arabic variant, False otherwise
        """
        return await self.unlikely_threshold(language) is not None

    async def unlikely_threshold(self, language: str | None) -> float | None:
        """
        Get threshold for a language (following LiveKit's exact pattern)

        Args:
            language: Language code

        Returns:
            Threshold if language is supported, None otherwise
        """
        if language is None:
            return None

        # Try the full language code first
        lang = language.lower()
        lang_data = self._languages.get(lang)

        # Try the base language if the full language code is not found
        if lang_data is None and "-" in lang:
            base_lang = lang.split("-")[0]
            lang_data = self._languages.get(base_lang)

        if not lang_data:
            return None

        # If a custom threshold is provided, use it
        if self._unlikely_threshold is not None:
            return self._unlikely_threshold
        else:
            return lang_data["threshold"]

    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext,
        *,
        timeout: float | None = 3,
    ) -> float:
        """
        Predict end-of-turn probability (following LiveKit's exact pattern)

        Args:
            chat_ctx: LiveKit chat context
            timeout: Prediction timeout (not used, kept for compatibility)

        Returns:
            EOU probability (0.0-1.0)
        """
        # Extract messages following LiveKit's exact pattern
        messages: list[dict[str, Any]] = []
        for item in chat_ctx.items:
            if item.type != "message":
                continue

            if item.role not in ("user", "assistant"):
                continue

            text_content = item.text_content
            if text_content:
                messages.append(
                    {
                        "role": item.role,
                        "content": text_content,
                    }
                )

        # Take last MAX_HISTORY_TURNS messages
        messages = messages[-MAX_HISTORY_TURNS:]

        if not messages:
            return 0.0

        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"]:
                last_user_msg = msg["content"]
                break

        if not last_user_msg or len(last_user_msg.strip()) < 2:
            return 0.0

        try:
            # Run inference with PyTorch model
            prob = self._predict_eou(last_user_msg)

            logger.debug(
                "eou prediction",
                extra={
                    "eou_probability": prob,
                    "input": last_user_msg[:50],
                }
            )

            return float(prob)
        except Exception as e:
            logger.error(f"Error predicting EOU: {e}")
            return 0.5  # Safe default

    def _predict_eou(self, text: str) -> float:
        """
        Internal EOU prediction using PyTorch model

        Args:
            text: Arabic text to analyze

        Returns:
            EOU probability (0.0-1.0)
        """
        # Format input as expected by the model
        formatted_text = f"<|im_start|>user\n{text}"

        # Tokenize
        inputs = self._tokenizer(
            formatted_text,
            return_tensors="pt",
            add_special_tokens=False
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            eou_prob = probs[self._eou_token_id]

        return eou_prob.cpu().item()


def load(
    *,
    model_id: str = ARABIC_MODEL_ID,
    unlikely_threshold: float | None = None,
    threshold: float | None = None,  # Backward compatibility alias
) -> ArabicTurnDetector:
    """
    Load Arabic turn detector (following LiveKit's load() pattern)

    Args:
        model_id: HuggingFace model ID
        unlikely_threshold: Optional threshold override (LiveKit pattern)
        threshold: Alias for unlikely_threshold (backward compatibility)

    Returns:
        ArabicTurnDetector instance
    """
    # Support both parameter names
    final_threshold = unlikely_threshold or threshold

    return ArabicTurnDetector(
        model_id=model_id,
        unlikely_threshold=final_threshold,
    )
