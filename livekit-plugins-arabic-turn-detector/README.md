# LiveKit Plugins - Arabic Turn Detector

[![Model on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-blue)](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Fine-tuned Arabic End-of-Utterance (EOU) detection for LiveKit voice agents. Optimized for Modern Standard Arabic and Gulf dialects with special handling of Arabic-specific conversational patterns.

## Features

- **Optimized for Arabic**: Fine-tuned on 57,475 Arabic EOU samples
- **Gulf Dialects**: Specialized handling for Saudi, UAE, Kuwaiti, and other Gulf Arabic dialects
- **Edge Case Detection**: Accurately distinguishes between:
  - **Hesitations** (اممم, يعني) → Continue listening
  - **Closures** (شكرا, تمام, نعم) → End of utterance
- **LiveKit Integration**: Drop-in replacement for LiveKit's turn detection
- **Free & Open Source**: Based on [livekit/turn-detector](https://huggingface.co/livekit/turn-detector)

## Model Details

- **Base Model**: LiveKit Turn Detector (Qwen2-0.5B)
- **Fine-tuning**: LoRA (rank=32, alpha=64)
- **Training Data**: 57,475 samples (35% complete, 65% incomplete utterances)
- **Model Size**: 494 MB
- **Languages**: Arabic (ar, ar-SA, ar-EG, ar-AE, ar-KW, etc.)

**Model on HuggingFace**: [Moustafa3092/livekit-turn-detector-arabic](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)

## Installation

### From Local (Development)

```bash
cd livekit-plugins-arabic-turn-detector
pip install -e .
```

### From PyPI (Coming Soon)

```bash
pip install livekit-plugins-arabic-turn-detector
```

### From GitHub

```bash
pip install git+https://github.com/Moustafa3092/livekit-plugins-arabic-turn-detector.git
```

## Quick Start

```python
from livekit.agents import AgentSession
from livekit.plugins import groq, silero
from livekit_plugins_arabic_turn_detector import load

session = AgentSession(
    stt=groq.STT(language="ar"),  # Arabic STT
    llm="openai/gpt-4o-mini",     # Any LLM
    tts="cartesia/sonic-3",        # Arabic TTS
    turn_detection=load(threshold=0.5),  # Arabic EOU detector
    vad=silero.VAD.load(),         # VAD for audio activity
    allow_interruptions=True,
)
```

## Usage Examples

### Basic Usage

```python
from livekit_plugins_arabic_turn_detector import load

# Load with default settings (threshold=0.5)
detector = load()

# Load with custom threshold
detector = load(threshold=0.6)

# Use in AgentSession
session = AgentSession(
    turn_detection=detector,
    # ... other config
)
```

### Advanced Usage

```python
from livekit_plugins_arabic_turn_detector import ArabicTurnDetector

# Full control over initialization
detector = ArabicTurnDetector(
    model_id="Moustafa3092/livekit-turn-detector-arabic",
    threshold=0.5,
    device="cuda",  # or "cpu"
)

# Update threshold dynamically
detector.update_threshold(0.6)
```

### Complete Agent Example

```python
import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import groq, silero
from livekit_plugins_arabic_turn_detector import load

load_dotenv()


class ArabicAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""أنت مساعد صوتي ذكي باللغة العربية.
تحدث دائماً باللغة العربية الفصحى أو اللهجة الخليجية حسب المستخدم.
أنت مفيد وودود وتجيب على الأسئلة بوضوح."""
        )


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=groq.STT(language="ar"),
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-3",
        turn_detection=load(threshold=0.5),  # Arabic turn detector
        vad=silero.VAD.load(),
        allow_interruptions=True,
    )

    await session.start(room=ctx.room, agent=ArabicAssistant())
    await session.generate_reply(
        instructions="رحب بالمستخدم باللغة العربية."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
```

## Configuration

### Threshold Tuning

The `threshold` parameter controls when to end a turn:

- **Lower threshold (0.3-0.4)**: More aggressive turn-taking, faster responses
- **Medium threshold (0.5)**: Balanced (recommended)
- **Higher threshold (0.6-0.7)**: More conservative, waits for clear completion

```python
# For faster, more responsive conversations
detector = load(threshold=0.4)

# For careful, deliberate turn-taking
detector = load(threshold=0.6)
```

### Supported Languages

All Arabic language variants are supported:

- `ar` - Generic Arabic
- `ar-SA` - Saudi Arabic
- `ar-EG` - Egyptian Arabic
- `ar-AE` - UAE Arabic
- `ar-KW` - Kuwaiti Arabic
- `ar-QA` - Qatari Arabic
- `ar-BH` - Bahraini Arabic
- `ar-OM` - Omani Arabic

## Testing

Test the detector with Arabic edge cases:

```python
from livekit_plugins_arabic_turn_detector import load

detector = load()

# Test cases
test_cases = [
    ("شكرا جزيلا", True),    # "Thank you very much" - complete
    ("تمام تمام", True),      # "Perfect" - complete
    ("اممممم", False),        # "Ummmm..." - incomplete
    ("يعني", False),          # "You know..." - incomplete
]

for text, expected in test_cases:
    # Simulated chat context with the text
    # (in real usage, LiveKit provides this)
    # ... check detector output
```

## Performance

- **Accuracy**: >90% on Arabic edge cases
- **Inference Speed**: Real-time compatible (CPU: ~50ms, GPU: ~20ms)
- **Memory**: ~500MB (model size)
- **Training Time**: ~20-30 minutes on T4 GPU

## Training Data

The model was fine-tuned on:

- **57,475 samples** total
  - **20,194 complete** utterances (EOU)
  - **37,281 incomplete** utterances (non-EOU)

**Edge Cases** (1,433 samples):
- **Hesitations**: اممم, يعني, خلاص بس, طيب و
- **Closures**: شكرا, تمام, نعم, لا, مع السلامة

## How It Works

The model uses the LiveKit turn detection format:

1. **Input**: Arabic conversation context in chat format
2. **Processing**: Qwen2-0.5B model analyzes text semantics
3. **Output**: Probability that `<|im_end|>` token should appear (0.0-1.0)
4. **Decision**: If probability > threshold → End turn

## Comparison with LiveKit Multilingual

| Feature | Arabic Turn Detector | LiveKit Multilingual |
|---------|---------------------|---------------------|
| Arabic Optimization | ✅ Fine-tuned on 57K samples | ❌ Generic multilingual |
| Gulf Dialects | ✅ Specialized handling | ⚠️ Basic support |
| Edge Cases (اممم, شكرا) | ✅ Trained specifically | ⚠️ May misclassify |
| Other Languages | ❌ Arabic only | ✅ 100+ languages |
| Model Size | 494 MB | Similar |

**Recommendation**: Use this plugin if your agent primarily handles Arabic conversations. Use LiveKit's multilingual for multi-language support.

## Limitations

- Optimized for Modern Standard Arabic and Gulf dialects
- May need additional fine-tuning for North African dialects (Maghrebi)
- Requires sufficient context (very short utterances may be less accurate)
- CPU inference is slower than GPU (~50ms vs ~20ms)

## License

Apache 2.0 - See [LICENSE](LICENSE) file

## Acknowledgements

- Based on [LiveKit Turn Detector](https://huggingface.co/livekit/turn-detector)
- Built with [Transformers](https://huggingface.co/docs/transformers) and [PEFT](https://huggingface.co/docs/peft)
- Fine-tuned for Arabic language support

## Citation

If you use this model, please cite:

```bibtex
@misc{livekit-arabic-turn-detector,
  author = {Moustafa3092},
  title = {LiveKit Plugins - Arabic Turn Detector},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Moustafa3092/livekit-plugins-arabic-turn-detector}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- **Issues**: [GitHub Issues](https://github.com/Moustafa3092/livekit-plugins-arabic-turn-detector/issues)
- **Model**: [HuggingFace Model Page](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
- **LiveKit Docs**: [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/)

## Changelog

### v0.1.0 (2024-12-14)

- Initial release
- Fine-tuned on 57,475 Arabic EOU samples
- Support for Modern Standard Arabic and Gulf dialects
- LiveKit AgentSession integration
- Edge case handling (hesitations vs closures)
