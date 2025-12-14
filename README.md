# Arabic Turn Detector for LiveKit

[![Model on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-blue)](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EOU-blue)](https://huggingface.co/datasets/Moustafa3092/EOU)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Fine-tuned End-of-Utterance (EOU) detection for Arabic voice agents. Optimized for Modern Standard Arabic and Gulf dialects.

## Features

- ✅ Fine-tuned on 57K Arabic samples from SADA 2022
- ✅ Specialized for Gulf dialects (Saudi, UAE, Kuwaiti)
- ✅ Handles Arabic edge cases (hesitations vs closures)
- ✅ Packaged as reusable SDK for LiveKit
- ✅ Training: 0.07 loss, 97.2% accuracy
- ✅ Fast: ~20ms GPU, ~50ms CPU

## Quick Start

### Installation

```bash
cd livekit-plugins-arabic-turn-detector
pip install -e .
```

### Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import groq, silero
from livekit_plugins_arabic_turn_detector import load

session = AgentSession(
    stt=groq.STT(language="ar"),
    llm="openai/gpt-4o",
    tts="cartesia/sonic-3",
    turn_detection=load(threshold=0.98),  # Higher threshold recommended
    vad=silero.VAD.load(),
)
```

### Run Demo

```bash
# Copy environment template and add your API keys
cp .env.example .env.local

# Run the agent
python agent.py start
```

## Project Structure

```
livekit-voice-agent/
├── livekit-plugins-arabic-turn-detector/    # SDK package
├── agent.py                                 # Demo agent
├── test_model.py                            # Performance testing
├── DATASET.md                               # Dataset documentation
├── MODEL_CARD.md                            # Training & metrics
└── README.md                                # This file
```

## Resources

### Model & Dataset

- **Model**: [Moustafa3092/livekit-turn-detector-arabic](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
- **Dataset**: [Moustafa3092/EOU](https://huggingface.co/datasets/Moustafa3092/EOU) (57,475 samples)
- **Source**: [SADA 2022](https://www.kaggle.com/datasets/sdaiancai/sada2022) (Kaggle)

### Documentation

- **[MODEL_CARD.md](MODEL_CARD.md)** - Complete training details, metrics, and performance evaluation
- **[DATASET.md](DATASET.md)** - Dataset source, processing methodology, and statistics
- **[SDK README](livekit-plugins-arabic-turn-detector/README.md)** - Plugin documentation and API reference

## Performance

**Validation Set**: 96.4% accuracy (5,747 samples)

**Real-World Edge Cases**: 54.3% accuracy (35 test cases)
- Complete utterances: 100% ✓
- Incomplete utterances: 17% ✗

**Known Issue**: Model over-predicts (high probabilities for most inputs). **Workaround**: Use threshold=0.98

See [MODEL_CARD.md](MODEL_CARD.md) for detailed metrics and analysis.

## Testing

```bash
# Quick test (12 cases)
python test_quick.py

# Comprehensive test (35+ cases)
python test_model.py
```

## Requirements

```txt
livekit-agents>=0.8.0
livekit-plugins-groq
livekit-plugins-silero
transformers>=4.45.2
torch>=2.0.0
peft>=0.18.0
```

## Citation

```bibtex
@software{arabic_turn_detector_2024,
  author = {Abdelsattar, Moustafa},
  title = {Arabic Turn Detector for LiveKit},
  year = {2024},
  url = {https://github.com/Moustafa-abdelsattar/livkit_eou}
}

@dataset{arabic_eou_dataset_2024,
  author = {Abdelsattar, Moustafa},
  title = {Arabic EOU Dataset for Turn Detection},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/Moustafa3092/EOU}
}
```

## License

Apache 2.0

---

**Model**: https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic
**Dataset**: https://huggingface.co/datasets/Moustafa3092/EOU
**Repository**: https://github.com/Moustafa-abdelsattar/livkit_eou
