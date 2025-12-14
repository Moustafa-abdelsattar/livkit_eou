# Arabic Turn Detector for LiveKit

Fine-tuned End-of-Utterance detection for Arabic voice agents.

## Quick Start

```bash
# 1. Install the plugin
cd livekit-plugins-arabic-turn-detector
pip install -e .

# 2. Set up environment (create .env.local with your API keys)
cp .env.example .env.local

# 3. Run the agent
python agent.py start
```

## Project Structure

```
livekit-voice-agent/
├── livekit-plugins-arabic-turn-detector/    # The SDK
│   ├── livekit_plugins_arabic_turn_detector/
│   │   ├── __init__.py
│   │   ├── arabic_eou.py
│   │   ├── models.py
│   │   └── version.py
│   ├── setup.py
│   ├── requirements.txt
│   └── README.md
│
├── agent.py                                 # Your agent
├── requirements.txt                         # Dependencies
└── .env.local                               # Your API keys
```

## Environment Setup

Create `.env.local`:

```bash
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import groq, silero
from livekit_plugins_arabic_turn_detector import load

session = AgentSession(
    stt=groq.STT(language="ar"),
    llm="openai/gpt-4o",
    tts="cartesia/sonic-3",
    turn_detection=load(threshold=0.5),
    vad=silero.VAD.load(),
)
```

## Threshold

The model currently over-predicts, use higher threshold:

```python
turn_detection=load(threshold=0.98)  # Recommended
```

## Model

- **HuggingFace**: [Moustafa3092/livekit-turn-detector-arabic](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
- **Base**: LiveKit Turn Detector (Qwen2-0.5B)
- **Training**: LoRA fine-tuned on 57k Arabic samples

## Links

- [Plugin README](livekit-plugins-arabic-turn-detector/README.md) - Full documentation
- [LiveKit Docs](https://docs.livekit.io/agents/)
- [Model Card](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
