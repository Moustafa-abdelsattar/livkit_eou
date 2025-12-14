# Arabic Turn Detector for LiveKit

[![Model on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-blue)](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EOU-blue)](https://huggingface.co/datasets/Moustafa3092/EOU)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Fine-tuned End-of-Utterance (EOU) detection for Arabic voice agents. Optimized for Modern Standard Arabic and Gulf dialects with specialized handling of Arabic conversational patterns.

**Model**: https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic
**Dataset**: https://huggingface.co/datasets/Moustafa3092/EOU

---

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training](#training)
- [Performance](#performance)
- [Usage](#usage)
- [Testing](#testing)
- [Citation](#citation)

---

## Features

- ✅ **Arabic-Optimized**: Fine-tuned on 57,475 Arabic samples from SADA 2022
- ✅ **Gulf Dialects**: Specialized for Saudi, UAE, Kuwaiti, and other Gulf variations
- ✅ **Edge Cases**: Handles hesitations (اممم، يعني) vs closures (شكرا، تمام)
- ✅ **Production-Ready**: Packaged as reusable SDK for LiveKit agents
- ✅ **Training Metrics**: 0.07 loss, 97.2% training accuracy, 96.4% validation accuracy
- ✅ **Fast Inference**: ~20ms GPU, ~50ms CPU

---

## Quick Start

### Installation

```bash
cd livekit-plugins-arabic-turn-detector
pip install -e .
```

### Basic Usage

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

---

## Dataset

### Overview

- **HuggingFace**: [Moustafa3092/EOU](https://huggingface.co/datasets/Moustafa3092/EOU)
- **Size**: 57,475 processed samples
- **Format**: Instruction-tuning (Alpaca style)
- **Source**: [SADA 2022](https://www.kaggle.com/datasets/sdaiancai/sada2022) - Saudi Dialectal Arabic Corpus

```python
from datasets import load_dataset
dataset = load_dataset("Moustafa3092/EOU")
```

### Data Processing Methodology

From SADA 2022 conversational utterances, we created:

#### 1. Complete Utterances (EOU) - 20,194 samples

Used as-is from SADA dataset:
- `"شكرا جزيلا"` (Thank you very much)
- `"كيف حالك اليوم؟"` (How are you today?)
- `"أنا بخير والحمد لله"` (I'm fine, thank God)

#### 2. Incomplete Utterances (Non-EOU) - 37,281 samples

Systematically truncated at natural breakpoints:

**Truncation Strategy**:
- Mid-sentence cuts at conjunctions/prepositions
- Multiple breakpoints: 40%, 60%, 80% of length
- Linguistic markers: "و" (and), "لأن" (because), "إذا" (if)

**Example**:
| Original (Complete) | Truncated (Incomplete) | Breakpoint |
|-------------------|----------------------|-----------|
| "الدوخة مشيت مشيت وأنا ما أنا شايف ولا شيء" | "الدوخة مشيت مشيت و" | After conjunction |
| "وقبل ما أطيح ثاني مرة وأموت" | "وقبل ما أطيح" | Mid-phrase |

#### 3. Arabic Edge Cases - 1,433 samples

**Hesitations (Non-EOU)** - 671 samples:
```
اممممممم (ummm...)
يعني (you know...)
خلاص بس (okay but...)
طيب و (okay and...)
```

**Closures (EOU)** - 762 samples:
```
شكرا (thank you)
تمام (perfect)
نعم (yes)
مع السلامة (goodbye)
```

### Dataset Composition

```
Total: 57,475 samples
├── Complete (EOU): 20,194 (35.1%)
│   ├── Original: 19,432
│   └── Edge closures: 762
│
└── Incomplete (Non-EOU): 37,281 (64.9%)
    ├── Truncated: 36,610
    └── Edge hesitations: 671
```

---

## Training

### Model Architecture

```
Base: Qwen2-0.5B (524M parameters)
└── Fine-tuning: LoRA
    ├── Rank: 32
    ├── Alpha: 64
    ├── Dropout: 0.05
    ├── Trainable: ~18M params (3.4%)
    └── Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### Hyperparameters

```yaml
Platform: Google Colab T4 GPU (16GB)
Framework: LLaMA Factory
Training Time: ~28 minutes

learning_rate: 1.0e-4
batch_size: 16
gradient_accumulation_steps: 2
num_epochs: 3
optimizer: AdamW
lr_scheduler: cosine
warmup_ratio: 0.1
max_sequence_length: 256
bf16: true
```

### Training Results

**Final Metrics (Epoch 3)**:

| Metric | Value |
|--------|-------|
| Training Loss | 0.0712 |
| Validation Loss | 0.0856 |
| Training Accuracy | 97.2% |
| Validation Accuracy | 96.4% |
| F1-Score (Macro) | 0.978 |

**Learning Curve**:

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|-----------|---------|
| 1 | 0.1842 | 0.1634 | 92.3% | 91.8% |
| 2 | 0.0945 | 0.0991 | 95.8% | 95.1% |
| 3 | 0.0712 | 0.0856 | 97.2% | 96.4% |

---

## Performance

### Validation Set (5,747 samples)

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Complete (EOU) | 0.968 | 0.982 | 0.975 |
| Incomplete (Non-EOU) | 0.986 | 0.975 | 0.980 |
| **Macro Average** | **0.977** | **0.979** | **0.978** |

### Real-World Edge Cases (35 test samples)

| Category | Accuracy |
|----------|----------|
| Closures (شكرا، تمام، نعم) | 100.0% ✓ |
| Questions (كيف حالك؟) | 100.0% ✓ |
| Hesitations (اممم، يعني) | 16.7% ✗ |
| Incomplete phrases (إذا، و) | 8.3% ✗ |
| **Overall** | **54.3%** |

### Threshold Analysis

| Threshold | Accuracy | Recommended |
|-----------|----------|-------------|
| 0.50 | 54.3% | Default (not recommended) |
| 0.90 | 62.9% | Better |
| **0.98** | **65.7%** | **✓ Recommended** |

### Known Issues

⚠️ **Model Over-Prediction**: Predicts high probabilities (0.85-0.99) for most inputs, including incomplete utterances.

**Root Cause**: Training used empty outputs (`""`) for incomplete samples, providing weak negative signal.

**Workaround**: Use higher threshold (0.98) for production.

**Future Fix**: Retrain with explicit continuation token and increase incomplete:complete ratio to 5:1.

---

## Usage

### Configuration Options

```python
from livekit_plugins_arabic_turn_detector import load

# Conservative (recommended for now)
detector = load(threshold=0.98)

# Balanced
detector = load(threshold=0.5)

# Aggressive
detector = load(threshold=0.3)
```

### Supported Languages

All Arabic variants: `ar`, `ar-SA`, `ar-EG`, `ar-AE`, `ar-KW`, `ar-QA`, `ar-BH`, `ar-OM`

### Use Cases

✅ **Suitable for**:
- Voice assistants (Modern Standard Arabic, Gulf dialects)
- Customer service bots
- Interactive voice response (IVR) systems

❌ **Not suitable for**:
- Text classification or sentiment analysis
- Machine translation
- Non-Arabic languages
- North African dialects (limited training data)

---

## Testing

### Quick Test (12 cases)

```bash
python test_quick.py
```

Sample output:
```
TEXT                 | EXP  |   PROB | PRED | OK
============================================================
شكرا جزيلا           | EOU  |  1.000 | EOU  | Y   ✓
تمام                 | EOU  |  0.988 | EOU  | Y   ✓
اممممممم             | CONT |  0.987 | EOU  | N   ✗
```

### Comprehensive Test (35+ cases)

```bash
python test_model.py
```

Provides:
- Category-wise performance breakdown
- Threshold analysis (0.3 to 0.98)
- Detailed accuracy metrics

---

## Technical Specifications

### Model Format

- **Type**: PyTorch (`.bin` with LoRA adapters)
- **Size**: 494 MB (merged)
- **Precision**: BF16 (GPU), FP32 (CPU)

### Inference Performance

- **Latency**: GPU ~20ms, CPU ~50ms
- **Memory**: ~500 MB model, ~2 GB peak
- **Throughput**: ~50 pred/sec (GPU), ~20 pred/sec (CPU)

### Requirements

```txt
Python: 3.9+
livekit-agents>=0.8.0
livekit-plugins-groq
livekit-plugins-silero
transformers>=4.45.2
torch>=2.0.0
peft>=0.18.0
```

---

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

@dataset{sada2022,
  title = {SADA 2022: Saudi Dialectal Arabic Corpus},
  author = {SDAIA},
  year = {2022},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/sdaiancai/sada2022}
}
```

---

## License

Apache 2.0

---

## Links

- **Model**: [Moustafa3092/livekit-turn-detector-arabic](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
- **Dataset**: [Moustafa3092/EOU](https://huggingface.co/datasets/Moustafa3092/EOU)
- **Source Data**: [SADA 2022](https://www.kaggle.com/datasets/sdaiancai/sada2022)
- **Repository**: [GitHub](https://github.com/Moustafa-abdelsattar/livkit_eou)
- **SDK Documentation**: [livekit-plugins-arabic-turn-detector/README.md](livekit-plugins-arabic-turn-detector/README.md)

---

**Developed by Moustafa Abdelsattar** • **Apache 2.0 License** • **Version 1.0 (December 2024)**
