# Arabic Turn Detector for LiveKit

[![Model on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-blue)](https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic)
[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-EOU-blue)](https://huggingface.co/datasets/Moustafa3092/EOU)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/Moustafa-abdelsattar/livkit_eou/actions/workflows/ci.yml/badge.svg)](https://github.com/Moustafa-abdelsattar/livkit_eou/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

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
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [CI/CD](#cicd)
- [Citation](#citation)

---

## Features

- **Arabic-Optimized**: Fine-tuned on 57,475 Arabic samples from SADA 2022
- **Gulf Dialects**: Specialized for Saudi, UAE, Kuwaiti, and other Gulf variations
- **Edge Cases**: Handles hesitations (اممم، يعني) vs closures (شكرا، تمام)
- **Production-Ready**: Packaged as reusable SDK for LiveKit agents
- **CI/CD Pipeline**: Automated lint, test (Python 3.10-3.12), and package build via GitHub Actions
- **Model Evaluation**: Threshold sweep, F1/precision/recall, confusion matrix, per-category breakdown
- **Docker Deployment**: Containerized with pre-baked model weights for instant cold-start
- **Training Metrics**: 0.07 loss, 97.2% training accuracy, 96.4% validation accuracy
- **Fast Inference**: ~20ms GPU, ~50ms CPU

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

### Unit Tests (no GPU required)

Mocked HuggingFace model — runs in CI without GPU or model download:

```bash
pip install -e "livekit-plugins-arabic-turn-detector[dev]" pytest-asyncio
pytest tests/ -v --ignore=tests/test_integration.py -m "not slow"
```

**Coverage**:
- Detector initialization and configuration
- Language support (all Arabic variants + fallback)
- EOU prediction (high/low probability, edge cases, error handling)
- `load()` factory with threshold overrides
- Constants validation (model ID, supported languages)

### Integration Tests (requires model download)

Loads the real HuggingFace model and runs predictions on Saudi dialect samples:

```bash
pytest tests/test_integration.py -m slow -v -s
```

**Includes**:
- 12 complete utterance tests (greetings, questions, requests, closings)
- 10 incomplete utterance tests (trailing phrases, hesitations, fillers)
- Edge cases (short text, single chars, punctuation)
- Threshold analysis report at 0.5, 0.7, 0.8, 0.9, 0.95, 0.98

---

## Evaluation

Standalone evaluation script with proper ML metrics on Saudi dialect test data (40 labeled samples):

```bash
# Evaluate at default threshold
python evaluate.py

# Sweep thresholds to find optimal operating point
python evaluate.py --sweep

# Detailed per-sample predictions
python evaluate.py --threshold 0.98 --detailed

# Export results to JSON
python evaluate.py --sweep --json results.json
```

**Output**:
```
============================================================
  EVALUATION REPORT — Threshold: 0.98
============================================================
  Accuracy:  75.0%
  Precision: 85.0%
  Recall:    65.0%
  F1 Score:  73.6%
  Avg Latency: 45.2ms

  Confusion Matrix:
                Predicted EOU    Predicted ~EOU
  Actual EOU      TP=13           FN=7
  Actual ~EOU     FP=3            TN=17
============================================================
```

**Test categories**: greetings, questions, requests, closings, confirmations, incomplete phrases, trailing phrases, hesitations, fillers.

---

## Deployment

### Docker

Build and run with pre-baked model weights (no download at runtime):

```bash
# Build image (downloads model during build)
docker build -t arabic-eou .

# Run with your API keys
docker run --env-file .env.local -p 8080:8080 arabic-eou
```

### Docker Compose

```bash
docker compose up --build
```

For GPU support, uncomment the `deploy.resources` section in `docker-compose.yml`.

---

## CI/CD

Automated pipeline via GitHub Actions on every push/PR to `main`:

| Stage | What It Does | Duration |
|-------|-------------|----------|
| **Lint** | `ruff check` + `ruff format --check` | ~6s |
| **Test** | `pytest` across Python 3.10, 3.11, 3.12 | ~1m30s |
| **Build** | `python -m build` + `twine check` | ~14s |

The build stage produces a distributable `.whl` package uploaded as a GitHub Actions artifact.

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
Python: 3.10+
livekit-agents>=1.3.0
livekit-plugins-turn-detector>=1.3.0
transformers>=4.45.0
torch>=2.0.0
huggingface-hub>=0.20.0
```

**Dev dependencies**: `pytest`, `pytest-asyncio`, `ruff`, `black`

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
