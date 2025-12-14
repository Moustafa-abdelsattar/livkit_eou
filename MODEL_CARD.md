# Model Card: Arabic Turn Detector

## Model Details

### Model Description

Fine-tuned End-of-Utterance (EOU) detection model for Arabic conversational agents, optimized for Modern Standard Arabic and Gulf dialects.

- **Developed by**: Moustafa Abdelsattar
- **Model type**: Causal Language Model (EOU Detection)
- **Language**: Arabic (ar, ar-SA, ar-EG, ar-AE, ar-KW, and other variants)
- **Base Model**: [livekit/turn-detector](https://huggingface.co/livekit/turn-detector) (Qwen2-0.5B)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **License**: Apache 2.0
- **Repository**: [GitHub](https://github.com/Moustafa-abdelsattar/livkit_eou)

### Model Architecture

```
Base Model: Qwen2-0.5B (524M parameters)
├── Fine-tuning: LoRA Adapters
│   ├── Rank (r): 32
│   ├── Alpha: 64
│   ├── Dropout: 0.05
│   └── Target Modules: q_proj, k_proj, v_proj, o_proj,
│                        gate_proj, up_proj, down_proj
└── Trainable Parameters: ~18M (3.4% of base model)
```

## Training Details

### Dataset

**Source**: [SADA 2022](https://www.kaggle.com/datasets/sdaiancai/sada2022) - Saudi Dialectal Arabic Corpus

**Processing**:
- Extracted conversational utterances from SADA 2022
- Generated incomplete samples via systematic truncation at linguistic breakpoints
- Added 1,433 Arabic-specific edge cases (hesitations vs closures)

**Final Composition**:
```
Total Samples: 57,475
├── Complete (EOU): 20,194 (35.1%)
│   ├── Original SADA utterances: 19,432
│   └── Edge case closures: 762
│
└── Incomplete (Non-EOU): 37,281 (64.9%)
    ├── Truncated utterances: 36,610
    └── Edge case hesitations: 671
```

See [DATASET.md](DATASET.md) for full dataset documentation.

### Training Procedure

**Platform**: Google Colab with T4 GPU (16GB VRAM)

**Framework**: LLaMA Factory

**Hyperparameters**:
```yaml
learning_rate: 1.0e-4
batch_size: 16
gradient_accumulation_steps: 2
effective_batch_size: 32
num_epochs: 3
optimizer: AdamW
lr_scheduler: cosine
warmup_ratio: 0.1
weight_decay: 0.01
max_sequence_length: 256
fp16: false
bf16: true (on T4)
```

**Training Time**: ~28 minutes on T4 GPU

**Training Steps**:
- Total steps: ~5,400
- Warmup steps: ~540
- Logging interval: 50 steps
- Evaluation interval: End of epoch

### Training Results

**Final Metrics** (Epoch 3):

| Metric | Value |
|--------|-------|
| **Training Loss** | 0.0712 |
| **Validation Loss** | 0.0856 |
| **Training Accuracy** | 97.2% |
| **Validation Accuracy** | 96.4% |
| **Training Perplexity** | 1.074 |
| **Validation Perplexity** | 1.089 |

**Learning Curve**:

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|-----------|---------|
| 1 | 0.1842 | 0.1634 | 92.3% | 91.8% |
| 2 | 0.0945 | 0.0991 | 95.8% | 95.1% |
| 3 | 0.0712 | 0.0856 | 97.2% | 96.4% |

**Performance by Category** (Validation Set):

| Category | Precision | Recall | F1-Score | Samples |
|----------|-----------|--------|----------|---------|
| Complete (EOU) | 0.968 | 0.982 | 0.975 | 2,019 |
| Incomplete (Non-EOU) | 0.986 | 0.975 | 0.980 | 3,728 |
| **Macro Avg** | **0.977** | **0.979** | **0.978** | **5,747** |
| **Weighted Avg** | **0.979** | **0.978** | **0.978** | **5,747** |

**Edge Case Performance** (Manual Test Set):

| Category | Accuracy | Test Samples |
|----------|----------|--------------|
| Closures (شكرا، تمام، نعم) | 100.0% | 12 |
| Questions (كيف حالك؟) | 100.0% | 6 |
| Hesitations (اممم، يعني) | 16.7% | 6 |
| Incomplete phrases (إذا، و) | 8.3% | 12 |
| **Overall** | **54.3%** | **35** |

### Known Issues

⚠️ **Model Over-prediction**:
- The model predicts high probabilities (0.85-0.99) for most inputs
- Poor performance on incomplete utterances and hesitations
- Root cause: Training used empty outputs for incomplete samples, providing weak negative signal

**Recommended Workaround**:
- Use higher threshold (0.98 instead of 0.5) for production
- Achieves 65.7% accuracy on edge cases with threshold=0.98

**Future Fix**:
- Retrain with explicit continuation token instead of empty output
- Increase incomplete sample ratio to 5:1

## Intended Use

### Primary Use Case

Real-time turn detection for Arabic conversational AI agents, specifically:
- Voice assistants speaking Modern Standard Arabic or Gulf dialects
- Customer service bots handling Arabic conversations
- Interactive voice response (IVR) systems

### Direct Use

```python
from livekit.agents import AgentSession
from livekit_plugins_arabic_turn_detector import load

session = AgentSession(
    stt=groq.STT(language="ar"),
    llm="openai/gpt-4o",
    tts="cartesia/sonic-3",
    turn_detection=load(threshold=0.98),  # High threshold recommended
    vad=silero.VAD.load(),
)
```

### Out-of-Scope Use

❌ Not suitable for:
- Text classification or sentiment analysis
- Machine translation
- Speech recognition (use separate STT model)
- Non-Arabic languages
- North African dialects (limited training data)

## Evaluation

### Testing Methodology

**Test Set**: 35 manually curated Arabic utterances covering:
- Complete closures (15 samples)
- Questions (3 samples)
- Incomplete phrases (13 samples)
- Hesitations and fillers (4 samples)

**Metrics**:
- Accuracy: Percentage of correct EOU/non-EOU predictions
- Threshold analysis: Testing at 0.3, 0.5, 0.7, 0.9, 0.95, 0.98

### Results

**Accuracy by Threshold**:

| Threshold | Accuracy | Precision (EOU) | Recall (EOU) |
|-----------|----------|-----------------|--------------|
| 0.50 | 54.3% | 0.68 | 1.00 |
| 0.90 | 62.9% | 0.75 | 1.00 |
| 0.98 | 65.7% | 0.82 | 0.94 |

**Best Threshold**: 0.98 (recommended for production)

### Limitations

1. **Dialect Coverage**: Primarily Gulf/Saudi dialects, limited North African coverage
2. **Over-prediction**: High false positive rate on incomplete utterances
3. **Context Length**: Trained on single-turn context, may not leverage multi-turn conversation history
4. **Edge Cases**: Struggles with Arabic hesitations (اممم، يعني) despite dedicated training samples

## Environmental Impact

**Hardware**: 1x NVIDIA T4 GPU
**Training Time**: 28 minutes
**Cloud Provider**: Google Colab
**Carbon Emissions**: ~0.02 kg CO2eq (estimated)

## Technical Specifications

### Model Format

- **Primary**: PyTorch (`.bin` files with LoRA adapters)
- **Size**: 494 MB (merged model)
- **Precision**: BF16 (GPU), FP32 (CPU)

### Inference

**Latency**:
- GPU (T4): ~20ms per prediction
- CPU: ~50ms per prediction

**Memory**:
- Model: ~500 MB RAM
- Peak: ~2 GB RAM (with tokenizer and overhead)

**Throughput**: ~50 predictions/second (GPU), ~20 predictions/second (CPU)

### Deployment

**Recommended Environment**:
```
Python: 3.9+
PyTorch: 2.0+
Transformers: 4.45.2+
PEFT: 0.18.0+
```

**Docker** (optional):
```dockerfile
FROM python:3.11-slim
RUN pip install livekit-plugins-arabic-turn-detector
```

## Citation

```bibtex
@software{arabic_turn_detector_2024,
  author = {Abdelsattar, Moustafa},
  title = {Arabic Turn Detector for LiveKit},
  year = {2024},
  url = {https://github.com/Moustafa-abdelsattar/livkit_eou},
  note = {Fine-tuned on SADA 2022 dataset}
}

@dataset{sada2022,
  title = {SADA 2022: Saudi Dialectal Arabic Corpus},
  author = {SDAIA},
  year = {2022},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/sdaiancai/sada2022}
}
```

## Contact

- **Repository**: https://github.com/Moustafa-abdelsattar/livkit_eou
- **Issues**: https://github.com/Moustafa-abdelsattar/livkit_eou/issues
- **Model**: https://huggingface.co/Moustafa3092/livekit-turn-detector-arabic

## Changelog

### Version 1.0 (December 2024)

- Initial release
- Fine-tuned on 57,475 Arabic samples
- LoRA rank=32, alpha=64
- Training loss: 0.0712
- Validation accuracy: 96.4%

---

**Model Card last updated**: December 2024
