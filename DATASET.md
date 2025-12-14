# Dataset Documentation

## Source Dataset

**SADA 2022 - Saudi Dialectal Arabic Corpus**

- **Source**: [Kaggle - SADA 2022](https://www.kaggle.com/datasets/sdaiancai/sada2022/data?select=train.csv)
- **Original Purpose**: Saudi dialectal Arabic conversation dataset
- **License**: Available on Kaggle under open license
- **Size**: Large-scale conversational Arabic corpus

### Citation

```bibtex
@dataset{sada2022,
  title={SADA 2022: Saudi Dialectal Arabic Corpus},
  author={SDAIA},
  year={2022},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/sdaiancai/sada2022}
}
```

## Data Processing Methodology

### 1. Data Selection

From the SADA 2022 `train.csv`, we extracted conversational utterances that represent natural Arabic speech patterns. Example utterance:

```
"الدوخة مشيت مشيت وأنا ما أنا شايف ولا شيء وقبل ما أطيح ثاني مرة وأموت لحقني بن فهره آه أموت آخ"
```

### 2. Creating Complete Utterances (EOU)

**Complete utterances** represent finished thoughts where the speaker has completed their turn. These were used as-is from the SADA dataset.

**Examples**:
- `"شكرا جزيلا"` (Thank you very much)
- `"كيف حالك اليوم؟"` (How are you today?)
- `"أنا بخير والحمد لله"` (I'm fine, thank God)

**Total Complete Samples**: 20,194

### 3. Creating Incomplete Utterances (Non-EOU)

To train the model to recognize incomplete speech, we systematically truncated complete utterances at natural breakpoints:

**Truncation Strategy**:
1. **Mid-sentence cuts**: Breaking at conjunctions, prepositions
2. **Multiple breakpoints**: 40%, 60%, 80% of utterance length
3. **Linguistic markers**: Cutting after words like "و" (and), "لأن" (because), "إذا" (if)

**Example Transformations**:

| Original (Complete) | Truncated (Incomplete) | Breakpoint |
|-------------------|----------------------|-----------|
| "الدوخة مشيت مشيت وأنا ما أنا شايف ولا شيء" | "الدوخة مشيت مشيت و" | After conjunction |
| "وقبل ما أطيح ثاني مرة وأموت" | "وقبل ما أطيح" | Mid-phrase |
| "لحقني بن فهره آه أموت آخ" | "لحقني بن فهره" | Before hesitation |

**Total Incomplete Samples**: 37,281 (generated from 18,641 source utterances with ~2 variants each)

### 4. Arabic Edge Cases

We added **1,433 edge case samples** to handle Arabic-specific conversational patterns that are often misclassified:

#### Hesitations (Non-EOU) - 671 samples

These indicate the speaker is **thinking and will continue**:

```
اممممممم    (ummm...)
يعني        (you know...)
امم يعني    (umm like...)
خلاص بس     (okay but...)
طيب و       (okay and...)
اه          (ahh...)
هممم        (hmmm...)
```

**Dialectal Variations**:
- Gulf: `ياخي` (oh man), `اي والله` (yes I swear)
- Egyptian: `يعني ايه` (what do you mean)
- General: `والله يا` (I swear), `بس يعني` (but like)

#### Closures (EOU) - 762 samples

These indicate the speaker has **finished their turn**:

```
شكرا         (thank you)
تمام         (perfect)
نعم          (yes)
لا           (no)
ماشي         (alright)
حاضر         (okay)
مع السلامة    (goodbye)
ان شاء الله   (God willing)
```

**Dialectal Variations**:
- Gulf: `زين` (good), `اي` (yes), `لا والله` (no really)
- Egyptian: `ايوه` (yeah), `ماشي` (alright)
- Levantine: `يسلمو` (thanks), `تمام` (perfect)

### 5. Final Dataset Composition

```
Total Samples: 57,475
├── Complete (EOU): 20,194 (35.1%)
│   ├── Original utterances: 19,432
│   └── Edge case closures: 762
│
└── Incomplete (Non-EOU): 37,281 (64.9%)
    ├── Generated truncations: 36,610
    └── Edge case hesitations: 671

Balance Ratio: 1.85:1 (incomplete:complete)
```

### 6. Data Format

Training data follows the instruction-tuning format:

**Complete Utterance** (predict `<|im_end|>`):
```json
{
  "instruction": "",
  "input": "<|im_start|>user\nشكرا جزيلا",
  "output": "<|im_end|>"
}
```

**Incomplete Utterance** (no prediction):
```json
{
  "instruction": "",
  "input": "<|im_start|>user\nاممممممم",
  "output": ""
}
```

## Data Quality

### Preprocessing Steps

1. **Cleaning**: Removed utterances with:
   - Less than 3 words (too short to be meaningful)
   - More than 50 words (too long for turn detection)
   - Invalid Unicode characters

2. **Deduplication**: Removed exact duplicates

3. **Validation**: Manual inspection of edge cases for accuracy

### Limitations

1. **Dialect Bias**: Primarily Gulf/Saudi dialects from SADA 2022
2. **Truncation Artifacts**: Some generated incomplete samples may not perfectly reflect natural speech interruptions
3. **Edge Case Coverage**: Limited to common conversational patterns

### Future Improvements

- Add North African (Maghrebi) dialectal variations
- Include more spontaneous speech hesitations
- Add code-switching patterns (Arabic-English)
- Incorporate longer conversational context (multi-turn)

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 57,475 |
| Complete (EOU) | 20,194 (35.1%) |
| Incomplete (Non-EOU) | 37,281 (64.9%) |
| Edge Cases | 1,433 (2.5%) |
| Avg. Words/Sample (Complete) | 8.3 |
| Avg. Words/Sample (Incomplete) | 4.7 |
| Unique Vocabulary | ~12,000 words |
| Source Files | SADA 2022 train.csv |

## Reproducibility

The dataset can be regenerated using:

```bash
python training/prepare_dataset.py
```

This will:
1. Load SADA 2022 CSV
2. Generate truncated incomplete samples
3. Add Arabic edge cases
4. Create balanced training set
5. Save to `data/train.json`

---

**Dataset prepared for**: Arabic End-of-Utterance (EOU) Detection
**Intended use**: Fine-tuning turn detection models for Arabic voice agents
**Version**: 1.0 (December 2024)
