# Nepali Speech-to-Text Project Guide

## Academic Reference & Self-Project Blueprint

---

**Document Purpose:** This guide serves as a dual-purpose resource: (1) a reference for educators designing coursework on low-resource ASR, and (2) a self-study project blueprint for students or researchers interested in building a Nepali speech-to-text system. The structure follows the assignment brief from the previous document while incorporating real-world research findings.

**Author:** [Professor Name]
**Institution:** Softwarica College of IT & E-Commerce / Coventry University
**Module Reference:** ST7088CEM Artificial Neural Networks
**Date:** July 2026

---

## 1. Executive Summary

**Project Scope:** Build an Automatic Speech Recognition (ASR) system for the Nepali language using a pre-trained multilingual model (Whisper) fine-tuned on Nepali speech data.

**Why This Project Matters:**
- Nepali is spoken by ~17 million native speakers but remains a low-resource language in AI
- Recent MoU between Digital India Bhashini and Kathmandu University aims to develop Nepali language datasets and AI resources
- State-of-the-art Nepali ASR systems achieve ~15% Word Error Rate (WER) on read speech
- Open challenges remain for conversational speech, code-switching, and noisy audio

**Target Audience:** Master's students in AI/ML, researchers in low-resource NLP, educators designing ASR coursework

**Estimated Time:** 6-8 weeks (with existing ML background)

---

## 2. Academic Context & Why This Project Is Valuable

### 2.1 The Low-Resource Language Challenge

High-resource languages (English, Mandarin) have abundant labeled speech data. Nepali, despite ~17 million speakers, has limited open resources for ASR. This creates a research gap where:

- **Pre-trained models** like Whisper perform poorly on Nepali without fine-tuning
- **Data scarcity** (~157K utterances in largest open corpus) limits model generalization
- **Dialectal variation** and code-switching add complexity
- **Community impact** is significant: ASR enables digital inclusion, education access, and preservation of linguistic heritage

### 2.2 Why Whisper Is the Right Starting Point

OpenAI's Whisper is a multilingual pre-trained model that already knows how to do speech recognition in 100+ languages. Fine-tuning it on Nepali speech adapts this general capability to the specific language.

| Model | Size (Parameters) | Nepali WER (In-domain) | Notes |
| :---- | :---------------- | :--------------------- | :---- |
| Whisper-Small | 244M | ~30% | Good starting point, lower compute |
| Whisper-Medium | 769M | **15.57%** | Best in-domain performance |
| Whisper-Large-v3 | 1.5B | Lower than Medium? | More compute, may overfit on small data |
| IndicWav2Vec | 94M | 14.89% | More efficient, CTC architecture |
| NepConformer | ~100M | **6.01% CER** | State-of-the-art for Nepali |

### 2.3 What Makes This Project Educational

| Learning Outcome | How This Project Delivers |
| :--------------- | :------------------------ |
| **Fine-tuning** | Adapt pre-trained model to low-resource language |
| **Data preparation** | Handle audio at 16kHz, text normalization (NFC), filtering short/long utterances |
| **Evaluation metrics** | Word Error Rate (WER) and Character Error Rate (CER) — standard in ASR |
| **Computational constraints** | Work within colab/Kaggle free tier limits |
| **Ethical AI** | Discuss bias, data limitations, community impact |

---

## 3. Recommended Project Pipeline

### 3.1 Phase 1: Dataset Selection & Preparation

#### Dataset Options

| Dataset | Size | Access | Best Use |
| :------ | :--- | :----- | :------- |
| **OpenSLR SLR54** | ~157K utterances, ~165 hours | [OpenSLR](https://openslr.magicdatatech.com/54/) | Primary training data |
| **Common Voice Nepali** | ~XX hours | Mozilla Data Collective | Evaluation/out-of-domain test |
| **Nepali Speech-to-Text Dataset** | Combined corpus | [Hugging Face](https://huggingface.co/datasets/amitpant7/nepali-speech-to-text) | Alternative/Supplementary |
| **FLEURS (ne_np)** | ~XX hours | Hugging Face | Out-of-domain test |

**Recommended:** Use SLR54 for training (80/10/10 split), Common Voice for evaluation.

#### Data Preparation Steps

```python
# Pseudocode for data preparation
- Resample audio to 16 kHz mono
- Normalize transcriptions (NFC unicode normalization)
- Filter utterances: 0.5s < duration < 30s
- Split: train (80%), validation (10%), test (10%)
- Ensure speaker-disjoint splits (no speaker appears in both train and test)
```

**Why these steps matter:**
- **16 kHz sampling**: Whisper's training standard
- **NFC normalization**: Prevents mismatched Devanagari characters from inflating error rates
- **Duration filters**: Removes corrupted/truncated files
- **Speaker-disjoint splits**: Realistic evaluation of generalization

### 3.2 Phase 2: Model Selection & Fine-Tuning

#### Option A: Whisper Fine-Tuning (Recommended for First Project)

```python
# Minimal fine-tuning setup
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-nepali",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # To handle limited GPU memory
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=2000,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=100,
    save_steps=500,
)
```

**Key Parameters (from successful models):**

| Parameter | Whisper Small | Whisper Medium | Why |
| :-------- | :------------ | :------------- | :-- |
| Learning Rate | 1e-4 | 1e-5 | Lower for larger models |
| Batch Size | 2 + grad_accum 4 | 2 | Memory constraints |
| Epochs | 30 | ~6 | Early stopping avoids overfitting |
| Optimizer | AdamW | AdamW | Standard |
| LoRA Rank | N/A | 64 | Parameter-efficient |

#### Option B: CTC-Based Models (More Efficient)

The IndicWav2Vec model achieves **14.89% WER** with only **94M parameters** — significantly smaller than Whisper-Medium (769M).

```python
from transformers import Wav2Vec2ForCTC, AutoProcessor

model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec_v1_hindi")
# Fine-tune with CTC loss on Nepali data
```

**Comparison of Approaches:**

| Approach | Compute | Performance | Best For |
| :------- | :------ | :---------- | :------- |
| Whisper Fine-tuning | High (colab T4) | Better out-of-domain | General flexibility |
| IndicWav2Vec | Low (CPU/GPU) | Good in-domain | Efficiency focus |
| NepConformer | Medium | State-of-the-art (6% CER) | Research/Publication |

### 3.3 Phase 3: Evaluation

#### Metrics

**Word Error Rate (WER)** — the primary metric for ASR:
```
WER = (Substitutions + Deletions + Insertions) / Total Words
```
Lower is better. Under 20% is considered good for low-resource languages.

**Character Error Rate (CER)** — more sensitive to Devanagari script differences:
```
CER = (Substitutions + Deletions + Insertions) / Total Characters
```

**Expected Results:**
- **In-domain (SLR54 test):** WER 15-20% for Whisper-Medium/IndicWav2Vec
- **Out-of-domain (Common Voice):** WER 40-50% (much harder)
- **Conversational/Code-switched:** Likely higher (open research problem)

#### Error Analysis Checklist

| Error Type | What to Look For |
| :--------- | :--------------- |
| **Phonetic confusions** | Similar-sounding Devanagari chars confused |
| **OOV/Uncommon words** | Rare vocabulary not in training data |
| **Background noise** | Degradation on noisy audio |
| **Dialect/Accent** | Performance variation across speakers |
| **Code-switching** | English words in Nepali speech |

---

## 4. Deliverables & Report Structure

### 4.1 Report Outline

| Section | Content | Word Count |
| :------ | :------ | :--------- |
| **Introduction** | Problem statement, why Nepali ASR matters, objectives | 500 |
| **Background** | ASR architectures, low-resource techniques, related work | 700 |
| **Methodology** | Dataset description, preprocessing, model architecture, training details | 900 |
| **Experiments** | Training runs, hyperparameter choices, validation curves | 800 |
| **Results & Discussion** | WER/CER tables, comparison across models, error analysis | 1000 |
| **Conclusion** | Key findings, limitations, future work | 300 |
| **References** | 20+ academic sources | 300 |

### 4.2 Appendices

- Project proposal (from Phase 0)
- Complete source code (notebooks, scripts)
- Screenshots of training runs (colab/kaggle logs, device info visible)
- Sample outputs with reference transcripts for qualitative analysis

---

## 5. Challenges to Expect (And How to Handle Them)

### 5.1 Computational Constraints

| Problem | Solution |
| :------ | :------- |
| **GPU memory limit** | Use gradient_accumulation_steps; consider LoRA adapter training |
| **Long training time** | Early stopping (3 eval rounds without improvement); reduce epochs if overfitting |
| **Colab disconnects** | Save checkpoints frequently; use Kaggle for longer runs |

### 5.2 Data Issues

| Problem | Solution |
| :------ | :------- |
| **Data format mismatch** | Ensure WAV, 16kHz mono |
| **Transcription noise** | Manual QC; use dataset citation guidelines |
| **Limited data** | Data augmentation (SpecAugment) — critical for low-resource |

### 5.3 Evaluation Pitfalls

| Problem | Solution |
| :------ | :------- |
| **Inflated metrics** | Match normalization (NFC) on both reference and hypothesis |
| **Overly optimistic** | Test on held-out unseen speakers; evaluate on both in-domain and out-of-domain sets |
| **Language model missing** | Recognize that no LM during decoding may understate possible performance |

---

## 6. Citation Requirements

### 6.1 Key Papers to Reference

1. **Kjartansson et al., 2018** — "Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali" (SLTU)
2. **NepConformer (2026)** — "A Conformer-Based Nepali Automatic Speech Recognition System" (LNNS)
3. **Sharma et al., 2026** — "Nwāchā Munā: A Devanagari Speech Corpus and Proximal Transfer Benchmark for Nepal Bhasha ASR" (arXiv)
4. **Devkota, 2026** — "Beyond Monolingual: Leveraging Multilingual Pre-trained Models for End-to-End Nepali-English Code-mixed Speech Recognition" (M.Sc. Thesis, TU)
5. **Paudel et al., 2026** — "Comparative Analysis of Multilingual Pre-trained Models for Nepali Automatic Speech Recognition" (referenced in model cards)

### 6.2 Suggested Citation Format

```bibtex
@inproceedings{kjartansson-etal-sltu2018,
  title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese,  Sinhala, Nepali, and Bangladeshi Bengali}},
  author = {Oddur Kjartansson and Supheakmungkol Sarin and Knot Pipatsrisawat and Martin Jansche and Linne Ha},
  booktitle = {Proc. The 6th Intl. Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)},
  year  = {2018},
  pages = {52--55},
  URL   = {http://dx.doi.org/10.21437/SLTU.2018-11}
}
```

---

## 7. Academic Integrity Notice

This document is designed for:
- **Educators** to design coursework and reference projects
- **Students** as a planning and learning resource
- **Researchers** as a self-study blueprint

**Important caveats:**
- All experiments, code writing, and data preparation must be **your own work** for any graded submission
- Any use of code from public repositories **must be clearly attributed** with comments and citations
- Results must come from **your own runs** — not copied from model cards
- The **self-project** aspect means you execute the work; this document provides guidance, not deliverables

---

## 8. Further Reading & Resources

### 8.1 Datasets
- **OpenSLR SLR54**: https://openslr.magicdatatech.com/54/ — ~157K utterances, 165 hours
- **Common Voice Nepali**: Mozilla Data Collective — evaluation/out-of-domain
- **Nepali Speech-to-Text Dataset** (Hugging Face): https://huggingface.co/datasets/amitpant7/nepali-speech-to-text
- **ELRA Nepali Spoken Corpus**: 31 hours, professional recordings

### 8.2 Models
- **WhisperV3_Nepali_v0.5** (Hugging Face): LoRA fine-tuned Whisper Large V3
- **nepali-asr-whisper-medium** (Hugging Face): Benchmark results on three test sets
- **nepali-asr-indicwav2vec** (Hugging Face): CTC-based efficient model
- **Code-mixed model**: Nepali-English with language model decoding

### 8.3 Tutorials
- [Hugging Face Whisper Fine-tuning Guide](https://huggingface.co/blog/fine-tune-whisper)
- [OpenSLR Data Access Guide](https://openslr.magicdatatech.com/54/)
- [Kaggle Notebook Example](https://www.kaggle.com/code/amieeeet/whispher-finetune-on-np)

---

## 9. Version History

| Version | Date | Changes |
| :------ | :--- | :------ |
| 1.0 | 2026-07-03 | Initial academic reference document |

---

*This document is intended for educational and research purposes. All use must comply with institutional academic integrity policies.*