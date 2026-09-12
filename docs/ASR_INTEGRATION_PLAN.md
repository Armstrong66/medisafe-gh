# G-MASS Audio & ASR Integration Plan: Voice-Driven Medical Safety Evaluation
**MediSafe-GH · Track II Africa AI Safety Prize · KNUST Bioinstrumentation & Medical Imaging Laboratory**

---

## 1. Executive Summary & Clinical Rationale

In Ghana and across Sub-Saharan Africa, a significant portion of patient populations interact with digital health tools, community health nurses, and voice bots via spoken language rather than written text. Akan (Twi) is predominantly an oral language for healthcare consultations. Illiteracy, poor typing fluency on smartphones, and urgent clinical emergencies make voice input the natural interaction modality in rural and semi-urban health posts (CHPS compounds).

Integrating **Automatic Speech Recognition (ASR)** into G-MASS will expand the evaluation suite from text-only safety screening to **End-to-End Multimodal Voice Safety Screening**, enabling detection of clinical safety hazards, misdiagnoses, and dangerous dosage recommendations directly from patient voice queries.

---

## 2. Model Architecture Selection & Trade-Offs

We evaluate three potential architectural paths for local and edge Ghanaian ASR:

| Model Candidate | Parameter Count | Native Twi Support | Inference Latency | Hallucination Risk | Edge / Web Suitability | Recommendation |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **OpenAI Whisper (tiny/base/small)** | 39M – 244M | ⚠️ Weak zero-shot (requires fine-tuning) | Low–Medium (~0.8s on CPU) | High (autoregressive hallucination on silence) | ⭐⭐⭐ (Good when fine-tuned + quantized) | **Secondary / Cloud Option** |
| **Meta MMS (`mms-300m` / `mms-1b`)** | 300M – 1B | ✅ **Native** (`aka-twi`, `aka-fante`) | Very Low (~0.2s, non-autoregressive CTC) | **Zero** (no language model hallucination) | ⭐⭐⭐⭐⭐ (Ideal for local / offline edge) | **Primary Recommended Base** |
| **GhanaNLP Khaya Audio API** | Cloud-hosted | ✅ High Accuracy (Commercial) | Network dependent (~1.2s) | Low | ⭐⭐⭐⭐ (Zero local compute, needs API key) | **Hosted Fallback Option** |

### Why Meta MMS over standard Whisper for G-MASS Core:
1. **Zero Hallucination Guarantee**: Meta MMS uses a Connectionist Temporal Classification (CTC) acoustic head rather than an autoregressive decoder. In clinical safety testing, an ASR model must **never hallucinate** phantom words during pauses or coughing spells.
2. **Native Akan Pre-training**: Meta MMS includes pre-trained acoustic weights for Twi and Fante, whereas standard Whisper was not officially pre-trained on Akan text corpora and frequently outputs nonsensical English phonetic approximations.
3. **Low Compute Footprint**: `mms-300m` runs comfortably within G-MASS Tier 1/2 compute envelopes (<4GB RAM on CPU).

---

## 3. Data Strategy & Corpus Requirements for Fine-Tuning

To achieve clinical-grade Word Error Rate (WER < 15%) on spoken Ghanaian medical queries, fine-tuning must be trained on a specialized three-tier corpus:

### A. General-Domain Acoustic Foundation (30–60 Hours)
- **Mozilla Common Voice (MCV) Akan (Twi)**: ~35 hours of validated native speaker recordings.
- **FLEURS Akan Subset**: Aligned audio recordings with multi-speaker acoustic diversity.
- **ALFFA Corpus**: African Language Speech Technology open acoustic data.

### B. In-Domain Clinical Consultation Audio (15–20 Hours)
- **Symptom Descriptions**: Native speakers recording queries for the 6 G-MASS endemic disease domains (Malaria, Hypertension, Sickle Cell, Diabetes, Stroke, TB).
- **Acoustic Diversity**: Multi-generational speakers (elders, mothers, youth) representing Asante Twi, Akuapem Twi, and Fante dialects.
- **Urgent & Emotional Prosody**: Speech spoken in hurried, distressed, or breathless tones to mirror real clinical crisis presentations.

### C. Ghanaian Code-Switching Speech (Crucial for Health in Ghana)
- Ghanaians frequently embed English pharmaceutical terms and clinical jargon within Twi syntax:
  - *Example*: `"Me papa BP tablet no asa, na ne ti pae no denneennen."`
  - *Example*: `"Manya fever na me sugar level no akɔ soro."`
- Fine-tuning data must include bilingual code-switched audio-text pairs to avoid ASR phonetic mutilation of drug names (e.g., *Artemether*, *Amlodipine*, *Metformin*).

---

## 4. End-to-End Voice Safety Pipeline Architecture

```
[ Patient Voice / Microphone / Audio File (.wav/.mp3) ]
                     │
                     ▼
          [ 16 kHz Mono Resampling ]
                     │
                     ▼
          [ Meta MMS-300M (Akan/EN) ] ➔ Outputs Raw Transcript + Audio Confidence
                     │
                     ▼
       [ fastText Response Router (lid.176.ftz) ]
                     │
                     ▼
    [ G-MASS Target Model Under Test (Zero-Shot) ]
                     │
                     ▼
       [ Multi-Agent Ensemble Judges ]
       ├── LlamaGuard-3-1B (Policy Safety)
       ├── AfroLM (Native Akan Morphology)
       ├── ReferralDetector (Emergency Directives)
       └── HallucinationDetector (Clinical Hedge)
                     │
                     ▼
      [ Voice Safety Report: CSR, SDS, RAR, Gate Verdict ]
```

---

## 5. Phased Implementation Roadmap

- **Phase 1 (v1.2.0)**: Add audio recording / `.wav` upload widget in Gradio using GhanaNLP Khaya Audio API or hosted Whisper endpoint.
- **Phase 2 (v1.3.0)**: Package local `mms-300m-aka` CTC pipeline via Hugging Face `transformers` pipeline for offline edge execution.
- **Phase 3 (v2.0.0)**: Release **G-MASS Voice Bench** containing 300 verified bilingual spoken clinical probes under CC-BY-4.0.
