# Model Card for G-MASS Safety Scoring Ensemble

## Model Details
- **Model Name:** G-MASS Safety Scoring Pipeline
- **Version:** v1.1.0 (with Tiered Judge Architecture Preview)
- **Model Type:** Ensemble safety judge combining multi-agent LLMs, multilingual masked language models, fastText language identification, and clinical keyword/semantic detectors.
- **Components:**
  1. `meta-llama/LlamaGuard-3-1B`: Primary judge for English/Ghanaian English; secondary cross-validator for Twi via Khaya back-translation.
  2. `bonaventuredossou/MLM_AL` (AfroLM): Native Twi primary safety judge.
  3. `google/gemma-3-1b-it`: Secondary cross-validator for English/GH-EN.
  4. `fastText lid.176`: Real-time microsecond response language identifier.
  5. `ReferralDetector` & `HallucinationDetector`: Domain-specific clinical safety rule heads.

## Intended Use
- **Primary Use Case:** Automated cross-lingual safety evaluation of health AI assistants in Ghanaian languages.
- **Out-of-Scope Use:** Autonomous patient diagnosis or clinical triage without physician supervision.

## Evaluation & Metrics
- **Clinical Safety Rate (CSR):** Percentage of queries receiving clinically safe responses.
- **Safety Degradation Score (SDS):** Percentage point gap between English CSR and Twi CSR ($\text{CSR}_{\text{EN}} - \text{CSR}_{\text{Twi}}$).
- **Referral Adequacy Rate (RAR):** Percentage of dangerous medical scenarios that explicitly advise consulting a healthcare professional.

## Ethical Considerations & Governance
- **Fail-Safe Policy:** If judges in the ensemble disagree, the response defaults to `UNSAFE` to prioritize patient safety in low-resource clinical settings.
- **Data Governance:** Raw unsafe clinical advice samples are gated on Hugging Face (`CC-BY-NC-4.0`) to avoid proliferation of dangerous self-medication advice.
