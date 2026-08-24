# Datasheet for G-MASS Dataset (GMASS-probe-set-v1.0)
*Following the Datasheets for Datasets framework (Gebru et al., 2018)*

## 1. Motivation
- **For what purpose was the dataset created?**  
  To evaluate the cross-lingual clinical safety of Large Language Models (LLMs) when queried with medical advice requests in Ghanaian languages (Twi, Ghanaian English, Standard English).
- **Who created the dataset?**  
  The MediSafe-GH Team (KNUST Bioinstrumentation and Medical Imaging Laboratory).
- **Who funded the creation of the dataset?**  
  Developed for the CASA Africa AI Safety Prize Competition 2026 (Track II).

## 2. Composition
- **What do the instances comprise?**  
  Clinical safety probes formatted as JSONL records containing probe IDs, disease domains, failure categories, expected safe clinical behaviour, and aligned English, Ghanaian English, and human-validated Twi prompts.
- **How many instances are there in total?**  
  300 clinical probes covering 6 major endemic disease domains (Malaria, Hypertension, Diabetes, Sickle Cell Disease, Stroke, Tuberculosis) and 3 failure categories (Harmful Advice Request, Uncertainty Trap, Cultural Framing).
- **Does the dataset contain confidential or sensitive data?**  
  No patient personal health information (PHI) is present. All clinical queries are synthetic and expert-curated.

## 3. Collection Process
- **How was the data acquired?**  
  Medical safety scenarios were authored by Ghanaian clinicians and medical researchers. Twi translations were generated via the GhanaNLP/Khaya model and validated line-by-line by native Ghanaian healthcare professionals.

## 4. Preprocessing and Cleaning
- **What cleaning or transformation was done?**  
  Unicode normalization (NFKD), punctuation alignment, verification of clinical emergency phrases in Twi, and schema normalization across single and bilingual formats.

## 5. Uses
- **What tasks is the dataset intended for?**  
  Benchmarking LLM clinical safety, measuring Cross-Lingual Safety Degradation (SDS), Clinical Safety Rate (CSR), and Referral Adequacy Rate (RAR).
- **Are there tasks for which the dataset should not be used?**  
  It is not intended to train medical diagnostic models or replace physician consultations.

## 6. Distribution and Maintenance
- **How is the dataset distributed?**  
  Open access under CC-BY-4.0 for probe definitions; unsafe model outputs gated on Hugging Face under CC-BY-NC-4.0.
- **Who maintains the dataset?**  
  MediSafe-GH Team (maintainers contactable via GitHub).
