# Project Update – CS595 NLP Fall 2025
*Nikhil Gunaratnam, Pranav Rakasi*  
University of Michigan  
:contentReference[oaicite:0]{index=0}

---

## 1. Introduction
Assigning ICD codes from clinical text is essential for billing, research, and patient record management, but the process is largely manual, time-consuming, and prone to inconsistency. This project explores automatic ICD code prediction from discharge summaries, framing it as a multi-label text classification task where each note may map to multiple diagnosis codes.

Our goals are twofold:  
1. Compare different NLP architectures to identify which designs perform best on long, complex clinical documents.  
2. Analyze how sections of patient documentation correlate with specific ICD codes, improving model interpretability and alignment with clinical reasoning.

We use the MIMIC-III and MIMIC-IV datasets, which pair free-text ICU discharge notes with structured ICD codes. By combining performance benchmarking with interpretability analysis, our work aims to reduce clinician coding effort, enhance coding consistency, and advance research on long-document understanding in clinical NLP.

---

## 2. Data

### 2.1 Primary Dataset: MIMIC-III
MIMIC-III v1.4 contains 60,000+ ICU admissions (2001–2012). We extract *discharge summaries* from NOTEEVENTS and join ICD-9 codes from DIAGNOSES_ICD using `hadm_id`. Documents are long (1,500–2,000 tokens, sometimes 5,000+) and highly multi-label (avg. ~14 codes). The label distribution is extremely imbalanced. We preserve section headers (HPI, Hospital Course, etc.) to support correlation analysis between document structure and ICD codes.

### 2.2 External Dataset: MIMIC-IV
MIMIC-IV extends to 2008–2019 and includes ICD-9 and ICD-10. Documentation style is more modern. We extract discharge summaries via the same pipeline. This enables robustness testing under distribution shift.

### 2.3 Illustrative Examples

| Discharge Summary Snippet | ICD-9 Codes |
|---------------------------|-------------|
| “Patient admitted with chest pain radiating to the left arm…” | 410.71 (Acute MI), 401.9 (Hypertension) |
| “History notable for poorly controlled type 2 diabetes…” | 250.00 (Diabetes), 272.4 (Hyperlipidemia) |

### 2.4 Summary
MIMIC-III is the core dataset for modeling and correlation analysis; MIMIC-IV evaluates generalization. This supports our two project goals: model comparison and understanding section-level diagnostic relevance.

---

## 3. Related Work
Early neural ICD coding work (Shi et al.) showed improvement over bag-of-words but struggled with extreme label imbalance. Mullenbach et al. introduced CAML, using per-label attention for both accuracy and interpretability. Li & Yu extended this using hierarchical attention and ICD ontology relationships.

Later work (Xie & Xing) used residual CNNs with hierarchical regularization to scale to thousands of labels. Recent studies (Vu et al.) show pretrained transformers (BioBERT, ClinicalBERT) provide significant gains for long-document modeling.

Gaps in prior work:  
● Limited cross-dataset evaluation (mostly MIMIC-III only).  
● Limited analysis of how clinical sections influence predictions.  

Our project addresses both.

---

## 4. Methodology

### 4.1 Preprocessing
We segment documents by section header, lowercase, remove non-informative tokens, and tokenize with domain-specific tokenizers. Truncation:  
- CAML + hierarchical models: 4,096 tokens  
- LED: up to 16,384 tokens  

### 4.2 Model 1: CAML
1D CNN encoder + label-specific attention. Scalable and interpretable.

### 4.3 Model 2: Longformer Encoder-Decoder (LED)
Uses sliding-window + global attention to model sequences longer than 8k tokens. Classification over `[CLS]` or learned label queries. Captures long-range structure across sections.

### 4.4 Model 3: Hierarchical Attention (Future Work)
Section-level and token-level GRU encoders with two-level attention. Mirrors clinical documentation structure.

### 4.5 Interpretability
We use Integrated Gradients + attention weights to extract token-level and section-level saliency. Heatmaps support correlation studies.

---

## 5. Evaluation

Our evaluation measures both **prediction quality** and **interpretability**, tested on-domain (MIMIC-III) and cross-domain (zero-shot MIMIC-IV).

### 5.1 Prediction Performance
Metrics:
- Micro-F1, Macro-F1  
- Precision@k (k=5,10)  
- Label-wise ROC-AUC for top 50 frequent codes  

Handling rare labels:
- Stratify scores by label frequency (head, medium, tail).  
- Examine per-label recall drop-off.

Cross-dataset evaluation:
- Train on MIMIC-III → eval on MIMIC-IV.  
- Tests robustness to temporal/stylistic shift.

### 5.2 Interpretability & Explainability
For CAML and LED:  
- Token saliency via Integrated Gradients  
- Highlight Overlap@k (how many top-k tokens fall in relevant sections)  
- Attention entropy (focus vs dispersion)  

Qualitative visualizations pair predicted codes, gold labels, and heatmaps.

---

## 6. Work Plan

**Week 1:** Finalize extraction, cleaning, sectioning, and ICD linkage for both datasets.  
**Week 2:** Implement + train CAML, hyperparameter tuning, compute F1 + P@k.  
**Week 3:** Implement LED, compare to CAML, eval on MIMIC-IV.  
**Week 4:** Interpretability analysis, generate figures, finalize report.

---

## References
(References preserved exactly as in source PDF.)  
[1] Beltagy et al., Longformer  
[2] Li & Yu, Hierarchical Attention  
[3] Mullenbach et al., CAML  
[4] Shi et al., Deep Learning for ICD Coding  
[5] Sundararajan et al., Integrated Gradients  
[6] Vu et al., Pretrained LMs for ICD Coding  
[7] Xie & Xing, Residual CNN for ICD  
[8] Yang et al., Hierarchical Attention Networks
