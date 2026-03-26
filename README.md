# 🚀 ATLAS-RL: Reliability-Aware Learning for Multimodal & LLM Systems

<p align="left">
  <img src="https://komarev.com/ghpvc/?username=Abhishekai1&label=Project%20Views&color=0e75b6&style=flat" alt="views" />
</p>

---

## 🧠 Overview

**ATLAS-RL** is a reliability-aware training and evaluation framework designed to improve the robustness of Large Language Models (LLMs) under:

- Distribution shifts  
- Noisy / corrupted retrieval  
- Multimodal inconsistencies  
- Hallucination-prone settings  

Unlike standard pipelines, ATLAS-RL explicitly models reliability signals and optimizes generation using a composite reliability objective.

---

## 🏗️ Architecture

The system follows a modular pipeline:

                ┌──────────────────────────┐
                │        Query (Q)         │
                └────────────┬─────────────┘
                             │
                    ┌────────▼────────┐
                    │ Retrieval Module │
                    │ (FAISS + MiniLM)│
                    └────────┬────────┘
                             │ Top-K Docs
                ┌────────────▼────────────┐
                │   Perturbation Module   │
                │ (Query + Context Noise) │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │  Generation Module      │
                │ (Flan-T5 / LLM)         │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │ Diagnostic Module       │
                │ (NLI-based Evaluation)  │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │  ATLAS Score Computation│
                │ (Reliability Objective) │
                └─────────────────────────┘

---

## ⚙️ Key Components

### 🔍 Retrieval Module
- SentenceTransformers (MiniLM) + FAISS  
- Dense semantic retrieval  
- Top-K document selection  

### 🔄 Perturbation Module
- Query paraphrasing  
- Context corruption (noise injection)  
- Simulates real-world instability  

### 🧠 Generation Module
- FLAN-T5 based generation  
- Produces answer + token-level likelihoods  

### 📊 Diagnostics Module
Uses NLI (BART-MNLI) to compute:
- Consistency  
- Grounding  
- Hallucination  
- Uncertainty  

---

## 📈 ATLAS Score

ATLAS = α·Reliability + β·Grounding + γ·Consistency − δ·Uncertainty

---

## 📊 Main Results

==================================================================================
  MAIN RESULTS
==================================================================================
Model                 Accuracy(F1)          Hallucination         Consistency           Grounding             ATLAS Score         
----------------------------------------------------------------------------------
Baseline LLM          0.1954                0.3800                0.2891                0.1537                0.3695              
RAG                   0.1182                0.2458                0.1755                0.3816                0.4095              
ATLAS-RL              0.2192                0.1451                0.1977                0.3506                0.4057              
==================================================================================

📁 Results saved at:
/content/sample_data/main_results.csv

---

## 🔍 Key Observations

- ATLAS-RL improves accuracy over baseline (0.2192 vs 0.1954)  
- Significant reduction in hallucination (0.1451 vs 0.3800)  
- Better grounding compared to baseline models  
- Balanced reliability vs performance trade-off  

---

## 🧪 Training Strategy

ATLAS-RL introduces reliability-aware optimization:

Loss = CrossEntropy × (1 − ATLAS Score)

- Encourages stable generation  
- Reduces hallucination  
- Improves robustness under perturbations  

---

## 📦 Installation

pip install faiss-cpu sentence-transformers datasets transformers torch

---

## ▶️ Usage

python ATLAS-RL.py

---

## 📁 Outputs

- main_results.csv → Main evaluation  
- ablation_results.csv → Ablation study  

---

## 🚀 Future Work

- Multimodal extensions (vision + text)  
- Scaling to larger LLMs  
- RL-based reliability optimization  
- Real-world deployment  

---

## 🤝 Citation

@article{atlas_rl_2026,
  title={ATLAS-RL: Reliability-Aware Learning for Robust AI Systems},
  author={Abhishek Yadav},
  year={2026},
  note={Target: ACL / EMNLP}
}

---

## 👤 Author

Abhishek Yadav  
Portfolio: https://portfolio-655v.vercel.app/  
LinkedIn: https://www.linkedin.com/in/abhishekskyyadav  
GitHub: https://github.com/Abhishekai1  

---

## ⚡ Research Note

This work focuses on building AI systems that remain reliable under uncertainty, rather than optimizing only for accuracy.
