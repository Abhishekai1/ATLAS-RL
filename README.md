# ATLAS-RL: Reliability-Aware Learning for Robust Language Models

---

## **Abstract**

We introduce **ATLAS-RL**, a reliability-aware framework for training and evaluating large language models (LLMs) under perturbations. Unlike conventional approaches that optimize primarily for accuracy, ATLAS-RL explicitly models **prediction stability, grounding, and consistency**. The framework integrates perturbation-based analysis, NLI-driven diagnostics, and a composite reliability objective to produce models that remain robust under distribution shifts, noisy retrieval, and hallucination-prone conditions.

---

## **1. Overview**

**ATLAS-RL** is a reliability-aware training and evaluation framework designed to enhance the robustness of **Large Language Models (LLMs)** under challenging and realistic conditions.

Specifically, the framework targets:

- **Distribution Shifts** — stable performance under unseen inputs  
- **Noisy / Corrupted Retrieval** — robustness to imperfect context  
- **Multimodal Inconsistencies** — handling conflicting signals  
- **Hallucination-Prone Settings** — reducing unsupported outputs  

Unlike standard pipelines, ATLAS-RL treats **reliability as a first-class objective**, ensuring models remain **stable, grounded, and consistent under perturbations**.

---

## **2. System Architecture**

```text
Input Query q
      │
      ▼
┌──────────────────────────┐
│   Retrieval Module       │
│  (MiniLM + FAISS Index)  │
└────────────┬─────────────┘
             │ Top-K Docs D
             ▼
┌──────────────────────────┐
│   Perturbation Module    │
│  q' = Paraphrase(q)      │
│  D' = Corrupt(D)         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Generation Module      │
│  a  = LM(q, D)           │
│  a' = LM(q', D')         │
│  (token log-probs)       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Diagnostics Module     │
│  Consistency (NLI)       │
│  Grounding (NLI)         │
│  Hallucination (NLI)     │
│  Uncertainty (entropy)   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   ATLAS Scoring          │
│  Reliability + Metrics   │
└──────────────────────────┘
```

---

## **3. Algorithm**

### **Algorithm 1: ATLAS-RL Evaluation Pipeline**

```python
Input: Query q, corpus C
Output: ATLAS score S

1: D  ← Retrieve(q, C)
2: a, logp      ← Generate(q, D)

3: q' ← Paraphrase(q)
4: D' ← Corrupt(D)
5: a', logp'    ← Generate(q', D')

6: C_score ← NLI(a ↔ a')
7: G_score ← NLI(D ⇒ a)
8: H_score ← NLI(D ⟂ a)
9: U_score ← Uncertainty(logp)

10: R ← Reliability(logp, logp')

11: S ← αR + βG_score + γC_score − δU_score

return S
```

---

## **4. Reliability Formulation**

```math
R = 1 − σ( KL(p || p̃) )
```

Where:
- p = original token distribution  
- p̃ = perturbed token distribution  

---

## **5. ATLAS Score**

```math
ATLAS = αR + βG + γC − δU
```

| Component | Description |
|----------|------------|
| R | Stability under perturbation |
| G | Grounding (NLI entailment) |
| C | Consistency (bidirectional NLI) |
| U | Uncertainty (entropy proxy) |

---

## **6. Training Objective**

```python
for batch in dataset:
    outputs = model(batch)
    ce_loss = CrossEntropy(outputs)

    atlas_score = estimate_atlas(batch)

    loss = ce_loss * (1 - atlas_score)

    loss.backward()
    optimizer.step()
```

---

## **7. Core Modules**

### Retrieval
```python
emb = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatIP(384)
```

### Perturbation
```python
q' = paraphrase(q)
D' = corrupt(D)
```

### Generation
```python
a, log_probs = model.generate(q, D)
```

### Diagnostics
```python
consistency   = NLI(a, a')
grounding     = NLI(D, a)
hallucination = NLI_contradiction(D, a)
uncertainty   = -mean(log_probs)
```

### Reliability
```python
R = 1 - sigmoid(KL(logp, logp'))
```

---

## **8. Main Results**

```text
==================================================================================
  MAIN RESULTS
==================================================================================
Model                 Accuracy(F1)    Hallucination    Consistency    Grounding    ATLAS Score
----------------------------------------------------------------------------------
Baseline LLM          0.1954          0.3800           0.2891         0.1537       0.3695
RAG                   0.1182          0.2458           0.1755         0.3816       0.4095
ATLAS-RL              0.2192          0.1451           0.1977         0.3506       0.4057
==================================================================================
```

---

## **9. Key Insight**

```text
Standard LLMs optimize for correctness.
RAG optimizes for retrieval alignment.
ATLAS-RL optimizes for reliability under perturbation.
```

---

## **10. Failure Mode Detection**

```python
if hallucination > 0.5:
    failure = "hallucination"
elif grounding < 0.3:
    failure = "retrieval_error"
elif consistency < 0.4:
    failure = "inconsistency"
```

---

## **11. Installation & Execution**

```bash
pip install faiss-cpu sentence-transformers datasets transformers torch
python ATLAS-RL.py
```

---

## **12. Outputs**

```text
main_results.csv      # evaluation results
ablation_results.csv  # ablation study
```

---

## **13. Contributions**

```text
• Reliability as a first-class optimization objective
• Perturbation-based stability modeling
• NLI-driven diagnostic decomposition
• Unified training + evaluation framework
```

---

## **14. Citation**

```bibtex
@article{atlas_rl_2026,
  title   = {ATLAS-RL: Reliability-Aware Learning for Robust Language Models},
  author  = {Abhishek Yadav},
  year    = {2026},
  note    = {Under submission to EMNLP}
}
```
