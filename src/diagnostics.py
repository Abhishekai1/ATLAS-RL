"""
ATLAS-RL: Diagnostics Module
Consistency · Uncertainty · Grounding · Hallucination
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)
from typing import List, Dict, Optional, Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# NLI backbone (facebook/bart-large-mnli)
# ══════════════════════════════════════════════════════════════════════════════
class NLIModule:
    """BART-large-MNLI wrapper for entailment scoring."""

    # MNLI label order: contradiction(0), neutral(1), entailment(2)
    ENTAIL_IDX = 2
    CONTRA_IDX = 0

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        print("[NLI] Loading BART-large-MNLI …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        enc = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return {
            "entailment": float(probs[self.ENTAIL_IDX]),
            "neutral":    float(probs[1]),
            "contradiction": float(probs[self.CONTRA_IDX]),
        }

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        return self.score(premise, hypothesis)["entailment"]

    def batch_entailment(
        self, premises: List[str], hypotheses: List[str], batch_size: int = 8
    ) -> List[float]:
        scores = []
        for i in range(0, len(premises), batch_size):
            bp = premises[i: i + batch_size]
            bh = hypotheses[i: i + batch_size]
            enc = self.tokenizer(
                bp, bh, return_tensors="pt",
                padding=True, truncation=True, max_length=512
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            scores.extend(probs[:, self.ENTAIL_IDX].tolist())
        return scores


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
class DiagnosticsModule:
    """
    Computes per-sample reliability signals:
      consistency  – semantic similarity between original & perturbed answers
      uncertainty  – entropy proxy from token log-probs
      grounding    – NLI entailment of answer given retrieved context
      hallucination – contradiction score of answer vs context
    """

    def __init__(self, nli: NLIModule):
        self.nli = nli

    # ── 1. Consistency ────────────────────────────────────────────────────
    def consistency_score(self, answer: str, perturbed_answer: str) -> float:
        """
        Bidirectional NLI entailment average as semantic consistency proxy.
        Range [0, 1]; higher = more consistent.
        """
        if not answer.strip() or not perturbed_answer.strip():
            return 0.0
        e_fwd = self.nli.entailment_score(answer, perturbed_answer)
        e_bwd = self.nli.entailment_score(perturbed_answer, answer)
        return float((e_fwd + e_bwd) / 2.0)

    # ── 2. Uncertainty ────────────────────────────────────────────────────
    @staticmethod
    def uncertainty_score(token_log_probs: torch.Tensor) -> float:
        """
        Approximates uncertainty as the negative mean log-probability
        of generated tokens (higher = more uncertain).
        Normalised to [0, 1] via sigmoid.
        """
        if token_log_probs is None or token_log_probs.numel() == 0:
            return 0.5
        mean_nlp = -token_log_probs.float().mean().item()
        # sigmoid to squash into (0,1); uncertainty ≈ 1 when model is unsure
        uncertainty = 1.0 / (1.0 + np.exp(-mean_nlp + 3))  # shift so typical range is 0-1
        return float(np.clip(uncertainty, 0.0, 1.0))

    @staticmethod
    def entropy_from_logits(logits: torch.Tensor) -> float:
        """Token-level entropy averaged over sequence."""
        if logits is None or logits.numel() == 0:
            return 0.0
        probs = torch.softmax(logits.float(), dim=-1)
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
        return float(np.clip(ent / np.log(logits.shape[-1] + 1e-9), 0.0, 1.0))

    # ── 3. Grounding ─────────────────────────────────────────────────────
    def grounding_score(self, answer: str, context_docs: List[str]) -> float:
        """
        NLI entailment of answer given context.
        Checks each doc and returns max entailment.
        """
        if not context_docs or not answer.strip():
            return 0.0
        context = " ".join(context_docs)[:1024]
        return self.nli.entailment_score(context, answer)

    # ── 4. Hallucination ─────────────────────────────────────────────────
    def hallucination_score(self, answer: str, context_docs: List[str]) -> float:
        """
        Contradiction score of answer vs context.
        Higher = more likely hallucinated.
        """
        if not context_docs or not answer.strip():
            return 0.0
        context = " ".join(context_docs)[:1024]
        scores = self.nli.score(context, answer)
        return float(scores["contradiction"])

    def is_hallucination(self, answer: str, context_docs: List[str],
                         threshold: float = 0.5) -> bool:
        return self.hallucination_score(answer, context_docs) >= threshold

    # ── 5. Full diagnostics for one sample ───────────────────────────────
    def diagnose(
        self,
        answer: str,
        perturbed_answer: str,
        context_docs: List[str],
        token_log_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        C = self.consistency_score(answer, perturbed_answer)
        U = self.uncertainty_score(token_log_probs) if token_log_probs is not None else 0.5
        G = self.grounding_score(answer, context_docs)
        H = self.hallucination_score(answer, context_docs)
        return {
            "consistency":   C,
            "uncertainty":   U,
            "grounding":     G,
            "hallucination": H,
        }

    # ── 6. Failure mode classification ───────────────────────────────────
    @staticmethod
    def classify_failure(diagnostics: Dict[str, float]) -> str:
        H = diagnostics.get("hallucination", 0.0)
        C = diagnostics.get("consistency",   1.0)
        G = diagnostics.get("grounding",     1.0)

        if H > 0.5:
            return "hallucination"
        if G < 0.3:
            return "retrieval_error"
        if C < 0.4:
            return "inconsistency"
        return "none"

    # ── 7. Batch diagnostics ─────────────────────────────────────────────
    def diagnose_batch(
        self,
        answers: List[str],
        perturbed_answers: List[str],
        contexts: List[List[str]],
        token_log_probs_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[Dict[str, float]]:
        results = []
        n = len(answers)
        tlp = token_log_probs_list or [None] * n
        for i in range(n):
            d = self.diagnose(
                answers[i],
                perturbed_answers[i],
                contexts[i],
                tlp[i],
            )
            results.append(d)
        return results
