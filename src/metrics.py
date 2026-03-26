"""
ATLAS-RL: Information-Theoretic Reliability Metrics
R  = 1 - KL( P(y|x) || P(y|x') )
ATLAS = α·R + β·G + γ·C − δ·U
"""

import torch
import numpy as np
from typing import Dict, Optional, List


# ══════════════════════════════════════════════════════════════════════════════
# KL-based Reliability (R)
# ══════════════════════════════════════════════════════════════════════════════
def kl_divergence_from_log_probs(
    log_p: torch.Tensor,   # log P(y|x)   shape [T]
    log_q: torch.Tensor,   # log P(y|x')  shape [T]
) -> float:
    """
    Approximates KL(P || Q) using per-token log-probabilities.

    KL(P||Q) = Σ p_i * (log p_i - log q_i)
             ≈ mean(log_p - log_q)   under the empirical distribution

    Returns scalar ≥ 0.
    """
    min_len = min(len(log_p), len(log_q))
    if min_len == 0:
        return 0.0

    lp = log_p[:min_len].float()
    lq = log_q[:min_len].float()

    # Clamp log probs to avoid -inf
    lp = torch.clamp(lp, min=-20.0)
    lq = torch.clamp(lq, min=-20.0)

    # Convert to probabilities (unnormalised – this is an approximation)
    p = torch.exp(lp)
    kl = (p * (lp - lq)).mean().item()
    return float(max(kl, 0.0))   # KL ≥ 0


def reliability_score(
    token_log_probs: torch.Tensor,
    perturbed_token_log_probs: torch.Tensor,
    kl_scale: float = 5.0,
) -> float:
    """
    R = 1 − sigmoid(KL / kl_scale)
    Range (0, 1); R ≈ 1 means output is stable under perturbation.
    """
    kl = kl_divergence_from_log_probs(token_log_probs, perturbed_token_log_probs)
    # normalise KL to [0,1] via sigmoid
    r = 1.0 - (1.0 / (1.0 + np.exp(-kl / kl_scale + 1)))
    return float(np.clip(r, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# ATLAS Score
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_WEIGHTS = dict(alpha=0.35, beta=0.30, gamma=0.25, delta=0.10)


def atlas_score(
    R: float,  # reliability (KL-based)
    G: float,  # grounding
    C: float,  # consistency
    U: float,  # uncertainty (to be penalised)
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    ATLAS = α·R + β·G + γ·C − δ·U

    All inputs in [0, 1]. Result clipped to [0, 1].
    """
    w = weights or DEFAULT_WEIGHTS
    score = w["alpha"] * R + w["beta"] * G + w["gamma"] * C - w["delta"] * U
    return float(np.clip(score, 0.0, 1.0))


def atlas_score_ablation(
    R: float, G: float, C: float, U: float,
    ablate_kl: bool = False,
    ablate_consistency: bool = False,
    ablate_grounding: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """ATLAS score with ablated components (set to 0.5 neutral)."""
    r_use = 0.5 if ablate_kl          else R
    g_use = 0.5 if ablate_grounding   else G
    c_use = 0.5 if ablate_consistency else C
    return atlas_score(r_use, g_use, c_use, U, weights)


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy helpers
# ══════════════════════════════════════════════════════════════════════════════
def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip().lower() == reference.strip().lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_toks = set(prediction.lower().split())
    ref_toks  = set(reference.lower().split())
    if not ref_toks:
        return 0.0
    common = pred_toks & ref_toks
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks) if pred_toks else 0.0
    rec  = len(common) / len(ref_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def compute_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
    em  = np.mean([exact_match(p, r) for p, r in zip(predictions, references)])
    f1  = np.mean([token_f1(p, r)   for p, r in zip(predictions, references)])
    return {"exact_match": float(em), "token_f1": float(f1)}


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate metrics over a dataset
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_metrics(
    predictions:      List[str],
    references:       List[str],
    diagnostics_list: List[Dict[str, float]],
    atlas_scores:     List[float],
) -> Dict[str, float]:

    acc   = compute_accuracy(predictions, references)
    hall  = np.mean([d.get("hallucination", 0.0) for d in diagnostics_list])
    cons  = np.mean([d.get("consistency",   0.5) for d in diagnostics_list])
    grnd  = np.mean([d.get("grounding",     0.5) for d in diagnostics_list])
    unc   = np.mean([d.get("uncertainty",   0.5) for d in diagnostics_list])
    atlas = np.mean(atlas_scores)

    return {
        "exact_match":       acc["exact_match"],
        "token_f1":          acc["token_f1"],
        "hallucination_rate": float(hall),
        "consistency":       float(cons),
        "grounding":         float(grnd),
        "uncertainty":       float(unc),
        "atlas_score":       float(atlas),
    }
