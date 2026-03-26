"""
ATLAS-RL: Training Loop
Loss = CrossEntropy + λ·(1 − ATLAS)
RL reward = Accuracy + α·ATLAS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional, Tuple
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helper
# ══════════════════════════════════════════════════════════════════════════════
class QADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        queries:   List[str],
        answers:   List[str],
        contexts:  List[List[str]],
        tokenizer: AutoTokenizer,
        max_src:   int = 512,
        max_tgt:   int = 64,
    ):
        self.queries   = queries
        self.answers   = answers
        self.contexts  = contexts
        self.tokenizer = tokenizer
        self.max_src   = max_src
        self.max_tgt   = max_tgt

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        ctx = " ".join(self.contexts[idx])
        prompt = (
            f"Answer based on context.\n"
            f"Context: {ctx}\n"
            f"Question: {self.queries[idx]}\nAnswer:"
        )
        src = self.tokenizer(
            prompt, max_length=self.max_src,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        tgt = self.tokenizer(
            self.answers[idx], max_length=self.max_tgt,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = tgt["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids":      src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ATLAS-RL loss
# ══════════════════════════════════════════════════════════════════════════════
def atlas_rl_loss(
    ce_loss:     torch.Tensor,   # standard cross-entropy
    atlas_score: float,          # ATLAS score for this batch
    lam:         float = 0.3,    # weight of reliability penalty
) -> torch.Tensor:
    """L = CE + λ·(1 − ATLAS)"""
    reliability_penalty = lam * (1.0 - atlas_score)
    return ce_loss + reliability_penalty


def rl_reward(
    accuracy:    float,
    atlas_score: float,
    alpha:       float = 0.5,
) -> float:
    """Reward = Accuracy + α·ATLAS"""
    return accuracy + alpha * atlas_score


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════
class ATLASTrainer:
    """
    Fine-tunes flan-t5-base with the ATLAS-RL objective.
    Lightweight; designed for Kaggle P100/T4 (small batches).
    """

    def __init__(
        self,
        model_name:   str   = "google/flan-t5-base",
        lr:           float = 3e-5,
        lam:          float = 0.3,
        rl_alpha:     float = 0.5,
        save_dir:     str   = "atlas_rl_checkpoint",
    ):
        self.lam      = lam
        self.rl_alpha = rl_alpha
        self.save_dir = save_dir

        print("[Trainer] Loading model …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.history: List[Dict] = []

    # ── single training step ──────────────────────────────────────────────
    def train_step(
        self,
        batch:       Dict[str, torch.Tensor],
        atlas_score: float = 0.5,
    ) -> Dict[str, float]:
        self.model.train()

        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        ce_loss = outputs.loss

        loss = atlas_rl_loss(ce_loss, atlas_score, self.lam)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "ce_loss":    ce_loss.item(),
            "total_loss": loss.item(),
        }

    # ── full training loop ────────────────────────────────────────────────
    def train(
        self,
        dataset:      QADataset,
        epochs:       int = 3,
        batch_size:   int = 4,
        atlas_scores: Optional[List[float]] = None,  # pre-computed per-sample ATLAS
        eval_every:   int = 50,
    ) -> List[Dict]:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(loader)
        )

        step = 0
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for batch in loader:
                # use mean ATLAS if available, else 0.5
                a_score = float(np.mean(atlas_scores)) if atlas_scores else 0.5
                info = self.train_step(batch, a_score)
                epoch_losses.append(info["total_loss"])
                scheduler.step()
                step += 1

                if step % eval_every == 0:
                    mean_loss = np.mean(epoch_losses[-eval_every:])
                    print(f"  [Train] epoch={epoch} step={step}  loss={mean_loss:.4f}")

            epoch_info = {
                "epoch":      epoch,
                "mean_loss":  float(np.mean(epoch_losses)),
            }
            self.history.append(epoch_info)
            print(f"[Train] Epoch {epoch}/{epochs}  mean_loss={epoch_info['mean_loss']:.4f}")

        return self.history

    # ── inference (for eval after training) ──────────────────────────────
    def predict(self, queries: List[str], contexts: List[List[str]],
                batch_size: int = 4) -> List[str]:
        self.model.eval()
        predictions = []
        for i in range(0, len(queries), batch_size):
            bq  = queries[i: i + batch_size]
            bc  = contexts[i: i + batch_size]
            prompts = [
                f"Answer based on context.\nContext: {' '.join(c)}\nQuestion: {q}\nAnswer:"
                for q, c in zip(bq, bc)
            ]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                max_length=512, truncation=True
            ).to(DEVICE)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=64, num_beams=1, do_sample=False
                )
            for seq in out:
                predictions.append(
                    self.tokenizer.decode(seq, skip_special_tokens=True)
                )
        return predictions

    # ── save / load ───────────────────────────────────────────────────────
    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        print(f"[Trainer] Model saved to {self.save_dir}")

    def load(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.save_dir).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        print(f"[Trainer] Model loaded from {self.save_dir}")
