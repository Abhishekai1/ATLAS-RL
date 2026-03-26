"""
ATLAS-RL: Retrieval and Generation Pipeline
"""

import torch
import numpy as np
import faiss
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    BlipProcessor, BlipForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import random
import re
from typing import List, Dict, Optional, Tuple


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ATLAS] Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RETRIEVAL MODULE
# ══════════════════════════════════════════════════════════════════════════════
class RetrievalModule:
    """FAISS-based dense retrieval over a document corpus."""

    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print("[Retrieval] Loading embedding model …")
        self.embedder = SentenceTransformer(embed_model_name, device=DEVICE)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[str] = []
        self.dim = 384  # MiniLM-L6 output dim

    # ── build / update index ──────────────────────────────────────────────
    def build_index(self, documents: List[str]) -> None:
        self.documents = documents
        embeddings = self.embedder.encode(
            documents, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True
        ).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        print(f"[Retrieval] Index built with {len(documents)} docs.")

    def add_documents(self, new_docs: List[str]) -> None:
        self.documents.extend(new_docs)
        embeddings = self.embedder.encode(
            new_docs, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True
        ).astype(np.float32)
        self.index.add(embeddings)

    # ── retrieve ──────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, idxs = self.index.search(q_emb, min(top_k, len(self.documents)))
        return [self.documents[i] for i in idxs[0] if i < len(self.documents)]

    def retrieve_batch(self, queries: List[str], top_k: int = 3) -> List[List[str]]:
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in queries]
        q_embs = self.embedder.encode(
            queries, batch_size=32, normalize_embeddings=True
        ).astype(np.float32)
        _, idxs_batch = self.index.search(q_embs, min(top_k, len(self.documents)))
        return [
            [self.documents[i] for i in row if i < len(self.documents)]
            for row in idxs_batch
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PERTURBATION / DISTRIBUTION-SHIFT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class PerturbationEngine:
    """Applies distributional shift to queries and contexts."""

    _PARAPHRASE_TEMPLATES = [
        "Rephrase the following question: {q}",
        "In other words: {q}",
        "Express differently: {q}",
        "Restate this: {q}",
    ]

    # simple lexical noise (no extra model needed)
    def paraphrase_query(self, query: str) -> str:
        words = query.split()
        if len(words) < 3:
            return query
        # swap two random adjacent words
        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
        noised = " ".join(words)
        # also lowercasing as lightweight shift
        return noised.lower()

    def corrupt_retrieval(
        self, original_docs: List[str], noise_docs: List[str], corrupt_ratio: float = 0.5
    ) -> List[str]:
        """Replace a fraction of retrieved docs with irrelevant noise docs."""
        n_corrupt = max(1, int(len(original_docs) * corrupt_ratio))
        corrupted = list(original_docs)
        for i in range(min(n_corrupt, len(corrupted))):
            if noise_docs:
                corrupted[i] = random.choice(noise_docs)
        return corrupted

    def add_image_noise(self, image_tensor: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
        """Gaussian noise on image tensor (optional)."""
        return (image_tensor + torch.randn_like(image_tensor) * noise_std).clamp(0, 1)

    def perturb_context(self, context: str) -> str:
        """Randomly drop sentences from context."""
        sentences = re.split(r'(?<=[.!?])\s+', context)
        if len(sentences) <= 1:
            return context
        keep = random.sample(sentences, max(1, len(sentences) - 1))
        return " ".join(keep)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GENERATION MODULE  (RAG)
# ══════════════════════════════════════════════════════════════════════════════
class GenerationModule:
    """Flan-T5-base: RAG-style generation (query + retrieved context → answer)."""

    MAX_INPUT_LEN = 512
    MAX_OUTPUT_LEN = 64

    def __init__(self, model_name: str = "google/flan-t5-base"):
        print("[Generation] Loading LLM …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    # ── single sample (with logits) ───────────────────────────────────────
    def generate_with_logits(
        self, query: str, context_docs: List[str]
    ) -> Tuple[str, torch.Tensor]:
        """Returns (answer_string, token_log_probs)."""
        context = " ".join(context_docs)
        prompt = f"Answer based on context.\nContext: {context}\nQuestion: {query}\nAnswer:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=self.MAX_INPUT_LEN, truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_OUTPUT_LEN,
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=1,          # greedy for speed
                do_sample=False,
            )

        answer_ids = outputs.sequences[0]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # stack scores → [T, vocab]
        if outputs.scores:
            scores = torch.stack(outputs.scores, dim=0)          # [T, vocab]
            log_probs = torch.log_softmax(scores, dim=-1)        # [T, vocab]
            # token log probs for the generated tokens
            gen_ids = answer_ids[1:]                             # skip pad/bos
            gen_ids = gen_ids[:scores.shape[0]]
            token_lp = log_probs[range(len(gen_ids)), gen_ids]   # [T]
        else:
            token_lp = torch.zeros(1, device=DEVICE)

        return answer, token_lp

    # ── batch generation (no logits, faster) ─────────────────────────────
    def generate_batch(
        self, queries: List[str], contexts: List[List[str]], batch_size: int = 4
    ) -> List[str]:
        answers = []
        for i in range(0, len(queries), batch_size):
            bq = queries[i: i + batch_size]
            bc = contexts[i: i + batch_size]
            prompts = [
                f"Answer based on context.\nContext: {' '.join(c)}\nQuestion: {q}\nAnswer:"
                for q, c in zip(bq, bc)
            ]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
                max_length=self.MAX_INPUT_LEN, truncation=True
            ).to(DEVICE)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.MAX_OUTPUT_LEN,
                    num_beams=1, do_sample=False
                )
            for seq in out:
                answers.append(self.tokenizer.decode(seq, skip_special_tokens=True))
        return answers

    # ── baseline (no context) ─────────────────────────────────────────────
    def generate_baseline(self, query: str) -> str:
        return self.generate_batch([query], [[]])[0]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VISION MODULE  (optional BLIP captions)
# ══════════════════════════════════════════════════════════════════════════════
class VisionModule:
    """BLIP image captioner for multimodal inputs."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        print("[Vision] Loading BLIP …")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def caption(self, pil_image) -> str:
        inputs = self.processor(pil_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=64)
        return self.processor.decode(ids[0], skip_special_tokens=True)

    def caption_batch(self, pil_images: list) -> List[str]:
        results = []
        for img in pil_images:
            results.append(self.caption(img))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class ATLASPipeline:
    """End-to-end ATLAS pipeline: retrieve → (optionally perturb) → generate."""

    def __init__(self, retriever: RetrievalModule, generator: GenerationModule,
                 perturber: Optional[PerturbationEngine] = None):
        self.retriever = retriever
        self.generator = generator
        self.perturber = perturber or PerturbationEngine()

    def run(self, query: str, top_k: int = 3,
            perturb: bool = False, noise_docs: Optional[List[str]] = None
            ) -> Dict:
        docs = self.retriever.retrieve(query, top_k=top_k)
        perturbed_query = query
        perturbed_docs = docs

        if perturb:
            perturbed_query = self.perturber.paraphrase_query(query)
            if noise_docs:
                perturbed_docs = self.perturber.corrupt_retrieval(docs, noise_docs)

        answer, token_lp = self.generator.generate_with_logits(query, docs)
        p_answer, p_token_lp = self.generator.generate_with_logits(perturbed_query, perturbed_docs)

        return {
            "query": query,
            "perturbed_query": perturbed_query,
            "docs": docs,
            "perturbed_docs": perturbed_docs,
            "answer": answer,
            "perturbed_answer": p_answer,
            "token_log_probs": token_lp,
            "perturbed_token_log_probs": p_token_lp,
        }
