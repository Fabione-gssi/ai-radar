from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
    """
    Carica il modello embedding multilingua IT+EN.
    """
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embeddings normalizzati [N, D], float32.
    """
    model = get_embedder()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def max_similarity_with_index(new_emb: np.ndarray, recent_embs: np.ndarray | None) -> Tuple[float, int]:
    """
    Ritorna (max_sim, argmax_idx) verso i recenti.
    Se recent_embs è vuoto => (0.0, -1)
    """
    if recent_embs is None or len(recent_embs) == 0:
        return 0.0, -1
    sims = recent_embs @ new_emb
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


def novelty_score(new_emb: np.ndarray, recent_embs: np.ndarray | None) -> float:
    """
    Novelty in [0..1] = 1 - max_similarity
    """
    if recent_embs is None or len(recent_embs) == 0:
        return 0.90
    max_sim, _ = max_similarity_with_index(new_emb, recent_embs)
    max_sim = max(-1.0, min(1.0, max_sim))
    return float(max(0.0, min(1.0, 1.0 - max_sim)))
