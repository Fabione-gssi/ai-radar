from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from radar.utils import normalize_whitespace

CONTENT_TYPES = ("tool", "research", "release", "industry", "news")


def _prepare(text: str) -> str:
    return normalize_whitespace((text or "").lower())


def _count_hits(text: str, keywords: List[str]) -> int:
    t = _prepare(text)
    hits = 0
    for kw in (keywords or []):
        k = (kw or "").strip().lower()
        if not k:
            continue
        # keyword corte: match su word-boundary per evitare falsi positivi
        if len(k) <= 3:
            if re.search(r"\b" + re.escape(k) + r"\b", t):
                hits += 1
        elif k in t:
            hits += 1
    return hits


def _content_type_keywords(taxonomy: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Legge taxonomy.content_types se presente; fallback su set interno.
    """
    raw = (taxonomy or {}).get("content_types", {})
    if isinstance(raw, dict) and raw:
        out: Dict[str, List[str]] = {}
        for k, v in raw.items():
            kk = (k or "").strip().lower()
            if kk not in CONTENT_TYPES:
                continue
            if isinstance(v, dict):
                out[kk] = list(v.get("keywords", []) or [])
            elif isinstance(v, list):
                out[kk] = list(v)
        if out:
            # assicura tutte le classi
            for ct in CONTENT_TYPES:
                out.setdefault(ct, [])
            return out

    return {
        "tool": [
            "framework", "sdk", "library", "package", "tool", "plugin", "api", "cli",
            "demo app", "assistant", "copilot", "workspace", "notebook",
            "ollama", "vllm", "langchain", "llama_index", "transformers",
            "strumento", "libreria", "pacchetto", "applicazione",
        ],
        "research": [
            "paper", "arxiv", "preprint", "state of the art", "sota", "benchmark",
            "dataset", "method", "evaluation", "ablation",
            "ricerca", "studio", "metodo", "valutazione",
        ],
        "release": [
            "release", "released", "launch", "launched", "announced", "announcement",
            "new model", "checkpoint", "weights", "model card", "version", "v1.", "v2.",
            "rilascio", "annuncio", "nuovo modello",
        ],
        "industry": [
            "case study", "customer story", "enterprise", "deployment", "pilot",
            "roi", "kpi", "cost saving", "time reduction", "operations",
            "manufacturing", "supply chain", "compliance", "contact center", "fraud",
            "caso studio", "caso d'uso", "adozione", "produzione", "industriale",
            "risparmio", "efficienza",
        ],
        "news": [
            "news", "update", "trend", "event", "interview",
            "notizia", "aggiornamento", "tendenza", "intervista",
        ],
    }


def classify_topic_and_relevance(text: str, taxonomy: Dict[str, Any]) -> Tuple[str, float]:
    """
    Topic fine-grained + relevance [0..1].
    """
    t = _prepare(text)
    topics = (taxonomy or {}).get("topics", []) or []

    best_topic = "Other"
    best_hits = 0
    for topic in topics:
        name = topic.get("name", "Other")
        kws = topic.get("keywords", []) or []
        hits = _count_hits(t, kws)
        if hits > best_hits:
            best_hits = hits
            best_topic = name

    relevance = 1.0 - (1.0 / (1.0 + best_hits))  # saturazione morbida
    return best_topic, float(relevance)


def classify_content_type(
    text: str,
    taxonomy: Dict[str, Any],
    source: str = "",
    source_type: str = "",
    url: str = "",
    content_type_hint: str = "",
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classifica in: tool / research / release / industry / news
    e restituisce confidence + scores grezzi.
    """
    t = _prepare(text)
    s = (source or "").lower()
    st = (source_type or "").lower()
    u = (url or "").lower()
    hint = (content_type_hint or "").lower().strip()

    kw_map = _content_type_keywords(taxonomy)
    scores: Dict[str, float] = {ct: float(_count_hits(t, kw_map.get(ct, []))) for ct in CONTENT_TYPES}

    # Priors da sorgente (punto 5: ridotto bias HF su release)
    if "arxiv" in s or "arxiv.org" in u:
        scores["research"] += 2.5
    if "github_release" in s or "/releases/" in u:
        scores["release"] += 1.8
        scores["tool"] += 1.2
    if "huggingface_model" in s or "huggingface.co/" in u:
        scores["release"] += 1.2  # era troppo alto
        scores["tool"] += 0.5
    if st == "rss":
        scores["news"] += 0.4

    # Segnali "prodotto/tool" per evitare tutto su release
    if any(k in t for k in ["assistant", "copilot", "workspace", "notebook", "plugin", "desktop app", "mobile app"]):
        scores["tool"] += 1.4

    # Hint esplicito da feed
    if hint in CONTENT_TYPES:
        scores[hint] += 1.8

    best_type = max(scores, key=scores.get)
    total = sum(max(v, 0.0) for v in scores.values()) + 1e-8
    confidence = float(max(0.0, min(1.0, scores[best_type] / total)))

    if scores[best_type] <= 0:
        best_type = "news"
        confidence = 0.2

    return best_type, confidence, scores


def compute_quality_score(
    text: str,
    content_type_confidence: float,
    keyword_strength: float,
    lang: str = "unknown",
) -> Tuple[float, str]:
    """
    Quality score [0..1] per filtro pre-store (punto 6).
    """
    t = _prepare(text)
    n_chars = len(t)

    length_score = min(1.0, n_chars / 900.0)
    kw_score = min(1.0, keyword_strength / 4.0)
    conf_score = max(0.0, min(1.0, content_type_confidence))

    q = 0.35 * length_score + 0.40 * conf_score + 0.25 * kw_score

    flags = []
    if n_chars < 140:
        flags.append("too_short")
    if conf_score < 0.33:
        flags.append("weak_type_conf")
    if kw_score < 0.15:
        flags.append("weak_keyword_signal")
    if lang == "unknown":
        flags.append("lang_unknown")

    return float(max(0.0, min(1.0, q))), ",".join(flags)


def classify_item(
    text: str,
    taxonomy: Dict[str, Any],
    source: str = "",
    source_type: str = "",
    url: str = "",
    content_type_hint: str = "",
    lang: str = "unknown",
) -> Dict[str, Any]:
    """
    Wrapper unico per pipeline.
    """
    topic, relevance = classify_topic_and_relevance(text, taxonomy)
    content_type, type_conf, raw_scores = classify_content_type(
        text=text,
        taxonomy=taxonomy,
        source=source,
        source_type=source_type,
        url=url,
        content_type_hint=content_type_hint,
    )
    keyword_strength = max(raw_scores.values()) if raw_scores else 0.0
    quality, flags = compute_quality_score(
        text=text,
        content_type_confidence=type_conf,
        keyword_strength=keyword_strength,
        lang=lang,
    )

    # boost lieve rilevanza per classi ad alto valore
    if content_type in {"industry", "tool", "release", "research"}:
        relevance = min(1.0, relevance + 0.05)

    return {
        "topic": topic,
        "relevance_score": float(relevance),
        "content_type": content_type,
        "content_type_confidence": float(type_conf),
        "quality_score": float(quality),
        "quality_flags": flags,
        "type_scores": raw_scores,
    }
