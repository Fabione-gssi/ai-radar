from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Mapping


def _clamp(v: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, v)))


def _norm(x: str) -> str:
    s = (x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_source_trust(
    author_org: str,
    trust_cfg: Dict[str, Any],
    source_kind: str = "institutional",
    creator_name: str = "",
) -> float:
    """
    Trust robusto (precision-first):
    - usa alias org/creator
    - supporta whitelist editoriale creator (strict_creator_whitelist)
    """
    default_trust = float(trust_cfg.get("default_trust", 0.55))
    creator_default = float(trust_cfg.get("creator_default_trust", 0.70))
    creator_block_trust = float(trust_cfg.get("creator_block_trust", 0.05))
    strict_creator = bool(trust_cfg.get("strict_creator_whitelist", False))

    org_trust = {_norm(k): float(v) for k, v in (trust_cfg.get("org_trust", {}) or {}).items()}
    creator_trust = {_norm(k): float(v) for k, v in (trust_cfg.get("creator_trust", {}) or {}).items()}

    org_alias = { _norm(k): _norm(v) for k, v in (trust_cfg.get("org_alias", {}) or trust_cfg.get("org_aliases", {}) or {}).items() }
    creator_alias = { _norm(k): _norm(v) for k, v in (trust_cfg.get("creator_alias", {}) or trust_cfg.get("creator_aliases", {}) or {}).items() }

    kind = (source_kind or "institutional").lower().strip()

    if kind == "creator":
        c = _norm(creator_name or author_org)
        c = creator_alias.get(c, c)
        if strict_creator and c and c not in creator_trust:
            return _clamp(creator_block_trust)
        if c in creator_trust:
            return _clamp(creator_trust[c])
        return _clamp(creator_default)

    k = _norm(author_org)
    k = org_alias.get(k, k)
    if k in org_trust:
        return _clamp(org_trust[k])

    if k in {"arxiv", "github", "huggingface"}:
        return 0.75

    return _clamp(default_trust)


def compute_source_mix_score(
    source_counts: Mapping[str, int],
    source: str,
    source_kind: str = "institutional",
    source_weight: float = 1.0,
    ranking_cfg: Dict[str, Any] | None = None,
    trust: float = 0.6,
) -> float:
    """
    Mix/diversity score (leggero): penalizza flooding da una singola fonte e, se richiesto,
    limita il rumore creator oltre il target ratio.

    Nota: non deve "spingere" fonti deboli (precision-first).
    """
    total = max(1, int(sum(source_counts.values())))
    src = (source or "").lower().strip()
    count = int(source_counts.get(src, 1))
    share = count / total

    kind = (source_kind or "institutional").lower().strip()
    w = _clamp(float(source_weight), 0.0, 1.0)

    # base: trust e peso contano più del kind
    base = 0.55 + 0.25 * _clamp(trust) + 0.10 * w
    if kind == "creator":
        base -= 0.05  # creator solo se whitelist+trust, quindi non serve bonus

    # penalizza over-representation (es. share > 20%)
    over = max(0.0, share - 0.20)
    base -= min(0.18, over * 0.75)

    # penalizza creator oltre target (se configurato)
    if ranking_cfg:
        target = float(ranking_cfg.get("creator_target_ratio", 0.40))
        # per stimare ratio, ci aspettiamo che 'source_counts' sia per source, quindi qui non lo calcoliamo.
        # La penalizzazione creator deve essere gestita a livello pipeline (ratio globale).
        _ = target

    return _clamp(base)


def compute_recency_score(published_at: datetime | None, half_life_days: float = 7.0) -> float:
    if not published_at:
        return 0.4

    now = datetime.now(timezone.utc)
    delta_days = (now - published_at.astimezone(timezone.utc)).total_seconds() / 86400.0
    if delta_days < 0:
        delta_days = 0

    return float(math.exp(-math.log(2) * (delta_days / max(half_life_days, 0.1))))


def compute_actionability(text: str, url: str) -> float:
    t = (text or "").lower()
    u = (url or "").lower()

    score = 0.0
    if "github.com" in u:
        score += 0.40
    if "huggingface.co" in u:
        score += 0.35
    if "arxiv.org" in u:
        score += 0.20

    keywords = [
        "code", "repo", "github", "pip install", "docker", "demo", "benchmark",
        "weights", "checkpoint", "api", "sdk", "release", "model card",
        "tutorial", "example", "notebook", "colab", "cli",
    ]
    hits = sum(1 for kw in keywords if kw in t)
    score += min(0.40, hits * 0.07)

    return _clamp(score)


def compute_breakout_signal(
    text: str,
    source: str,
    novelty: float,
    recency: float,
    actionability: float,
) -> float:
    """
    Segnale early-scout:
    più alto per novelty+recency+actionability, con leggera prior su discovery source.
    """
    t = (text or "").lower()
    s = (source or "").lower()

    score = 0.0
    if s in {"github_discovery", "huggingface_discovery"}:
        score += 0.30

    product_kws = [
        "assistant", "copilot", "workspace", "notebook", "voice", "speech",
        "transcription", "browser", "agentic", "automation",
    ]
    hits = sum(1 for kw in product_kws if kw in t)
    score += min(0.25, 0.04 * hits)

    score += 0.25 * _clamp(novelty)
    score += 0.15 * _clamp(recency)
    score += 0.20 * _clamp(actionability)

    return _clamp(score)


def combine_priority(scores: Dict[str, float], ranking_cfg: Dict[str, Any]) -> float:
    """
    Combinazione pesata con normalizzazione robusta.
    """
    w = ranking_cfg.get("weights", {}) or {}
    if not w:
        w = {
            "source_trust": 0.28,
            "novelty": 0.24,
            "relevance": 0.20,
            "actionability": 0.14,
            "recency": 0.10,
            "source_mix": 0.04,
        }

    tot = 0.0
    agg = 0.0
    for k, wk in w.items():
        wk = float(wk)
        if wk <= 0:
            continue
        tot += wk
        agg += wk * float(scores.get(k, 0.0))

    if tot <= 0:
        return 0.0
    return float(agg / tot)
