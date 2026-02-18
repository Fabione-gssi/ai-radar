from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _prep(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _count_hits(text: str, keywords: List[str]) -> int:
    t = _prep(text)
    hits = 0
    for kw in (keywords or []):
        k = (kw or "").strip().lower()
        if not k:
            continue
        if len(k) <= 3:
            if re.search(r"\b" + re.escape(k) + r"\b", t):
                hits += 1
        elif k in t:
            hits += 1
    return hits


def suggest_tags(text: str, taxonomy: Dict[str, Any], max_tags: int = 5) -> List[Dict[str, Any]]:
    """
    Suggerisce tag deterministici (no LLM), con motivazione.

    taxonomy.tag_rules:
      - tag: "agents"
        keywords: ["agent", "tool calling", ...]
        min_hits: 1
    """
    rules = (taxonomy or {}).get("tag_rules", []) or []
    scored: List[Tuple[str, int, List[str]]] = []

    for r in rules:
        if not isinstance(r, dict):
            continue
        tag = (r.get("tag") or "").strip().lower()
        if not tag:
            continue
        kws = list(r.get("keywords", []) or [])
        min_hits = int(r.get("min_hits", 1))
        hits = _count_hits(text, kws)
        if hits >= min_hits:
            # salva qualche keyword “spiegazione”
            t = _prep(text)
            matched = [kw for kw in kws if (kw or "").lower() in t][:4]
            scored.append((tag, hits, matched))

    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    for tag, hits, matched in scored[:max_tags]:
        reason = f"match({hits}): " + ", ".join(matched) if matched else f"match({hits})"
        out.append({"tag": tag, "hits": hits, "reason": reason})
    return out
