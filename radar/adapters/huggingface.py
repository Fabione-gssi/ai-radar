from __future__ import annotations

from typing import Dict, List, Tuple
import os
import requests

from radar.utils import safe_text, canonicalize_url, new_uuid, utc_now, parse_datetime_maybe


def _headers() -> Dict[str, str]:
    token = os.getenv("HF_TOKEN")
    headers = {"User-Agent": "ai-news-radar/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_recent_models(max_models: int = 50, timeout: int = 25) -> List[Dict]:
    url = "https://huggingface.co/api/models"
    params = {"sort": "lastModified", "direction": -1, "limit": int(max_models)}

    r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    models = r.json()

    out: List[Dict] = []
    for m in models:
        model_id = m.get("modelId") or m.get("id") or ""
        if not model_id:
            continue

        card = m.get("cardData") or {}
        pipeline_tag = m.get("pipeline_tag") or ""
        tags = m.get("tags") or []
        likes = int(m.get("likes", 0) or 0)
        downloads = int(m.get("downloads", 0) or 0)
        last_modified = m.get("lastModified")

        title = f"HF model: {model_id}"
        desc = card.get("description") or ""
        content = safe_text(
            f"{title}. pipeline={pipeline_tag}. tags={tags}. likes={likes}. downloads={downloads}. {desc}"
        )

        author = model_id.split("/")[0] if "/" in model_id else "huggingface"

        out.append({
            "id": new_uuid(),
            "source": "huggingface_model",
            "source_type": "api",
            "author_org": author,
            "url": canonicalize_url(f"https://huggingface.co/{model_id}"),
            "title": safe_text(title),
            "published_at": parse_datetime_maybe(last_modified),
            "fetched_at": utc_now(),
            "content_text": content,
            "source_kind": "institutional",
            "source_weight": 1.0,
            "content_type_hint": "release",
            "hf_likes": likes,
            "hf_downloads": downloads,
            "hf_pipeline_tag": pipeline_tag,
        })

    return out


def keyword_filter(items: List[Dict], include: List[str], exclude: List[str]) -> List[Dict]:
    inc = [s.lower() for s in (include or [])]
    exc = [s.lower() for s in (exclude or [])]

    out = []
    for it in items:
        text = (it.get("content_text") or "").lower()
        if inc and not any(k in text for k in inc):
            continue
        if exc and any(k in text for k in exc):
            continue
        out.append(it)
    return out


def split_known_vs_emerging(items: List[Dict], known_authors: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Divide in:
    - known: autori/org whitelist (reliable)
    - emerging: resto (scout)
    """
    known = {(a or "").strip().lower() for a in (known_authors or []) if (a or "").strip()}

    base: List[Dict] = []
    emerg: List[Dict] = []

    for it in items:
        author = (it.get("author_org") or "").lower().strip()
        if author in known:
            base.append(it)
        else:
            x = dict(it)
            x["source"] = "huggingface_discovery"
            x["source_weight"] = 0.72
            x["content_type_hint"] = "release"
            emerg.append(x)

    return base, emerg
