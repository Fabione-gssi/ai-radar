from __future__ import annotations

from typing import Dict, List
import requests
import feedparser

from radar.utils import safe_text, canonicalize_url, new_uuid, utc_now, parse_datetime_maybe


def fetch_arxiv(query: str, max_results: int = 25, timeout: int = 20) -> List[Dict]:
    base = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    r = requests.get(base, params=params, timeout=timeout, headers={"User-Agent": "ai-news-radar/1.0"})
    r.raise_for_status()

    feed = feedparser.parse(r.text)
    out: List[Dict] = []

    for e in feed.entries:
        url = canonicalize_url(e.get("link", ""))
        title = safe_text(e.get("title", ""))
        summary = safe_text(e.get("summary", ""))

        published_at = parse_datetime_maybe(e.get("published"))
        out.append({
            "id": new_uuid(),
            "source": "arxiv",
            "source_type": "api",
            "author_org": "arxiv",
            "source_kind": "institutional",
            "source_weight": 1.0,
            "content_type_hint": "research",
            "url": url,
            "title": title,
            "published_at": published_at,
            "fetched_at": utc_now(),
            "content_text": f"{title}. {summary}",
        })

    return out
