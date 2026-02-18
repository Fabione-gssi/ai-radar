from __future__ import annotations

from typing import Any, Dict, List, Optional

import feedparser

from radar.utils import (
    safe_text,
    canonicalize_url,
    new_uuid,
    utc_now,
    parse_datetime_maybe,
)


def fetch_rss(feed_url: str, feed_name: str = "rss", meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Legge RSS/Atom (feedparser). Per ridurre rumore, per default usa summary/content del feed.

    Se vuoi tentare full-text scraping, imposta nella sorgente:
      fetch_fulltext: true
    La fetch+estrazione avviene in pipeline (per evitare fetch inutili su item poi scartati).
    """
    meta = meta or {}
    source_kind = (meta.get("source_kind") or "institutional").lower()
    creator_name = safe_text(meta.get("creator_name", ""))
    source_weight = float(meta.get("source_weight", 1.0))
    content_type_hint = safe_text(meta.get("content_type_hint", ""))

    # opzioni full-text (gestite in pipeline)
    fetch_fulltext = bool(meta.get("fetch_fulltext", False))
    fulltext_min_chars = int(meta.get("fulltext_min_chars", 400))
    fulltext_max_chars = int(meta.get("fulltext_max_chars", 6000))
    fulltext_timeout_s = float(meta.get("fulltext_timeout_s", 10.0))
    fulltext_sleep_s = float(meta.get("fulltext_sleep_s", 0.0))

    feed = feedparser.parse(feed_url)
    feed_title = safe_text(getattr(feed, "feed", {}).get("title", "") if getattr(feed, "feed", None) else "")

    out: List[Dict[str, Any]] = []

    for e in (feed.entries or []):
        url = canonicalize_url(e.get("link", ""))
        title = safe_text(e.get("title", ""))

        summary = safe_text(e.get("summary", "") or e.get("description", ""))
        content_blocks = e.get("content", []) or []
        full_content = " ".join(
            safe_text(c.get("value", "")) for c in content_blocks if isinstance(c, dict)
        ).strip()

        body = full_content if len(full_content) > len(summary) else summary
        entry_author = safe_text(e.get("author", "") or e.get("dc_creator", ""))

        # Per creator feed: autore coerente
        if source_kind == "creator" and creator_name:
            author_org = creator_name
        else:
            author_org = entry_author or feed_title or feed_name

        published_at = parse_datetime_maybe(
            e.get("published")
            or e.get("updated")
            or e.get("created")
        )

        content_text = safe_text(f"{title}. {body}")

        out.append(
            {
                "id": new_uuid(),
                "source": feed_name,
                "source_type": "rss",
                "author_org": author_org,
                "creator_name": creator_name,
                "source_kind": source_kind,
                "source_weight": source_weight,
                "url": url,
                "title": title,
                "published_at": published_at,
                "fetched_at": utc_now(),
                "content_text": content_text,
                "content_type_hint": content_type_hint,
                # pass-through fulltext policy to pipeline
                "fetch_fulltext": fetch_fulltext,
                "fulltext_min_chars": fulltext_min_chars,
                "fulltext_max_chars": fulltext_max_chars,
                "fulltext_timeout_s": fulltext_timeout_s,
                "fulltext_sleep_s": fulltext_sleep_s,
            }
        )

    return out
