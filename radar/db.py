from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
import psycopg2.extras


class NewsDB:
    """
    Implementazione Postgres/Supabase
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.con = psycopg2.connect(self.database_url)
        self.con.autocommit = True

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    # -----------------------------
    # News items
    # -----------------------------
    def upsert_item(self, item: Dict[str, Any]) -> None:
        sql = """
        INSERT INTO news_items (
          id, source, source_type, author_org, source_kind, source_weight, creator_name,
          url, title, published_at, fetched_at, lang, content_text,
          topic, content_type, content_type_confidence, lane, breakout_signal,
          novelty_score, relevance_score, actionability_score, recency_score,
          source_trust_score, source_mix_score, quality_score, quality_flags,
          priority_score, status
        )
        VALUES (
          %(id)s, %(source)s, %(source_type)s, %(author_org)s, %(source_kind)s, %(source_weight)s, %(creator_name)s,
          %(url)s, %(title)s, %(published_at)s, %(fetched_at)s, %(lang)s, %(content_text)s,
          %(topic)s, %(content_type)s, %(content_type_confidence)s, %(lane)s, %(breakout_signal)s,
          %(novelty_score)s, %(relevance_score)s, %(actionability_score)s, %(recency_score)s,
          %(source_trust_score)s, %(source_mix_score)s, %(quality_score)s, %(quality_flags)s,
          %(priority_score)s, %(status)s
        )
        ON CONFLICT (url) DO UPDATE SET
          id = EXCLUDED.id,
          source = EXCLUDED.source,
          source_type = EXCLUDED.source_type,
          author_org = EXCLUDED.author_org,
          source_kind = EXCLUDED.source_kind,
          source_weight = EXCLUDED.source_weight,
          creator_name = EXCLUDED.creator_name,
          title = EXCLUDED.title,
          published_at = EXCLUDED.published_at,
          fetched_at = EXCLUDED.fetched_at,
          lang = EXCLUDED.lang,
          content_text = EXCLUDED.content_text,
          topic = EXCLUDED.topic,
          content_type = EXCLUDED.content_type,
          content_type_confidence = EXCLUDED.content_type_confidence,
          lane = EXCLUDED.lane,
          breakout_signal = EXCLUDED.breakout_signal,
          novelty_score = EXCLUDED.novelty_score,
          relevance_score = EXCLUDED.relevance_score,
          actionability_score = EXCLUDED.actionability_score,
          recency_score = EXCLUDED.recency_score,
          source_trust_score = EXCLUDED.source_trust_score,
          source_mix_score = EXCLUDED.source_mix_score,
          quality_score = EXCLUDED.quality_score,
          quality_flags = EXCLUDED.quality_flags,
          priority_score = EXCLUDED.priority_score,
          status = EXCLUDED.status
        ;
        """

        payload = {
            "id": item.get("id"),
            "source": item.get("source"),
            "source_type": item.get("source_type"),
            "author_org": item.get("author_org"),
            "source_kind": item.get("source_kind", "institutional"),
            "source_weight": float(item.get("source_weight", 1.0)),
            "creator_name": item.get("creator_name", ""),
            "url": item.get("url"),
            "title": item.get("title"),
            "published_at": item.get("published_at"),
            "fetched_at": item.get("fetched_at"),
            "lang": item.get("lang"),
            "content_text": item.get("content_text"),
            "topic": item.get("topic"),
            "content_type": item.get("content_type", "news"),
            "content_type_confidence": float(item.get("content_type_confidence", 0.0)),
            "lane": item.get("lane", "reliable"),
            "breakout_signal": float(item.get("breakout_signal", 0.0)),
            "novelty_score": float(item.get("novelty_score", 0.0)),
            "relevance_score": float(item.get("relevance_score", 0.0)),
            "actionability_score": float(item.get("actionability_score", 0.0)),
            "recency_score": float(item.get("recency_score", 0.0)),
            "source_trust_score": float(item.get("source_trust_score", 0.0)),
            "source_mix_score": float(item.get("source_mix_score", 0.0)),
            "quality_score": float(item.get("quality_score", 0.0)),
            "quality_flags": item.get("quality_flags", ""),
            "priority_score": float(item.get("priority_score", 0.0)),
            "status": item.get("status", "new"),
        }

        with self.con.cursor() as cur:
            cur.execute(sql, payload)

    def get_existing_urls(self, urls: List[str]) -> set[str]:
        clean = [u for u in {(u or "").strip() for u in (urls or [])} if u]
        if not clean:
            return set()
        with self.con.cursor() as cur:
            cur.execute("SELECT url FROM news_items WHERE url = ANY(%s)", (clean,))
            return {r[0] for r in cur.fetchall()}

    def get_recent_texts(self, limit: int = 200) -> List[Tuple[str, str]]:
        with self.con.cursor() as cur:
            cur.execute(
                """
                SELECT url, content_text
                FROM news_items
                ORDER BY published_at DESC NULLS LAST, fetched_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
        return [(r[0], r[1] or "") for r in rows]

    def update_status(self, url: str, status: str) -> None:
        with self.con.cursor() as cur:
            cur.execute("UPDATE news_items SET status = %s WHERE url = %s", (status, url))

    # -----------------------------
    # Tags
    # -----------------------------
    def list_tags(self) -> List[str]:
        with self.con.cursor() as cur:
            cur.execute("SELECT tag FROM tags ORDER BY tag")
            return [r[0] for r in cur.fetchall()]

    def tag_counts(self, limit: int = 50) -> List[tuple[str, int]]:
        with self.con.cursor() as cur:
            cur.execute(
                """
                SELECT tag, COUNT(*)::int as n
                FROM item_tags
                GROUP BY tag
                ORDER BY n DESC, tag ASC
                LIMIT %s
                """,
                (int(limit),),
            )
            return [(r[0], int(r[1])) for r in cur.fetchall()]

    def add_tags(self, tags: List[str]) -> None:
        clean = sorted({(t or "").strip().lower() for t in tags if (t or "").strip()})
        if not clean:
            return
        with self.con.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO tags(tag) VALUES %s ON CONFLICT (tag) DO NOTHING",
                [(t,) for t in clean],
            )

    def assign_tags_bulk(self, urls: List[str], tags: List[str]) -> int:
        clean_tags = sorted({(t or "").strip().lower() for t in tags if (t or "").strip()})
        clean_urls = sorted({(u or "").strip() for u in urls if (u or "").strip()})
        if not clean_tags or not clean_urls:
            return 0

        self.add_tags(clean_tags)

        pairs = [(u, t) for u in clean_urls for t in clean_tags]
        with self.con.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO item_tags(url, tag) VALUES %s
                ON CONFLICT (url, tag) DO NOTHING
                """,
                pairs,
            )

        with self.con.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)::int
                FROM item_tags
                WHERE url = ANY(%s) AND tag = ANY(%s)
                """,
                (clean_urls, clean_tags),
            )
            return int(cur.fetchone()[0])

    def remove_tags_bulk(self, urls: List[str], tags: List[str]) -> int:
        clean_tags = sorted({(t or "").strip().lower() for t in tags if (t or "").strip()})
        clean_urls = sorted({(u or "").strip() for u in urls if (u or "").strip()})
        if not clean_tags or not clean_urls:
            return 0
        with self.con.cursor() as cur:
            cur.execute(
                "DELETE FROM item_tags WHERE url = ANY(%s) AND tag = ANY(%s)",
                (clean_urls, clean_tags),
            )
            return cur.rowcount or 0

    def get_tags_map(self, urls: List[str]) -> Dict[str, List[str]]:
        clean_urls = sorted({(u or "").strip() for u in urls if (u or "").strip()})
        if not clean_urls:
            return {}
        out: Dict[str, List[str]] = {u: [] for u in clean_urls}
        with self.con.cursor() as cur:
            cur.execute(
                """
                SELECT url, tag
                FROM item_tags
                WHERE url = ANY(%s)
                ORDER BY tag
                """,
                (clean_urls,),
            )
            for u, t in cur.fetchall():
                out.setdefault(u, []).append(t)
        return out

    # -----------------------------
    # Saved views
    # -----------------------------
    def list_saved_views(self) -> List[str]:
        with self.con.cursor() as cur:
            cur.execute("SELECT name FROM saved_views ORDER BY name")
            return [r[0] for r in cur.fetchall()]

    def save_view(self, name: str, filters: Dict[str, Any]) -> None:
        n = (name or "").strip()
        if not n:
            return
        payload = json.dumps(filters, ensure_ascii=False)
        with self.con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO saved_views(name, filters_json, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (name) DO UPDATE SET
                  filters_json = EXCLUDED.filters_json,
                  updated_at = now()
                """,
                (n, payload),
            )

    def get_saved_view(self, name: str) -> Dict[str, Any]:
        with self.con.cursor() as cur:
            cur.execute("SELECT filters_json FROM saved_views WHERE name = %s LIMIT 1", (name,))
            r = cur.fetchone()
        if not r:
            return {}
        try:
            return json.loads(r[0] or "{}")
        except Exception:
            return {}

    def delete_saved_view(self, name: str) -> None:
        with self.con.cursor() as cur:
            cur.execute("DELETE FROM saved_views WHERE name = %s", (name,))

    # -----------------------------
    # Query
    # -----------------------------
    def query_items(
        self,
        min_priority: float = 0.0,
        status: Optional[str] = None,
        topic: Optional[str] = None,
        content_type: Optional[str] = None,
        lane: Optional[str] = None,
        lang: Optional[str] = None,
        source_kind: Optional[str] = None,
        tag_any: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        where = ["news_items.priority_score >= %s"]
        params: List[Any] = [float(min_priority)]

        if status and status != "all":
            where.append("news_items.status = %s")
            params.append(status)

        if topic and topic != "all":
            where.append("news_items.topic = %s")
            params.append(topic)

        if content_type and content_type != "all":
            where.append("news_items.content_type = %s")
            params.append(content_type)

        if lane and lane != "all":
            where.append("news_items.lane = %s")
            params.append(lane)

        if lang and lang != "all":
            where.append("news_items.lang = %s")
            params.append(lang)

        if source_kind and source_kind != "all":
            where.append("news_items.source_kind = %s")
            params.append(source_kind)

        if tag_any:
            tag_any_clean = sorted({(t or "").strip().lower() for t in tag_any if (t or "").strip()})
            if tag_any_clean:
                where.append(
                    """
                    EXISTS (
                      SELECT 1 FROM item_tags it
                      WHERE it.url = news_items.url
                        AND it.tag = ANY(%s)
                    )
                    """
                )
                params.append(tag_any_clean)

        if search:
            where.append("(lower(news_items.title) LIKE %s OR lower(news_items.content_text) LIKE %s)")
            s = f"%{search.lower()}%"
            params.extend([s, s])

        where_sql = " AND ".join(where)
        sql = f"""
        SELECT
          news_items.source,
          news_items.source_type,
          news_items.author_org,
          news_items.source_kind,
          news_items.source_weight,
          news_items.creator_name,
          news_items.title,
          news_items.url,
          news_items.published_at,
          news_items.lang,
          news_items.topic,
          news_items.content_type,
          news_items.content_type_confidence,
          news_items.lane,
          news_items.breakout_signal,
          news_items.novelty_score,
          news_items.relevance_score,
          news_items.actionability_score,
          news_items.recency_score,
          news_items.source_trust_score,
          news_items.source_mix_score,
          news_items.quality_score,
          news_items.quality_flags,
          news_items.priority_score,
          news_items.status,
          substring(news_items.content_text from 1 for 800) AS snippet
        FROM news_items
        WHERE {where_sql}
        ORDER BY news_items.priority_score DESC,
                 news_items.published_at DESC NULLS LAST,
                 news_items.fetched_at DESC
        LIMIT %s
        """
        params.append(int(limit))

        return pd.read_sql_query(sql, self.con, params=params)
