from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS news_items (
  id                      VARCHAR,
  source                  VARCHAR,
  source_type             VARCHAR,
  author_org              VARCHAR,
  source_kind             VARCHAR,
  source_weight           DOUBLE,
  creator_name            VARCHAR,

  url                     VARCHAR,
  title                   VARCHAR,
  published_at            TIMESTAMP,
  fetched_at              TIMESTAMP,
  lang                    VARCHAR,
  content_text            VARCHAR,

  topic                   VARCHAR,
  content_type            VARCHAR,
  content_type_confidence DOUBLE,
  lane                    VARCHAR,   -- reliable | scout
  breakout_signal         DOUBLE,

  novelty_score           DOUBLE,
  relevance_score         DOUBLE,
  actionability_score     DOUBLE,
  recency_score           DOUBLE,
  source_trust_score      DOUBLE,
  source_mix_score        DOUBLE,
  quality_score           DOUBLE,
  quality_flags           VARCHAR,

  priority_score          DOUBLE,
  status                  VARCHAR,

  PRIMARY KEY(url)
);

CREATE TABLE IF NOT EXISTS tags (
  tag         VARCHAR PRIMARY KEY,
  created_at  TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS item_tags (
  url         VARCHAR,
  tag         VARCHAR,
  created_at  TIMESTAMP DEFAULT now(),
  PRIMARY KEY(url, tag)
);

CREATE TABLE IF NOT EXISTS saved_views (
  name         VARCHAR PRIMARY KEY,
  filters_json VARCHAR,
  created_at   TIMESTAMP DEFAULT now(),
  updated_at   TIMESTAMP DEFAULT now()
);
"""


class NewsDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.con = duckdb.connect(str(db_path))
        self.con.execute(SCHEMA_SQL)
        self._migrate_schema_if_needed()

    def _migrate_schema_if_needed(self) -> None:
        rows = self.con.execute("PRAGMA table_info('news_items')").fetchall()
        existing = {r[1] for r in rows}

        add_cols = {
            "source_kind": "VARCHAR",
            "source_weight": "DOUBLE",
            "creator_name": "VARCHAR",
            "content_type": "VARCHAR",
            "content_type_confidence": "DOUBLE",
            "lane": "VARCHAR",
            "breakout_signal": "DOUBLE",
            "source_mix_score": "DOUBLE",
            "quality_score": "DOUBLE",
            "quality_flags": "VARCHAR",
        }
        for col, typ in add_cols.items():
            if col not in existing:
                self.con.execute(f"ALTER TABLE news_items ADD COLUMN {col} {typ}")

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    # -----------------------------
    # News items
    # -----------------------------
    def upsert_item(self, item: Dict[str, Any]) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO news_items
            (id, source, source_type, author_org, source_kind, source_weight, creator_name,
             url, title, published_at, fetched_at, lang, content_text,
             topic, content_type, content_type_confidence, lane, breakout_signal,
             novelty_score, relevance_score, actionability_score, recency_score,
             source_trust_score, source_mix_score, quality_score, quality_flags,
             priority_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?)
            """,
            [
                item.get("id"),
                item.get("source"),
                item.get("source_type"),
                item.get("author_org"),
                item.get("source_kind", "institutional"),
                float(item.get("source_weight", 1.0)),
                item.get("creator_name", ""),
                item.get("url"),
                item.get("title"),
                item.get("published_at"),
                item.get("fetched_at"),
                item.get("lang"),
                item.get("content_text"),
                item.get("topic"),
                item.get("content_type", "news"),
                float(item.get("content_type_confidence", 0.0)),
                item.get("lane", "reliable"),
                float(item.get("breakout_signal", 0.0)),
                float(item.get("novelty_score", 0.0)),
                float(item.get("relevance_score", 0.0)),
                float(item.get("actionability_score", 0.0)),
                float(item.get("recency_score", 0.0)),
                float(item.get("source_trust_score", 0.0)),
                float(item.get("source_mix_score", 0.0)),
                float(item.get("quality_score", 0.0)),
                item.get("quality_flags", ""),
                float(item.get("priority_score", 0.0)),
                item.get("status", "new"),
            ],
        )

    def url_exists(self, url: str) -> bool:
        r = self.con.execute("SELECT 1 FROM news_items WHERE url = ? LIMIT 1", [url]).fetchone()
        return r is not None

    def get_existing_urls(self, urls: List[str]) -> set[str]:
        """Batch check per dedup (molto più veloce di N query)."""
        clean = [u for u in {(u or "").strip() for u in (urls or [])} if u]
        if not clean:
            return set()
        placeholders = ",".join(["?"] * len(clean))
        rows = self.con.execute(
            f"SELECT url FROM news_items WHERE url IN ({placeholders})",
            clean,
        ).fetchall()
        return {r[0] for r in rows}

    def get_recent_texts(self, limit: int = 200) -> List[Tuple[str, str]]:
        rows = self.con.execute(
            """
            SELECT url, content_text
            FROM news_items
            ORDER BY published_at DESC NULLS LAST, fetched_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [(r[0], r[1] or "") for r in rows]

    def update_status(self, url: str, status: str) -> None:
        self.con.execute("UPDATE news_items SET status = ? WHERE url = ?", [status, url])

    # -----------------------------
    # Tags
    # -----------------------------
    def list_tags(self) -> List[str]:
        rows = self.con.execute("SELECT tag FROM tags ORDER BY tag").fetchall()
        return [r[0] for r in rows]

    def tag_counts(self, limit: int = 50) -> List[tuple[str, int]]:
        """Ritorna (tag, count) ordinati per frequenza."""
        rows = self.con.execute(
            """
            SELECT tag, COUNT(*) as n
            FROM item_tags
            GROUP BY tag
            ORDER BY n DESC, tag ASC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    def add_tags(self, tags: List[str]) -> None:
        clean = sorted({(t or "").strip().lower() for t in tags if (t or "").strip()})
        for t in clean:
            self.con.execute("INSERT OR IGNORE INTO tags(tag) VALUES (?)", [t])

    def assign_tags_bulk(self, urls: List[str], tags: List[str]) -> int:
        if not urls or not tags:
            return 0
        clean_tags = sorted({(t or "").strip().lower() for t in tags if (t or "").strip()})
        clean_urls = sorted({(u or "").strip() for u in urls if (u or "").strip()})
        if not clean_tags or not clean_urls:
            return 0

        self.add_tags(clean_tags)

        inserted = 0
        for u in clean_urls:
            for t in clean_tags:
                try:
                    r = self.con.execute(
                        "INSERT OR IGNORE INTO item_tags(url, tag) VALUES (?, ?) RETURNING 1",
                        [u, t],
                    ).fetchone()
                    if r is not None:
                        inserted += 1
                except Exception:
                    # fallback compatibilità: conta come inserito solo se non esisteva
                    exists = self.con.execute(
                        "SELECT 1 FROM item_tags WHERE url = ? AND tag = ? LIMIT 1",
                        [u, t],
                    ).fetchone()
                    if exists is None:
                        self.con.execute("INSERT OR IGNORE INTO item_tags(url, tag) VALUES (?, ?)", [u, t])
                        inserted += 1
        return inserted

    def remove_tags_bulk(self, urls: List[str], tags: List[str]) -> int:
        if not urls or not tags:
            return 0
        clean_tags = sorted({(t or "").strip().lower() for t in tags if (t or "").strip()})
        clean_urls = sorted({(u or "").strip() for u in urls if (u or "").strip()})
        removed = 0
        for u in clean_urls:
            for t in clean_tags:
                self.con.execute("DELETE FROM item_tags WHERE url = ? AND tag = ?", [u, t])
                removed += 1
        return removed

    def get_tags_map(self, urls: List[str]) -> Dict[str, List[str]]:
        clean_urls = sorted({(u or "").strip() for u in urls if (u or "").strip()})
        if not clean_urls:
            return {}

        placeholders = ",".join(["?"] * len(clean_urls))
        rows = self.con.execute(
            f"""
            SELECT url, tag
            FROM item_tags
            WHERE url IN ({placeholders})
            ORDER BY tag
            """,
            clean_urls,
        ).fetchall()

        out: Dict[str, List[str]] = {u: [] for u in clean_urls}
        for u, t in rows:
            out.setdefault(u, []).append(t)
        return out

    # -----------------------------
    # Saved views
    # -----------------------------
    def list_saved_views(self) -> List[str]:
        rows = self.con.execute("SELECT name FROM saved_views ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def save_view(self, name: str, filters: Dict[str, Any]) -> None:
        n = (name or "").strip()
        if not n:
            return
        payload = json.dumps(filters, ensure_ascii=False)
        self.con.execute(
            "INSERT OR REPLACE INTO saved_views(name, filters_json, updated_at) VALUES (?, ?, now())",
            [n, payload],
        )

    def get_saved_view(self, name: str) -> Dict[str, Any]:
        r = self.con.execute(
            "SELECT filters_json FROM saved_views WHERE name = ? LIMIT 1",
            [name],
        ).fetchone()
        if not r:
            return {}
        try:
            return json.loads(r[0] or "{}")
        except Exception:
            return {}

    def delete_saved_view(self, name: str) -> None:
        self.con.execute("DELETE FROM saved_views WHERE name = ?", [name])

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
    ):
        where = ["news_items.priority_score >= ?"]
        params: List[Any] = [min_priority]

        if status and status != "all":
            where.append("news_items.status = ?")
            params.append(status)

        if topic and topic != "all":
            where.append("news_items.topic = ?")
            params.append(topic)

        if content_type and content_type != "all":
            where.append("news_items.content_type = ?")
            params.append(content_type)

        if lane and lane != "all":
            where.append("news_items.lane = ?")
            params.append(lane)

        if lang and lang != "all":
            where.append("news_items.lang = ?")
            params.append(lang)

        if source_kind and source_kind != "all":
            where.append("news_items.source_kind = ?")
            params.append(source_kind)

        if tag_any:
            tag_any_clean = sorted({(t or "").strip().lower() for t in tag_any if (t or "").strip()})
            if tag_any_clean:
                placeholders = ",".join(["?"] * len(tag_any_clean))
                where.append(
                    f"""EXISTS (
                        SELECT 1
                        FROM item_tags it
                        WHERE it.url = news_items.url
                          AND it.tag IN ({placeholders})
                    )"""
                )
                params.extend(tag_any_clean)

        if search:
            where.append("(lower(news_items.title) LIKE ? OR lower(news_items.content_text) LIKE ?)")
            s = f"%{search.lower()}%"
            params.extend([s, s])

        where_sql = " AND ".join(where)

        return self.con.execute(
            f"""
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
              substr(news_items.content_text, 1, 800) AS snippet
            FROM news_items
            WHERE {where_sql}
            ORDER BY news_items.priority_score DESC, news_items.published_at DESC NULLS LAST, news_items.fetched_at DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchdf()
