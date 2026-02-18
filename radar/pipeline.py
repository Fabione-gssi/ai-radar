from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from radar.config import Config, load_config
from radar.db import NewsDB
from radar.utils import safe_text, canonicalize_url

from radar.processing.lang import detect_lang_with_confidence
from radar.processing.classify import classify_item
from radar.processing.score import (
    compute_actionability,
    compute_breakout_signal,
    compute_recency_score,
    compute_source_mix_score,
    compute_source_trust,
    combine_priority,
)
from radar.processing.embed import embed_texts, novelty_score
from radar.processing.crawler import fetch_article_text

from radar.adapters.arxiv import fetch_arxiv
from radar.adapters.github import fetch_latest_releases, fetch_discovery_repos
from radar.adapters.huggingface import fetch_recent_models, keyword_filter, split_known_vs_emerging
from radar.adapters.rss import fetch_rss


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any((kw or "").lower() in t for kw in (keywords or []))


def _rss_sources(cfg_sources: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    rss_cfg = cfg_sources.get("rss", {}) or {}
    for f in rss_cfg.get("feeds", []) or []:
        x = dict(f)
        x.setdefault("source_kind", "institutional")
        x.setdefault("source_weight", 1.0)
        x.setdefault("creator_name", "")
        x.setdefault("lane_hint", "reliable")
        out.append(x)

    creators_cfg = cfg_sources.get("creators", {}) or {}
    for f in creators_cfg.get("feeds", []) or []:
        x = dict(f)
        x.setdefault("source_kind", "creator")
        x.setdefault("source_weight", 1.0)
        x.setdefault("creator_name", x.get("name", ""))
        x.setdefault("lane_hint", "scout")
        out.append(x)

    return out


def collect_items(cfg: Config) -> List[Dict[str, Any]]:
    """
    Esegue tutti i collector abilitati e ritorna lista di item grezzi.
    (isolando errori per fonte: un feed che cade non blocca tutto)
    """
    items: List[Dict[str, Any]] = []
    sources = cfg.sources

    # arXiv
    arxiv_cfg = sources.get("arxiv", {}) or {}
    if arxiv_cfg.get("enabled", False):
        for q in arxiv_cfg.get("queries", []) or []:
            query = q.get("search_query", "")
            max_results = int(q.get("max_results", 25))
            if not query:
                continue
            try:
                batch = fetch_arxiv(query=query, max_results=max_results)
                for it in batch:
                    it["lane_hint"] = "reliable"
                items.extend(batch)
            except Exception:
                continue

    # GitHub releases (core)
    gh_cfg = sources.get("github", {}) or {}
    if gh_cfg.get("enabled", False):
        repos = gh_cfg.get("repos", []) or []
        if repos:
            try:
                batch = fetch_latest_releases(repos=repos)
                for it in batch:
                    it["lane_hint"] = "reliable"
                items.extend(batch)
            except Exception:
                pass

    # GitHub discovery (scout)
    gh_disc_cfg = sources.get("github_discovery", {}) or {}
    if gh_disc_cfg.get("enabled", False):
        queries = gh_disc_cfg.get("queries", []) or []
        per_query = int(gh_disc_cfg.get("per_query", 20))
        min_stars = int(gh_disc_cfg.get("min_stars", 30))
        cap = int(gh_disc_cfg.get("cap", 20))
        try:
            disc = fetch_discovery_repos(
                queries=queries,
                per_query=per_query,
                min_stars=min_stars,
            )
            for it in disc[:cap]:
                it["lane_hint"] = "scout"
                items.append(it)
        except Exception:
            pass

    # Hugging Face recent models
    hf_cfg = sources.get("huggingface", {}) or {}
    if hf_cfg.get("enabled", False):
        try:
            max_models = int(hf_cfg.get("max_models", 80))
            raw = fetch_recent_models(max_models=max_models)
            filt = keyword_filter(
                raw,
                include=hf_cfg.get("include_keywords", []) or [],
                exclude=hf_cfg.get("exclude_keywords", []) or [],
            )

            known, emerg = split_known_vs_emerging(
                filt,
                known_authors=hf_cfg.get("allow_authors", []) or [],
            )

            for it in known:
                it["lane_hint"] = "reliable"
                items.append(it)

            discover_cap = int(hf_cfg.get("discover_cap", 20))
            for it in emerg[:discover_cap]:
                it["lane_hint"] = "scout"
                items.append(it)
        except Exception:
            pass

    # RSS (istituzionali + creator)
    rss_cfg = sources.get("rss", {}) or {}
    if rss_cfg.get("enabled", False):
        include_kw = rss_cfg.get("include_keywords", []) or []
        exclude_kw = rss_cfg.get("exclude_keywords", []) or []

        for f in _rss_sources(sources):
            url = f.get("url")
            name = f.get("name", "rss")
            if not url:
                continue

            try:
                batch = fetch_rss(feed_url=url, feed_name=name, meta=f)
            except Exception:
                continue

            for it in batch:
                blob = f"{it.get('title','')} {it.get('content_text','')}".lower()
                if include_kw and not _contains_any(blob, include_kw):
                    continue
                if exclude_kw and _contains_any(blob, exclude_kw):
                    continue

                it["source_kind"] = f.get("source_kind", it.get("source_kind", "institutional"))
                it["source_weight"] = float(f.get("source_weight", it.get("source_weight", 1.0)))
                it["creator_name"] = f.get("creator_name", it.get("creator_name", ""))
                it["lane_hint"] = f.get("lane_hint", "reliable")
                items.append(it)

    return items


def enrich_and_store(base_dir: Path) -> Dict[str, Any]:
    """
    Pipeline completa precision-first:
    - carica config
    - raccoglie items
    - normalizza + dedup URL (batch)
    - (opzionale) full-text fetch RSS dopo dedup
    - arricchisce (lang/topic/type/quality/trust/actionability/recency/novelty/mix)
    - quality gate pre-store (diverso per reliable/scout)
    - salva su DuckDB
    """
    cfg = load_config(base_dir)

    db_path = base_dir / "data" / "news.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = NewsDB(db_path)

    try:
        raw_items = collect_items(cfg)

        # 1) normalizzazione base
        cleaned: List[Dict[str, Any]] = []
        for it in raw_items:
            it["url"] = canonicalize_url(it.get("url", ""))
            it["title"] = safe_text(it.get("title", ""))
            it["content_text"] = safe_text(it.get("content_text", ""))
            if not it["url"] or not it["title"]:
                continue
            it.setdefault("lane_hint", "reliable")
            cleaned.append(it)

        if not cleaned:
            return {
                "fetched": 0,
                "new_items": 0,
                "skipped_existing": 0,
                "skipped_language": 0,
                "skipped_quality": 0,
                "skipped_trust": 0,
                "stored": 0,
                "lane_counts": {},
                "content_type_counts": {},
            }

        # 2) dedup batch
        urls = [it["url"] for it in cleaned]
        existing = db.get_existing_urls(urls)
        deduped = [it for it in cleaned if it["url"] not in existing]
        skipped_existing = len(cleaned) - len(deduped)

        if not deduped:
            return {
                "fetched": len(raw_items),
                "new_items": 0,
                "skipped_existing": skipped_existing,
                "skipped_language": 0,
                "skipped_quality": 0,
                "skipped_trust": 0,
                "stored": 0,
                "lane_counts": {},
                "content_type_counts": {},
            }

        # 3) cap per fonte per run (anti-flood)
        max_per_source = int(cfg.ranking.get("max_items_per_source_per_run", 30))
        per_source: Dict[str, int] = defaultdict(int)
        capped: List[Dict[str, Any]] = []
        for it in deduped:
            src = (it.get("source") or "unknown").lower().strip()
            if per_source[src] >= max_per_source:
                continue
            per_source[src] += 1
            capped.append(it)

        # 4) language policy (con conf)
        lp = cfg.sources.get("language_policy", {}) or {}
        allowed_langs = list(lp.get("allowed", ["it", "en"]))
        drop_disallowed = bool(lp.get("drop_disallowed", True))
        min_lang_conf = float(lp.get("min_confidence", 0.60))

        prepped: List[Dict[str, Any]] = []
        skipped_language = 0

        for it in capped:
            text_blob = f"{it.get('title','')} {it.get('content_text','')}"
            lang, conf, method = detect_lang_with_confidence(
                text_blob,
                allowed_langs=allowed_langs,
                min_confidence=min_lang_conf,
            )

            # se conf troppo bassa → unknown
            if conf < min_lang_conf and lang not in {"unknown", "other"}:
                lang = "unknown"

            if drop_disallowed and lang == "other":
                skipped_language += 1
                continue

            it["lang"] = lang
            it["lang_confidence"] = conf
            it["lang_method"] = method

            prepped.append(it)

        if not prepped:
            return {
                "fetched": len(raw_items),
                "new_items": len(capped),
                "skipped_existing": skipped_existing,
                "skipped_language": skipped_language,
                "skipped_quality": 0,
                "skipped_trust": 0,
                "stored": 0,
                "lane_counts": {},
                "content_type_counts": {},
            }

        # 5) full-text fetch (solo RSS) dopo dedup+lang, con cap globale
        fulltext_cap = int(cfg.ranking.get("fulltext_max_fetch_per_run", 18))
        fulltext_done = 0
        for it in prepped:
            if fulltext_done >= fulltext_cap:
                break
            if (it.get("source_type") or "") != "rss":
                continue
            if not bool(it.get("fetch_fulltext", False)):
                continue

            url = it.get("url", "")
            timeout_s = float(it.get("fulltext_timeout_s", 10.0))
            min_chars = int(it.get("fulltext_min_chars", 400))
            max_chars = int(it.get("fulltext_max_chars", 6000))
            sleep_s = float(it.get("fulltext_sleep_s", 0.0))

            txt, _method = fetch_article_text(
                url=url,
                timeout_s=timeout_s,
                min_chars=min_chars,
                max_chars=max_chars,
                sleep_s=sleep_s,
            )
            # sostituisci solo se significativamente migliore del feed summary
            if txt and len(txt) > len(it.get("content_text", "")) + 300:
                it["content_text"] = safe_text(f"{it.get('title','')}. {txt}")
                it["fulltext_method"] = _method
                fulltext_done += 1

        # 6) classification + basic scoring (no novelty yet)
        min_trust_core = float(cfg.ranking.get("min_trust_core", 0.60))
        min_trust_scout = float(cfg.ranking.get("min_trust_scout", 0.40))
        min_creator_trust = float(cfg.ranking.get("min_creator_trust", 0.75))
        min_type_conf_core = float(cfg.ranking.get("min_type_confidence_core", 0.28))
        min_type_conf_scout = float(cfg.ranking.get("min_type_confidence_scout", 0.33))

        enriched: List[Dict[str, Any]] = []
        skipped_trust = 0

        for it in prepped:
            text_blob = f"{it.get('title','')} {it.get('content_text','')}"

            cls = classify_item(
                text=text_blob,
                taxonomy=cfg.taxonomy,
                source=str(it.get("source", "")),
                source_type=str(it.get("source_type", "")),
                url=str(it.get("url", "")),
                content_type_hint=str(it.get("content_type_hint", "")),
                lang=str(it.get("lang", "unknown")),
            )
            it.update({
                "topic": cls.get("topic", "Other"),
                "relevance_score": float(cls.get("relevance_score", 0.0)),
                "content_type": cls.get("content_type", "news"),
                "content_type_confidence": float(cls.get("content_type_confidence", 0.0)),
                "quality_score": float(cls.get("quality_score", 0.0)),
                "quality_flags": cls.get("quality_flags", ""),
            })

            it["actionability_score"] = compute_actionability(it.get("content_text", ""), it.get("url", ""))

            it["source_trust_score"] = compute_source_trust(
                author_org=str(it.get("author_org", "")),
                trust_cfg=cfg.trust,
                source_kind=str(it.get("source_kind", "institutional")),
                creator_name=str(it.get("creator_name", "")),
            )

            it["recency_score"] = compute_recency_score(
                it.get("published_at"),
                half_life_days=float(cfg.ranking.get("recency_half_life_days", 7)),
            )

            lane = "scout" if it.get("lane_hint") == "scout" else "reliable"

            # trust gate: precision-first (più severo sui creator)
            trust_min = min_trust_scout if lane == "scout" else min_trust_core
            if (it.get("source_kind") or "") == "creator":
                trust_min = max(trust_min, min_creator_trust)

            if float(it["source_trust_score"]) < trust_min:
                skipped_trust += 1
                continue

            # type confidence gate (riduce rumore rss generico)
            type_conf_min = min_type_conf_scout if lane == "scout" else min_type_conf_core
            if float(it["content_type_confidence"]) < type_conf_min:
                # non scartare tutto: ma se è "news" con conf bassa, via
                if str(it.get("content_type")) == "news":
                    continue

            enriched.append(it)

        if not enriched:
            return {
                "fetched": len(raw_items),
                "new_items": len(capped),
                "skipped_existing": skipped_existing,
                "skipped_language": skipped_language,
                "skipped_quality": 0,
                "skipped_trust": skipped_trust,
                "stored": 0,
                "lane_counts": {},
                "content_type_counts": {},
            }

        # 7) source mix score (anti-flood, precision-first)
        source_counts = Counter([(it.get("source") or "unknown").lower().strip() for it in enriched])
        for it in enriched:
            it["source_mix_score"] = compute_source_mix_score(
                source_counts=source_counts,
                source=(it.get("source") or ""),
                source_kind=str(it.get("source_kind", "institutional")),
                source_weight=float(it.get("source_weight", 1.0)),
                ranking_cfg=cfg.ranking,
                trust=float(it.get("source_trust_score", 0.0)),
            )

        # 8) novelty (embedding) per ultime N
        n_compare = int(cfg.ranking.get("novelty_compare_last_n", 200))
        recent_texts = db.get_recent_texts(limit=n_compare)
        recent_contents = [t[1] for t in recent_texts if t[1]]

        recent_embs = embed_texts(recent_contents) if recent_contents else None
        new_texts = [f"{it['title']}. {it['content_text']}" for it in enriched]
        new_embs = embed_texts(new_texts)

        min_quality_core = float(cfg.ranking.get("min_quality_core", 0.45))
        min_quality_scout = float(cfg.ranking.get("min_quality_scout", 0.62))
        breakout_promote_threshold = float(cfg.ranking.get("breakout_promote_threshold", 0.68))

        stored = 0
        skipped_quality = 0
        lane_counts = Counter()
        content_type_counts = Counter()

        for it, emb in zip(enriched, new_embs):
            it["novelty_score"] = novelty_score(emb, recent_embs)

            base_scores = {
                "source_trust": float(it.get("source_trust_score", 0.0)),
                "novelty": float(it.get("novelty_score", 0.0)),
                "relevance": float(it.get("relevance_score", 0.0)),
                "actionability": float(it.get("actionability_score", 0.0)),
                "recency": float(it.get("recency_score", 0.0)),
                "source_mix": float(it.get("source_mix_score", 0.0)),
            }
            base_priority = combine_priority(base_scores, cfg.ranking)

            lane = "scout" if it.get("lane_hint") == "scout" else "reliable"
            breakout = compute_breakout_signal(
                text=f"{it.get('title','')} {it.get('content_text','')}",
                source=str(it.get("source", "")),
                novelty=float(it.get("novelty_score", 0.0)),
                recency=float(it.get("recency_score", 0.0)),
                actionability=float(it.get("actionability_score", 0.0)),
            )
            if breakout >= breakout_promote_threshold:
                lane = "scout"

            it["lane"] = lane
            it["breakout_signal"] = breakout

            # boost scout LIMITATO (precision-first): solo se trust alto
            scout_boost = 0.10 * breakout + 0.06 * float(it.get("novelty_score", 0.0))
            if lane == "scout" and float(it.get("source_trust_score", 0.0)) >= 0.80:
                it["priority_score"] = min(1.0, base_priority + scout_boost)
            else:
                it["priority_score"] = base_priority

            q_min = min_quality_scout if lane == "scout" else min_quality_core
            if float(it.get("quality_score", 0.0)) < q_min:
                skipped_quality += 1
                continue

            it["status"] = "new"
            db.upsert_item(it)
            stored += 1
            lane_counts[lane] += 1
            content_type_counts[str(it.get("content_type", "news"))] += 1

        return {
            "fetched": len(raw_items),
            "new_items": len(capped),
            "skipped_existing": skipped_existing,
            "skipped_language": skipped_language,
            "skipped_quality": skipped_quality,
            "skipped_trust": skipped_trust,
            "stored": stored,
            "lane_counts": dict(lane_counts),
            "content_type_counts": dict(content_type_counts),
        }

    finally:
        db.close()
