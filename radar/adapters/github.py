from __future__ import annotations

from typing import Dict, List
import os
import requests

from radar.utils import safe_text, canonicalize_url, new_uuid, utc_now, parse_datetime_maybe


def _headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "ai-news-radar/1.0"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def fetch_latest_releases(repos: List[str], timeout: int = 20) -> List[Dict]:
    out: List[Dict] = []
    for full in repos:
        if "/" not in full:
            continue

        owner, repo = full.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        r = requests.get(url, headers=_headers(), timeout=timeout)

        if r.status_code == 404:
            # repo senza release/latest
            continue
        r.raise_for_status()

        j = r.json()
        html_url = canonicalize_url(j.get("html_url", ""))
        title = j.get("name") or j.get("tag_name") or f"{full} release"
        body = safe_text(j.get("body", ""))
        prerelease = bool(j.get("prerelease", False))
        draft = bool(j.get("draft", False))
        tag = safe_text(j.get("tag_name", ""))
        published_at = parse_datetime_maybe(j.get("published_at"))

        out.append({
            "id": new_uuid(),
            "source": "github_release",
            "source_type": "api",
            "author_org": owner,
            "url": html_url,
            "title": safe_text(title),
            "published_at": published_at,
            "fetched_at": utc_now(),
            "content_text": safe_text(
                f"{title}. repo={full}. tag={tag}. prerelease={prerelease}. draft={draft}. {body}"
            ),
            "source_kind": "institutional",
            "source_weight": 1.0,
            "content_type_hint": "release",
        })
    return out


def fetch_discovery_repos(
    queries: List[str],
    per_query: int = 20,
    timeout: int = 20,
    min_stars: int = 30,
) -> List[Dict]:
    """
    Discovery lane:
    usa GitHub Search API per trovare repo AI attivi/recenti,
    oltre la tua whitelist statica.
    """
    out: List[Dict] = []
    seen = set()

    for q in queries or []:
        if not q:
            continue

        r = requests.get(
            "https://api.github.com/search/repositories",
            params={
                "q": q,
                "sort": "updated",
                "order": "desc",
                "per_page": max(1, min(int(per_query), 100)),
            },
            headers=_headers(),
            timeout=timeout,
        )
        r.raise_for_status()

        for repo in (r.json().get("items") or []):
            full_name = repo.get("full_name") or ""
            if not full_name or full_name in seen:
                continue
            seen.add(full_name)

            stars = int(repo.get("stargazers_count", 0) or 0)
            if stars < int(min_stars):
                continue

            owner = ((repo.get("owner") or {}).get("login") or "").strip()
            html_url = canonicalize_url(repo.get("html_url", ""))
            desc = safe_text(repo.get("description", ""))
            topics = repo.get("topics") or []
            language = repo.get("language") or ""
            pushed_at = parse_datetime_maybe(repo.get("pushed_at"))

            out.append({
                "id": new_uuid(),
                "source": "github_discovery",
                "source_type": "api",
                "author_org": owner or "github",
                "url": html_url,
                "title": safe_text(f"GitHub discovery: {full_name}"),
                "published_at": pushed_at,
                "fetched_at": utc_now(),
                "content_text": safe_text(
                    f"{full_name}. stars={stars}. language={language}. topics={topics}. description={desc}"
                ),
                "source_kind": "institutional",
                "source_weight": 0.75,
                "content_type_hint": "tool",
            })

    return out
