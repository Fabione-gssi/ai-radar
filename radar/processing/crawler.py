from __future__ import annotations

import re
import time
from html import unescape
from typing import Optional, Tuple
from urllib.parse import urlparse

DEFAULT_UA = "AI-News-Radar/1.0 (+internal monitoring)"


def _strip_html_basic(html: str) -> str:
    # rimuove script/style e tag
    x = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html or "")
    x = re.sub(r"(?is)<.*?>", " ", x)
    x = unescape(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def fetch_html(url: str, timeout_s: float = 10.0, user_agent: str = DEFAULT_UA, max_bytes: int = 2_000_000) -> Tuple[str, str]:
    """Ritorna (html, method). Nessuna dipendenza obbligatoria."""
    # 1) httpx (se presente)
    try:
        import httpx  # type: ignore

        with httpx.Client(timeout=timeout_s, headers={"User-Agent": user_agent}, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            content = r.content[:max_bytes]
            return content.decode(r.encoding or "utf-8", errors="ignore"), "httpx"
    except Exception:
        pass

    # 2) requests (se presente)
    try:
        import requests  # type: ignore

        r = requests.get(url, timeout=timeout_s, headers={"User-Agent": user_agent}, allow_redirects=True)
        r.raise_for_status()
        content = (r.content or b"")[:max_bytes]
        enc = r.encoding or "utf-8"
        return content.decode(enc, errors="ignore"), "requests"
    except Exception:
        pass

    # 3) urllib fallback
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            content = resp.read(max_bytes)
        return content.decode("utf-8", errors="ignore"), "urllib"
    except Exception:
        return "", "error"


def extract_main_text(html: str, url: str = "") -> Tuple[str, str]:
    """Ritorna (text, method). Prova prima trafilatura/readability, poi bs4, poi regex."""
    if not html:
        return "", "empty"

    # trafilatura (top per articoli)
    try:
        import trafilatura  # type: ignore

        txt = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 200:
            return txt, "trafilatura"
    except Exception:
        pass

    # readability-lxml + bs4
    try:
        from readability import Document  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore

        doc = Document(html)
        main = doc.summary(html_partial=True)
        soup = BeautifulSoup(main, "html.parser")
        txt = soup.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 200:
            return txt, "readability"
    except Exception:
        pass

    # BeautifulSoup semplice
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        txt = soup.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 200:
            return txt, "bs4"
    except Exception:
        pass

    # fallback regex
    txt = _strip_html_basic(html)
    return txt, "regex"


def fetch_article_text(
    url: str,
    timeout_s: float = 10.0,
    min_chars: int = 400,
    max_chars: int = 6000,
    sleep_s: float = 0.0,
) -> Tuple[str, str]:
    """
    Fetch+extract, con limiti. Restituisce (text, method).

    Nota: per evitare carico eccessivo, usa sleep_s > 0 se stai colpendo molte pagine dello stesso dominio.
    """
    if not url:
        return "", "empty_url"

    if sleep_s and sleep_s > 0:
        time.sleep(float(sleep_s))

    html, m1 = fetch_html(url, timeout_s=timeout_s)
    if not html:
        return "", m1

    text, m2 = extract_main_text(html, url=url)
    if not text or len(text) < min_chars:
        return "", f"{m1}+{m2}+too_short"

    return text[:max_chars], f"{m1}+{m2}"
