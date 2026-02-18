from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from bs4 import BeautifulSoup


def new_uuid() -> str:
    """Genera un id unico per ogni news item."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Timestamp UTC coerente per tutto il progetto."""
    return datetime.now(timezone.utc)


def canonicalize_url(url: str) -> str:
    """
    Normalizza URL per dedup: rimuove fragment, ordina query params
    e ripulisce trailing slash.
    """
    try:
        u = urlparse(url.strip())
        # Rimuove fragment (#...)
        fragmentless = u._replace(fragment="")
        # Ordina query params (a=b&c=d) per confronto stabile
        q = parse_qsl(fragmentless.query, keep_blank_values=True)
        q_sorted = urlencode(sorted(q))
        cleaned = fragmentless._replace(query=q_sorted)
        out = urlunparse(cleaned)
        # Uniforma trailing slash
        out = out[:-1] if out.endswith("/") else out
        return out
    except Exception:
        return url.strip()


def strip_html(text: str) -> str:
    """Rimuove HTML e mantiene testo leggibile."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


def normalize_whitespace(text: str) -> str:
    """Compatta spazi e newline."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_text(text: str, max_len: int = 20000) -> str:
    """Ripulisce e tronca testi troppo lunghi (evita DB gonfi)."""
    text = strip_html(text)
    text = normalize_whitespace(text)
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def parse_datetime_maybe(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Prova a parse di datetime da stringa ISO o formati comuni.
    Se fallisce ritorna None.
    """
    if not dt_str:
        return None
    # Prova ISO
    try:
        # gestisce "Z"
        s = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        pass

    # Fallback: prova dateutil se disponibile (dipendenza già inclusa)
    try:
        from dateutil import parser
        return parser.parse(dt_str)
    except Exception:
        return None
