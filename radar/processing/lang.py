from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from langcodes import standardize_tag

try:
    import fasttext  # type: ignore
except Exception:  # pragma: no cover
    fasttext = None


# Stopwords minimali per fallback euristico
_STOPWORDS: Dict[str, set[str]] = {
    "it": {"il","lo","la","gli","le","un","una","di","che","per","con","nel","nella","della","delle","dei","ai","allo"},
    "en": {"the","and","for","with","new","from","on","in","to","of","is","are","release","paper"},
    "fr": {"le","la","les","de","des","et","dans","pour","une","un","avec","sur","du"},
    "es": {"el","la","los","las","de","del","y","con","para","una","un","en"},
    "de": {"der","die","das","und","mit","für","ein","eine","im","von","zu"},
}


def _normalize_text(text: str, max_chars: int = 8000) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zàèéìòùçñäöüß]{2,}", text.lower())


def _score_by_stopwords(text: str) -> Tuple[str, float]:
    tokens = _tokenize(text)
    if len(tokens) < 4:
        return "unknown", 0.0

    counts: Dict[str, int] = {k: 0 for k in _STOPWORDS}
    for lang, sw in _STOPWORDS.items():
        counts[lang] = sum(1 for t in tokens if t in sw)

    best_lang = max(counts, key=counts.get)
    best = counts[best_lang]
    ordered = sorted(counts.values(), reverse=True)
    second = ordered[1] if len(ordered) > 1 else 0

    if best == 0:
        return "unknown", 0.0

    # conf semplice basata su gap best-second
    conf = min(1.0, (best - second + 1.0) / max(3.0, len(tokens) * 0.18))
    return best_lang, float(conf)


@lru_cache(maxsize=1)
def _load_fasttext_model() -> Optional[object]:
    """
    Cerca modello LID fastText in:
    - env RADAR_FASTTEXT_MODEL_PATH
    - ./models/lid.176.ftz
    - ~/.cache/ai_news_radar/lid.176.ftz
    """
    if fasttext is None:
        return None

    candidates = []
    env_path = os.getenv("RADAR_FASTTEXT_MODEL_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        Path.cwd() / "models" / "lid.176.ftz",
        Path.home() / ".cache" / "ai_news_radar" / "lid.176.ftz",
    ])

    for p in candidates:
        if p.exists():
            try:
                return fasttext.load_model(str(p))
            except Exception:
                continue
    return None


def _normalize_fasttext_label(label: str) -> str:
    raw = (label or "").replace("__label__", "").strip().lower()
    if not raw:
        return "unknown"
    try:
        return standardize_tag(raw).split("-")[0]
    except Exception:
        return raw.split("-")[0]


def detect_lang_with_confidence(
    text: str,
    allowed_langs: Optional[Iterable[str]] = ("it", "en"),
    min_confidence: float = 0.60,
) -> Tuple[str, float, str]:
    """
    Ritorna: (lang, confidence, method)
    lang è:
      - 'it'/'en'/... se ammessa
      - 'other' se detect valida ma fuori allowlist
      - 'unknown' se non determinabile
    """
    clean = _normalize_text(text)
    if not clean:
        return "unknown", 0.0, "empty"

    allowed = {x.lower() for x in (allowed_langs or []) if x}
    model = _load_fasttext_model()

    # 1) fastText (se disponibile)
    if model is not None:
        try:
            labels, probs = model.predict(clean, k=1)
            lang = _normalize_fasttext_label(labels[0] if labels else "")
            conf = float(probs[0]) if probs else 0.0

            # fallback euristico se conf bassa
            if conf < min_confidence:
                h_lang, h_conf = _score_by_stopwords(clean)
                if h_conf > conf:
                    lang, conf = h_lang, h_conf

            if allowed and lang not in allowed and lang != "unknown":
                return "other", conf, "fasttext+allowed"

            return lang, conf, "fasttext"
        except Exception:
            pass

    # 2) euristico puro
    lang, conf = _score_by_stopwords(clean)
    if allowed and lang not in allowed and lang != "unknown":
        return "other", conf, "heuristic+allowed"
    return lang, conf, "heuristic"


def detect_lang(text: str) -> str:
    lang, _, _ = detect_lang_with_confidence(text)
    return lang
