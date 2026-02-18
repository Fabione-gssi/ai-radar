from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
import os


@dataclass(frozen=True)
class Config:
    sources: Dict[str, Any]
    trust: Dict[str, Any]
    taxonomy: Dict[str, Any]
    ranking: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File config mancante: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _norm_key(x: str) -> str:
    return (x or "").strip().lower()


def _merge_editorial_whitelist(trust: Dict[str, Any], editorial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge editoriale (versionabile) dentro trust.yaml.

    editorial_whitelist.yaml può contenere:
      creators:
        - name: "simone rizzo"
          trust: 0.82
          aliases: ["simone-rizzo", "s rizzo"]
      orgs:
        - name: "openai"
          trust: 0.95
          aliases: ["openai news"]
    """
    out = dict(trust or {})

    out.setdefault("org_trust", {})
    out.setdefault("creator_trust", {})
    out.setdefault("org_alias", {})
    out.setdefault("creator_alias", {})

    strict = editorial.get("strict_creator_whitelist")
    if strict is not None:
        out["strict_creator_whitelist"] = bool(strict)
    else:
        out.setdefault("strict_creator_whitelist", True)

    # creators
    for c in (editorial.get("creators") or []):
        if not isinstance(c, dict):
            continue
        name = _norm_key(c.get("name", ""))
        if not name:
            continue
        if "trust" in c and c["trust"] is not None:
            out["creator_trust"][name] = float(c["trust"])
        # aliases
        for a in (c.get("aliases") or []):
            ak = _norm_key(a)
            if ak:
                out["creator_alias"][ak] = name

    # orgs
    for o in (editorial.get("orgs") or []):
        if not isinstance(o, dict):
            continue
        name = _norm_key(o.get("name", ""))
        if not name:
            continue
        if "trust" in o and o["trust"] is not None:
            out["org_trust"][name] = float(o["trust"])
        for a in (o.get("aliases") or []):
            ak = _norm_key(a)
            if ak:
                out["org_alias"][ak] = name

    return out


def load_config(base_dir: Path) -> Config:
    """
    Carica tutte le configurazioni dal folder configs/.
    """
    cfg_dir = base_dir / "configs"
    sources = _load_yaml(cfg_dir / "sources.yaml")
    trust = _load_yaml(cfg_dir / "trust.yaml")
    taxonomy = _load_yaml(cfg_dir / "taxonomy.yaml")
    ranking = _load_yaml(cfg_dir / "ranking.yaml")

    # whitelist editoriale opzionale (non rompe se assente)
    editorial_path = cfg_dir / "editorial_whitelist.yaml"
    if editorial_path.exists():
        editorial = _load_yaml(editorial_path)
        trust = _merge_editorial_whitelist(trust, editorial)

    return Config(
        sources=sources,
        trust=trust,
        taxonomy=taxonomy,
        ranking=ranking,
    )


def topic_list(taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ritorna lista topics con keyword."""
    return taxonomy.get("topics", []) or []

import os

def get_database_url() -> str:
    """
    Ordine:
    1) Streamlit secrets (quando gira in cloud)
    2) env var DATABASE_URL (quando gira altrove)
    """
    try:
        import streamlit as st  # lazy import
        if "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])
    except Exception:
        pass

    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL non configurata (secrets o env).")
    return url

