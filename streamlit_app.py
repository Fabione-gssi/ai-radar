from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from radar.config import load_config, get_database_url
from radar.db import NewsDB
from radar.pipeline import enrich_and_store
from radar.processing.autotag import suggest_tags

import os
import time
import streamlit as st

RUN_TOKEN = st.secrets.get("RUN_TOKEN", "")

# Trigger pipeline via URL: ?run=1&token=...
qp = st.query_params
if qp.get("run") == "1" and RUN_TOKEN and qp.get("token") == RUN_TOKEN:
    from radar.pipeline import enrich_and_store
    start = time.time()
    enrich_and_store()
    st.success(f"Pipeline eseguita in {time.time() - start:.1f}s")
    st.stop()


BASE_DIR = Path(__file__).parent
DB_URL = get_database_url()
CFG = load_config(BASE_DIR)

st.set_page_config(page_title="AI News Radar", layout="wide")
st.title("AI News Radar")
st.caption("Reliable (qualità alta) + Scout (early signals). Precision-first + discovery controllato.")


DEFAULTS = {
    "ui_mode": "Base",
    "min_priority": 0.40,
    "status": "all",
    "lang": "all",
    "topic": "all",
    "content_type": "all",
    "lane": "all",
    "source_kind": "all",
    "search": "",
    "tag_filter": [],
    "limit": 200,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _load_saved_view_into_session(db: NewsDB, view_name: str) -> None:
    if not view_name:
        return
    payload = db.get_saved_view(view_name)
    if not payload:
        return
    for k in DEFAULTS.keys():
        if k in payload:
            st.session_state[k] = payload[k]


def _normalize_new_tags(raw: str) -> List[str]:
    return [x.strip().lower() for x in (raw or "").split(",") if x.strip()]


def _why(row: pd.Series) -> List[str]:
    why = []
    if float(row.get("source_trust_score", 0.0)) >= 0.90:
        why.append("Fonte top-tier / comunicazione ufficiale")
    if str(row.get("content_type")) in {"release", "tool"} and float(row.get("actionability_score", 0.0)) >= 0.55:
        why.append("Subito testabile (repo/model/API)")
    if str(row.get("content_type")) == "research":
        why.append("Ricerca: potenziale impatto tecnico (valuta novelty + benchmark)")
    if float(row.get("novelty_score", 0.0)) >= 0.70:
        why.append("Alta novelty rispetto al tuo storico recente")
    if float(row.get("breakout_signal", 0.0)) >= 0.70:
        why.append("Segnale scout (breakout)")
    return why[:4] if why else ["Segnale coerente con i filtri attuali"]


def _action(row: pd.Series) -> str:
    ct = str(row.get("content_type", "news"))
    u = str(row.get("url", ""))
    if ct in {"tool", "release"}:
        if "github.com" in u:
            return "Apri il repo: verifica README, changelog e quickstart. Se utile, pianifica un test rapido."
        if "huggingface.co" in u:
            return "Apri la model card: controlla licenza, benchmark e requisiti. Valuta un test con prompt standard."
        return "Apri la fonte e cerca: link a demo/codice/weights. Se c’è, pianifica test."
    if ct == "research":
        return "Leggi abstract+figures: identifica claim misurabili. Cerca codice/dataset e prova a replicare un baseline."
    if ct == "industry":
        return "Estrarre pattern (KPI/ROI, contesto). Valuta trasferibilità nel tuo dominio e prepara 2-3 domande per un pilot."
    return "Leggi e valuta se richiede follow-up (tool, paper, release)."


def _tag_suggestions(row: pd.Series, max_tags: int = 5) -> List[Dict[str, str]]:
    blob = f"{row.get('title','')} {row.get('snippet','')}"
    return suggest_tags(blob, CFG.taxonomy, max_tags=max_tags)


# Sidebar
with st.sidebar:
    st.header("Pipeline")
    if st.button("Aggiorna ora (run pipeline)"):
        stats = enrich_and_store(BASE_DIR)
        st.success(
            f"Stored: {stats['stored']} | fetch={stats['fetched']} | "
            f"skip_existing={stats.get('skipped_existing',0)} | "
            f"skip_trust={stats.get('skipped_trust',0)} | skip_quality={stats.get('skipped_quality',0)}"
        )
        if stats.get("lane_counts") or stats.get("content_type_counts"):
            st.json({"lane_counts": stats.get("lane_counts", {}), "content_type_counts": stats.get("content_type_counts", {})})

    st.divider()

    # saved views
    db_sv = NewsDB(DB_URL)
    try:
        saved_names = db_sv.list_saved_views()
    finally:
        db_sv.close()

    st.subheader("Saved views")
    selected_view = st.selectbox("Carica view", [""] + saved_names, index=0)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Applica view"):
            db_load = NewsDB(DB_URL)
            try:
                _load_saved_view_into_session(db_load, selected_view)
            finally:
                db_load.close()
            st.rerun()
    with col_b:
        if st.button("Elimina view") and selected_view:
            db_del = NewsDB(DB_URL)
            try:
                db_del.delete_saved_view(selected_view)
            finally:
                db_del.close()
            st.success(f"View '{selected_view}' eliminata")
            st.rerun()

    new_view_name = st.text_input("Nome nuova view")
    if st.button("Salva filtri correnti"):
        payload = {k: st.session_state.get(k) for k in DEFAULTS.keys()}
        db_save = NewsDB(DB_URL)
        try:
            db_save.save_view(new_view_name, payload)
        finally:
            db_save.close()
        st.success(f"View salvata: {new_view_name}")

    st.divider()
    st.header("Filtri")
    st.radio("Modalità", ["Base", "Avanzata"], key="ui_mode", horizontal=True)

    if st.session_state["ui_mode"] == "Base":
        preset = st.selectbox("Soglia (preset)", ["Media", "Alta", "Tutto"], index=0)
        if preset == "Alta":
            st.session_state["min_priority"] = 0.55
        elif preset == "Tutto":
            st.session_state["min_priority"] = 0.25
        else:
            st.session_state["min_priority"] = 0.40

        focus = st.multiselect("Focus contenuti", ["release", "tool", "research", "industry", "news"], default=["release", "tool", "industry"])
        if len(focus) == 1:
            st.session_state["content_type"] = focus[0]
        else:
            st.session_state["content_type"] = "all"

        st.selectbox("Lane", ["all", "reliable", "scout"], key="lane")
        st.selectbox("Tipo fonte", ["all", "institutional", "creator"], key="source_kind")
        st.selectbox("Lingua", ["all", "it", "en", "unknown"], key="lang")
        st.text_input("Cerca", key="search")
        st.slider("Max risultati", 50, 500, key="limit", step=10)

    else:
        st.slider("Priority minima", 0.0, 1.0, key="min_priority", step=0.01)
        st.selectbox("Lane", ["all", "reliable", "scout"], key="lane")
        st.selectbox("Tipo contenuto", ["all", "tool", "research", "release", "industry", "news"], key="content_type")
        st.selectbox("Tipo fonte", ["all", "institutional", "creator"], key="source_kind")
        st.selectbox("Status", ["all", "new", "reviewed", "test-planned", "discarded"], key="status")
        st.selectbox("Lingua", ["all", "it", "en", "unknown"], key="lang")
        st.text_input("Topic (esatto o 'all')", key="topic")
        st.text_input("Cerca (titolo/testo)", key="search")
        st.slider("Numero massimo risultati", 50, 500, key="limit", step=10)

    # tags
    db_tags = NewsDB(DB_URL)
    try:
        all_tags = db_tags.list_tags()
        tag_counts = db_tags.tag_counts(limit=30)
    finally:
        db_tags.close()

    st.multiselect("Filtra per tag (OR)", all_tags, key="tag_filter")

    with st.expander("Guida tag (playbook)", expanded=False):
        playbook = (CFG.taxonomy or {}).get("tag_playbook", {}) or {}
        if not playbook:
            st.info("Nessun playbook tag configurato in taxonomy.yaml (tag_playbook).")
        else:
            for tag, info in playbook.items():
                st.markdown(f"**{tag}** — {info.get('desc','')}")
                ex = info.get("examples", [])
                if ex:
                    st.caption("Esempi: " + ", ".join(ex))

    with st.expander("Top tag usati", expanded=False):
        if tag_counts:
            st.write({t: n for t, n in tag_counts})
        else:
            st.info("Nessun tag assegnato ancora.")


# Query
db = NewsDB(DB_URL)
try:
    df = db.query_items(
        min_priority=float(st.session_state["min_priority"]),
        status=st.session_state["status"],
        topic=(st.session_state["topic"] if st.session_state["topic"] else "all"),
        content_type=st.session_state["content_type"],
        lane=st.session_state["lane"],
        lang=st.session_state["lang"],
        source_kind=st.session_state["source_kind"],
        tag_any=st.session_state.get("tag_filter", []),
        search=(st.session_state["search"] if st.session_state["search"] else None),
        limit=int(st.session_state["limit"]),
    )
    tag_map = db.get_tags_map(df["url"].tolist()) if not df.empty else {}
finally:
    db.close()

st.write(f"Risultati: {len(df)}")
if df.empty:
    st.info("Nessuna news con i filtri correnti.")
    st.stop()

# --- selezione + bulk tagging ---
st.subheader("Selezione rapida + tagging bulk (con suggerimenti)")
sel_df = df[["title", "source", "content_type", "lane", "priority_score", "url"]].copy()
sel_df.insert(0, "select", False)

edited = st.data_editor(
    sel_df,
    use_container_width=True,
    hide_index=True,
    disabled=["title", "source", "content_type", "lane", "priority_score", "url"],
)
selected_urls = edited.loc[edited["select"] == True, "url"].tolist()

with st.form("bulk_tag_form"):
    db_tags2 = NewsDB(DB_URL)
    try:
        existing_tags = db_tags2.list_tags()
    finally:
        db_tags2.close()

    chosen_existing = st.multiselect("Tag esistenti", existing_tags)
    new_tags_input = st.text_input("Nuovi tag (separati da virgola)")
    remove_mode = st.checkbox("Rimuovi invece di assegnare")
    submitted = st.form_submit_button("Applica ai selezionati")

if submitted:
    tags_to_apply = list(dict.fromkeys(chosen_existing + _normalize_new_tags(new_tags_input)))
    if not selected_urls:
        st.warning("Seleziona almeno una riga.")
    elif not tags_to_apply:
        st.warning("Specifica almeno un tag.")
    else:
        db_bulk = NewsDB(DB_URL)
        try:
            if remove_mode:
                n = db_bulk.remove_tags_bulk(selected_urls, tags_to_apply)
                st.success(f"Tag rimossi: {n} operazioni su {len(selected_urls)} news.")
            else:
                n = db_bulk.assign_tags_bulk(selected_urls, tags_to_apply)
                st.success(f"Tag assegnati: {n} nuovi link tag su {len(selected_urls)} news.")
        finally:
            db_bulk.close()
        st.rerun()

st.divider()

tab_feed, tab_digest = st.tabs(["Feed", "Digest giornaliero"])

def render_cards(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("Nessun elemento in questa vista.")
        return

    for i, row in frame.iterrows():
        tags = tag_map.get(row["url"], [])
        with st.container(border=True):
            st.subheader(row["title"])
            st.write(row["snippet"])
            st.markdown(f"- **URL:** {row['url']}")
            st.markdown(
                f"- **Fonte:** {row['source']} | **Org:** {row['author_org']} | "
                f"**Lang:** {row['lang']} | **Type:** {row['content_type']} | "
                f"**Topic:** {row['topic']} | **Lane:** {row['lane']}"
            )

            st.markdown("- **Perché rilevante:** " + "; ".join(_why(row)))
            st.markdown(f"- **Azione suggerita:** {_action(row)}")

            if tags:
                st.markdown("- **Tag:** " + " ".join([f"`{t}`" for t in tags]))

            # suggerimenti tag
            with st.expander("Suggerimenti tag", expanded=False):
                sugg = _tag_suggestions(row, max_tags=5)
                if not sugg:
                    st.caption("Nessun suggerimento forte per questo item.")
                else:
                    cols = st.columns(min(5, len(sugg)))
                    for j, s in enumerate(sugg):
                        tag = s["tag"]
                        reason = s.get("reason", "")
                        with cols[j]:
                            if st.button(f"+ {tag}", key=f"addtag_{row['url']}_{tag}"):
                                db_one = NewsDB(DB_URL)
                                try:
                                    db_one.assign_tags_bulk([row["url"]], [tag])
                                finally:
                                    db_one.close()
                                st.rerun()
                        st.caption(reason)

            st.markdown(
                f"- **Scores** → priority: **{row['priority_score']:.2f}** | "
                f"quality: {row['quality_score']:.2f} | "
                f"breakout: {row['breakout_signal']:.2f} | "
                f"trust: {row['source_trust_score']:.2f} | novelty: {row['novelty_score']:.2f} | "
                f"relevance: {row['relevance_score']:.2f} | actionable: {row['actionability_score']:.2f} | "
                f"recency: {row['recency_score']:.2f}"
            )


with tab_feed:
    tab_rel, tab_scout = st.tabs(["Reliable feed", "Scout feed"])
    with tab_rel:
        render_cards(df[df["lane"] == "reliable"])
    with tab_scout:
        render_cards(df[df["lane"] == "scout"])


with tab_digest:
    st.subheader("Digest giornaliero commentato (export Markdown)")
    today = datetime.now().strftime("%Y-%m-%d")
    top_n = st.slider("Numero item nel digest", 5, 40, 15, step=1)
    # usa ordinamento già in df (priority desc)
    digest_df = df.head(int(top_n)).copy()

    def _digest_line(r: pd.Series) -> str:
        title = str(r.get("title", "")).strip()
        url = str(r.get("url", "")).strip()
        why = "; ".join(_why(r))
        act = _action(r)
        return f"- [{title}]({url}) — {why}. Azione: {act}"

    lines = [f"# AI Radar Digest — {today}", ""]
    for ct in ["release", "tool", "research", "industry", "news"]:
        sub = digest_df[digest_df["content_type"] == ct]
        if sub.empty:
            continue
        lines.append(f"## {ct}")
        for _, r in sub.iterrows():
            lines.append(_digest_line(r))
        lines.append("")

    md = "\n".join(lines).strip() + "\n"
    st.text_area("Preview", md, height=320)

    out_path = BASE_DIR / "data" / f"digest_{today}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    st.download_button("Download digest .md", data=md, file_name=out_path.name, mime="text/markdown")
