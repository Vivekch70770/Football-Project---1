"""Small reusable Streamlit UI helpers (theme CSS, stat cards, badges)."""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
section[data-testid="stSidebar"] {background: #0e1a14; border-right: 1px solid #1f3a2b;}
.vv-brand {font-size: 1.5rem; font-weight: 800; letter-spacing: .5px; color: #e8f5ec;}
.vv-brand span {color: #22c55e;}
.vv-sub {color: #7fa38d; font-size: .8rem; margin-bottom: 1rem;}
div[data-testid="stMetric"] {background: linear-gradient(160deg,#13211a,#0f1b15); border: 1px solid #1f3a2b;
  border-left: 4px solid #22c55e; padding: .7rem .9rem; border-radius: 10px;}
div[data-testid="stMetricLabel"] p {color: #7fa38d; font-size: .78rem;}
.vv-badge {display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: .72rem; font-weight: 700; margin-right: 6px;}
.vv-measured {background:#14532d; color:#bbf7d0;} .vv-estimated {background:#713f12; color:#fde68a;}
.vv-uncertain {background:#7f1d1d; color:#fecaca;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown('<div class="vv-brand">VAR<span>Vision</span> AI</div>'
                '<div class="vv-sub">Football match analysis · review-assistance prototype</div>',
                unsafe_allow_html=True)


def stat_cards(items) -> None:
    """items: list of (label, value[, help])."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.metric(it[0], it[1], help=it[2] if len(it) > 2 else None)


def badge(kind: str, text: str | None = None) -> str:
    label = text or kind.capitalize()
    return f'<span class="vv-badge vv-{kind}">{label}</span>'


def data_legend() -> None:
    st.markdown(badge("measured", "Measured: detector output") + badge("estimated", "Estimated: derived / heuristic")
                + badge("uncertain", "Uncertain: needs human review"), unsafe_allow_html=True)


def empty_state(title: str, hint: str) -> None:
    st.info(f"**{title}**  \n{hint}")
