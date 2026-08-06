"""
Smart MCQ Solver — Streamlit App (Model 3: LightGBM + TF-IDF + MiniLM)

Run locally:
    cd streamlit_deploy && streamlit run app.py

Pipeline
--------
1. @st.cache_resource loaders initialise once: LGB model, TF-IDF bundle, MiniLM encoder.
2. On submit: extract 21 features per option (lexical + semantic + structural + relative rank).
3. lgb_model.predict_proba() scores each option; top-1 shown as the prediction.

No external LLM API calls — all inference runs locally.
"""

import os
import re
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart MCQ Solver — Model 3 Ensemble",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — premium dark-card theme ───────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.3rem 0;
        line-height: 1.2;
    }
    .hero-sub { color: rgba(255,255,255,0.55); font-size: 0.95rem; margin: 0; }
    .winner-banner {
        background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(96,165,250,0.15));
        border: 1px solid rgba(52,211,153,0.4);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .winner-label {
        font-size: 0.8rem; letter-spacing: 0.12em;
        text-transform: uppercase; color: rgba(52,211,153,0.85); margin-bottom: 0.4rem;
    }
    .winner-option { font-size: 2rem; font-weight: 700; color: #34d399; margin-bottom: 0.3rem; }
    .winner-text { font-size: 1rem; color: rgba(255,255,255,0.78); line-height: 1.55; }
    .score-row {
        display: flex; align-items: center; gap: 0.8rem;
        margin: 0.45rem 0; background: rgba(255,255,255,0.04);
        border-radius: 8px; padding: 0.5rem 0.9rem;
    }
    .score-badge { font-weight: 600; font-size: 1.05rem; min-width: 1.8rem; color: #e2e8f0; }
    .score-bar-track {
        flex: 1; height: 8px; background: rgba(255,255,255,0.08);
        border-radius: 4px; overflow: hidden;
    }
    .score-bar-fill { height: 100%; border-radius: 4px; }
    .score-num { font-size: 0.82rem; color: rgba(255,255,255,0.55); min-width: 3.5rem; text-align: right; }
    label { color: rgba(255,255,255,0.8) !important; font-weight: 500; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white; border: none; border-radius: 10px;
        padding: 0.7rem 1.4rem; font-size: 1rem; font-weight: 600;
        letter-spacing: 0.02em; cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _asset(fname):
    return os.path.join(BASE_DIR, fname)

def clean(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())

def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Cached resource loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LightGBM model…")
def load_lgb():
    return joblib.load(_asset("lgb_model.joblib"))


@st.cache_resource(show_spinner="Loading TF-IDF vectorizers…")
def load_tfidf():
    bundle = joblib.load(_asset("tfidf_vectorizer.joblib"))
    return bundle["word"], bundle["char"]


@st.cache_resource(show_spinner="Loading MiniLM sentence encoder…")
def load_minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Reading config…")
def load_config():
    with open(_asset("config.json")) as f:
        return json.load(f)


# Initialise all resources at startup (cached after first call)
lgb_model              = load_lgb()
word_tfidf, char_tfidf = load_tfidf()
minilm                 = load_minilm()
cfg                    = load_config()
FEAT_COLS              = cfg["feature_cols"]
RANK_FEATS             = cfg["rank_feats"]
OPTION_COLS            = cfg["option_cols"]


# ── Feature extraction ─────────────────────────────────────────────────────────
def build_feature_matrix(prompt, options):
    """
    Returns ndarray shape (5, 21) matching FEAT_COLS order.
    Base (11): word_cos, char_cos, jaccard, word_intersect, minilm_cos, structural ratios
    Relative (10): {feat}_diff and {feat}_rank per RANK_FEATS
    """
    q_clean  = clean(prompt)
    q_tokens = q_clean.split()
    q_words  = set(q_tokens)
    p_len    = len(q_clean)
    p_wc     = len(q_tokens)

    wq_vec = word_tfidf.transform([q_clean])
    cq_vec = char_tfidf.transform([q_clean])

    # One batch forward pass: query + 5 options (faster than individual encodes)
    texts  = [q_clean] + [clean(o) for o in options]
    embeds = minilm.encode(texts, normalize_embeddings=True, batch_size=6,
                           show_progress_bar=False)
    q_emb  = embeds[0]
    o_embs = embeds[1:]

    rows = []
    for j, opt_raw in enumerate(options):
        opt_txt   = clean(opt_raw) if opt_raw.strip() else ""
        opt_words = set(opt_txt.split())
        o_len     = len(opt_txt)
        o_wc      = len(opt_txt.split())

        wo_vec = word_tfidf.transform([opt_txt])
        co_vec = char_tfidf.transform([opt_txt])

        rows.append({
            "word_cos":       float(cosine_similarity(wq_vec, wo_vec)[0][0]),
            "char_cos":       float(cosine_similarity(cq_vec, co_vec)[0][0]),
            "jaccard":        jaccard(q_words, opt_words),
            "word_intersect": len(q_words & opt_words) / max(len(opt_words), 1),
            "minilm_cos":     float(np.dot(q_emb, o_embs[j])),
            "opt_len":        o_len,
            "prompt_len":     p_len,
            "len_ratio":      o_len / (p_len + 1),
            "opt_wc":         o_wc,
            "prompt_wc":      p_wc,
            "wc_ratio":       o_wc / (p_wc + 1),
        })

    feat_df = pd.DataFrame(rows)
    for feat in RANK_FEATS:
        feat_df[f"{feat}_diff"] = feat_df[feat] - feat_df[feat].mean()
        feat_df[f"{feat}_rank"] = feat_df[feat].rank(ascending=False, method="min")

    return feat_df[FEAT_COLS].values.astype(np.float32)


# ── Score-bar HTML helper ──────────────────────────────────────────────────────
_PALETTE = ["#34d399", "#60a5fa", "#a78bfa", "#f59e0b", "#f87171"]
_MEDALS  = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

def score_bar_html(label, text, score, max_score, rank):
    pct     = int(score / (max_score + 1e-9) * 100)
    color   = _PALETTE[rank]
    medal   = _MEDALS[rank]
    preview = (text[:72] + "\u2026") if len(text) > 72 else text
    # Zero-indented HTML — any leading whitespace on a line triggers Markdown
    # code-block parsing in st.markdown, showing raw tags instead of rendered HTML.
    return (
        f'<div class="score-row">'
        f'<span class="score-badge">{medal} <b>{label}</b></span>'
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
        f'<span class="score-num">{score:.4f}</span>'
        f'</div>'
        f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.5);'
        f'padding:0.1rem 0.9rem 0.5rem 1rem;line-height:1.4;">'
        f'{preview}</div>'
    )


# ── Sample questions ───────────────────────────────────────────────────────────
SAMPLES = {
    "— Select a sample —": None,
    "Cell Biology": {
        "q": "What is the powerhouse of the cell?",
        "A": "The nucleus, which controls cell activity and stores DNA",
        "B": "The mitochondria, which produces ATP via cellular respiration",
        "C": "The ribosome, which synthesises proteins from mRNA templates",
        "D": "The cell membrane, which regulates ion transport and signalling",
        "E": "The endoplasmic reticulum, which folds and transports proteins",
    },
    "Astronomy": {
        "q": "Which planet is closest to the Sun?",
        "A": "Venus",
        "B": "Earth",
        "C": "Mercury",
        "D": "Mars",
        "E": "Jupiter",
    },
    "Philosophy (Heidegger)": {
        "q": "What is Heidegger's view on the relationship between time and human existence?",
        "A": "Humans exist in an infinite time continuum with no defined beginning or end.",
        "B": "Humans do not exist inside time — they ARE time; the past is a present awareness of having been.",
        "C": "Heidegger rejects the existence of time and its effect on consciousness entirely.",
        "D": "Time and existence are cyclical; the future is predetermined and free will is illusory.",
        "E": "Time is an illusion; humans exist outside it, guided by a higher power.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:0.5rem 0 1.2rem">
          <div style="font-size:2.8rem;margin-bottom:0.3rem">🧠</div>
          <div style="font-weight:700;font-size:1.1rem;color:#e2e8f0">Smart MCQ Solver</div>
          <div style="font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:0.2rem">
            Model 3 · LightGBM Ensemble
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 📚 Sample Questions")
    sample_key = st.selectbox(
        "Load a demo question",
        options=list(SAMPLES.keys()),
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        """
        **Pipeline**
        1. Word + Char TF-IDF cosine similarity
        2. Jaccard & vocabulary overlap
        3. MiniLM-L6-v2 semantic embedding
        4. Relative rank features (within-question)
        5. LightGBM `predict_proba` scoring

        **Features:** 21 total (11 base + 10 rank)
        **Val accuracy:** 96.4% (2-fold GroupKFold)
        **No external LLM API calls.**
        """,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="card">
      <p class="hero-title">Smart MCQ Solver</p>
      <p class="hero-sub">
        Model 3 — LightGBM + TF-IDF + MiniLM ensemble &nbsp;·&nbsp;
        Enter a question and up to 5 options to get a ranked prediction.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

_s       = SAMPLES.get(sample_key)
_default = lambda k: _s[k] if _s and k in _s else ""

col_q, col_opts = st.columns([3, 2], gap="large")
with col_q:
    st.markdown("#### 📝 Question")
    question = st.text_area(
        "Question",
        value=_default("q"),
        height=130,
        placeholder="Enter your MCQ question here…",
        label_visibility="collapsed",
    )

with col_opts:
    st.markdown("#### 🔤 Answer Options")
    opt_a = st.text_input("Option A", value=_default("A"), placeholder="Answer A…")
    opt_b = st.text_input("Option B", value=_default("B"), placeholder="Answer B…")

col_c, col_d, col_e = st.columns(3)
with col_c:
    opt_c = st.text_input("Option C", value=_default("C"), placeholder="Answer C…")
with col_d:
    opt_d = st.text_input("Option D", value=_default("D"), placeholder="Answer D…")
with col_e:
    opt_e = st.text_input("Option E (optional)", value=_default("E"), placeholder="Answer E…")

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Correct Answer", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
if predict_btn:
    options_raw   = [opt_a, opt_b, opt_c, opt_d, opt_e]
    option_labels = list("ABCDE")

    if not question.strip():
        st.error("⚠️ Please enter a question before predicting.")
        st.stop()

    filled = [(lbl, txt) for lbl, txt in zip(option_labels, options_raw) if txt.strip()]
    if len(filled) < 2:
        st.error("⚠️ Please fill in at least 2 answer options.")
        st.stop()

    with st.spinner("Computing features and scoring options…"):
        try:
            X      = build_feature_matrix(question, options_raw)
            scores = lgb_model.predict_proba(X)[:, 1]   # P(relevant) per option
        except Exception as exc:
            st.error(f"❌ Inference failed: {exc}")
            st.stop()

    # Rank descending by score
    ranked = sorted(
        zip(option_labels, options_raw, scores),
        key=lambda x: -x[2],
    )
    top_lbl, top_txt, top_score = ranked[0]
    max_score = ranked[0][2]

    # ── Winner banner ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="winner-banner">
          <p class="winner-label">🏆 Top Prediction</p>
          <p class="winner-option">Option {top_lbl}</p>
          <p class="winner-text">{top_txt}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metric tiles ──────────────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Top Option", f"Option {top_lbl}")
    mc2.metric("Confidence Score", f"{top_score:.4f}")
    gap = top_score - ranked[1][2] if len(ranked) > 1 else 0.0
    mc3.metric("Gap to 2nd Place", f"{gap:+.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ranked scoreboard + bar chart ─────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown("#### 📊 Full Ranking")
        bars_html = "".join(
            score_bar_html(lbl, txt, sc, max_score, rank_i)
            for rank_i, (lbl, txt, sc) in enumerate(ranked)
        )
        # Bubble card: glassmorphism background + gradient glow border
        bubble_style = (
            "background:rgba(255,255,255,0.05);"
            "border:1.5px solid rgba(167,139,250,0.45);"
            "border-radius:18px;"
            "padding:1.1rem 1.3rem;"
            "box-shadow:0 0 22px rgba(124,58,237,0.25),0 0 6px rgba(96,165,250,0.15);"
            "backdrop-filter:blur(14px);"
        )
        st.markdown(
            f'<div style="{bubble_style}">{bars_html}</div>',
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("#### 📈 Score Chart")
        chart_df = pd.DataFrame(
            {"Option": [lbl for lbl, _, _ in ranked],
             "Score":  [sc  for _, _,  sc in ranked]},
        ).set_index("Option")
        st.bar_chart(chart_df, use_container_width=True)

    # ── Feature breakdown expander ────────────────────────────────────────────
    with st.expander("🔬 Feature Breakdown (per option)", expanded=False):
        st.markdown(
            "Raw feature values before relative-ranking augmentation. "
            "`minilm_cos` is computed live using the MiniLM-L6-v2 sentence encoder."
        )
        feat_rows = []
        for lbl, opt_raw in zip(option_labels, options_raw):
            if not opt_raw.strip():
                continue
            q_c   = clean(question)
            o_c   = clean(opt_raw)
            qw    = set(q_c.split())
            ow    = set(o_c.split())
            wq    = word_tfidf.transform([q_c])
            wo    = word_tfidf.transform([o_c])
            cq_v  = char_tfidf.transform([q_c])
            co_v  = char_tfidf.transform([o_c])
            feat_rows.append({
                "Option":         lbl,
                "word_cos":       round(float(cosine_similarity(wq, wo)[0][0]), 4),
                "char_cos":       round(float(cosine_similarity(cq_v, co_v)[0][0]), 4),
                "jaccard":        round(jaccard(qw, ow), 4),
                "word_intersect": round(len(qw & ow) / max(len(ow), 1), 4),
                "opt_len":        len(o_c),
                "opt_wc":         len(o_c.split()),
                "len_ratio":      round(len(o_c) / (len(q_c) + 1), 4),
            })
        if feat_rows:
            st.dataframe(
                pd.DataFrame(feat_rows).set_index("Option"),
                use_container_width=True,
            )

    # ── Viva defence note ─────────────────────────────────────────────────────
    st.info(
        "**Viva Defence** — The model scores each option independently: "
        "high `minilm_cos` means the MiniLM sentence embedding of the option is "
        "semantically close to the question; high `word_cos` reflects TF-IDF lexical overlap; "
        "`_rank` features reveal which option is *relatively* strongest within the question, "
        "making the ranker scale-invariant across questions of different difficulty."
    )
