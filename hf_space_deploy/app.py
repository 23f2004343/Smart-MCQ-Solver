"""
Smart MCQ Solver — Gradio Inference App
Space: 23f2004343/smart-mcq-ensemble

Pipeline:
  1. Load TF-IDF vectorizers + LightGBM model from serialised .joblib files.
  2. Load sentence-transformers/all-MiniLM-L6-v2 for live semantic embedding.
  3. On each query, extract lexical + semantic + structural features for all 5 options.
  4. LightGBM predicts a relevance score per (question, option) pair.
  5. Return options ranked by score with the top pick highlighted.

No external LLM API calls — all inference is local within the Space container.
"""

import os, re, json
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

# ── Startup: Load Artifacts ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading TF-IDF vectorizers...")
tfidf_bundle = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.joblib"))
word_tfidf   = tfidf_bundle["word"]
char_tfidf   = tfidf_bundle["char"]

print("Loading LightGBM model...")
lgb_model = joblib.load(os.path.join(BASE_DIR, "lgb_model.joblib"))

print("Loading config...")
with open(os.path.join(BASE_DIR, "config.json")) as f:
    cfg = json.load(f)
FEAT_COLS   = cfg["feature_cols"]
RANK_FEATS  = cfg["rank_feats"]
OPTION_COLS = cfg["option_cols"]

print("Loading MiniLM sentence encoder (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ All models loaded — app ready.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# ── Feature Extraction (single question) ──────────────────────────────────────
def build_features(prompt: str, options: list[str]) -> np.ndarray:
    """
    Build the feature matrix for one question with 5 options.
    Returns ndarray of shape (5, n_features) matching FEAT_COLS order.
    """
    q_clean = clean(prompt)
    q_tokens = q_clean.split()
    q_words  = set(q_tokens)
    p_len    = len(q_clean)
    p_wc     = len(q_tokens)

    wq_vec = word_tfidf.transform([q_clean])
    cq_vec = char_tfidf.transform([q_clean])

    # Batch MiniLM encoding: query + all 5 options in one forward pass
    texts_to_encode  = [q_clean] + [clean(o) for o in options]
    embeddings       = minilm.encode(texts_to_encode, normalize_embeddings=True, batch_size=6)
    q_emb            = embeddings[0]
    opt_embs         = embeddings[1:]   # shape (5, 384)

    rows = []
    for j, opt_raw in enumerate(options):
        opt_txt   = clean(opt_raw)
        opt_words = set(opt_txt.split())
        o_len     = len(opt_txt)
        o_wc      = len(opt_txt.split())

        wo_vec = word_tfidf.transform([opt_txt])
        co_vec = char_tfidf.transform([opt_txt])

        word_cos       = float(cosine_similarity(wq_vec, wo_vec)[0][0])
        char_cos_val   = float(cosine_similarity(cq_vec, co_vec)[0][0])
        jacc           = jaccard(q_words, opt_words)
        word_int       = len(q_words & opt_words) / max(len(opt_words), 1)
        mlm_sim        = float(np.dot(q_emb, opt_embs[j]))

        rows.append({
            "word_cos":      word_cos,
            "char_cos":      char_cos_val,
            "jaccard":       jacc,
            "word_intersect":word_int,
            "minilm_cos":    mlm_sim,
            "opt_len":       o_len,
            "prompt_len":    p_len,
            "len_ratio":     o_len / (p_len + 1),
            "opt_wc":        o_wc,
            "prompt_wc":     p_wc,
            "wc_ratio":      o_wc / (p_wc + 1),
        })

    feat_df = pd.DataFrame(rows)

    # Add relative-ranking features (diff-from-mean and descending rank)
    for feat in RANK_FEATS:
        if feat in feat_df.columns:
            feat_df[f"{feat}_diff"] = feat_df[feat] - feat_df[feat].mean()
            feat_df[f"{feat}_rank"] = feat_df[feat].rank(ascending=False, method="min")
        else:
            feat_df[f"{feat}_diff"] = 0.0
            feat_df[f"{feat}_rank"] = 3.0

    return feat_df[FEAT_COLS].values.astype(np.float32)

# ── Inference Function ─────────────────────────────────────────────────────────
def predict_mcq(prompt: str, opt_a: str, opt_b: str, opt_c: str, opt_d: str, opt_e: str):
    """Main Gradio handler: score all 5 options and return ranked result."""
    if not prompt.strip():
        return "⚠️ Please enter a question prompt.", ""

    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    labels  = list("ABCDE")

    # Validate at least 2 non-empty options
    filled = [(lbl, opt) for lbl, opt in zip(labels, options) if opt.strip()]
    if len(filled) < 2:
        return "⚠️ Please fill in at least 2 answer options.", ""

    # Pad missing options with empty string (scored as 0 similarity)
    options_padded = [o if o.strip() else "" for o in options]

    try:
        X = build_features(prompt, options_padded)
        scores = lgb_model.predict_proba(X)[:, 1]   # P(relevant)
    except Exception as e:
        return f"❌ Inference error: {e}", ""

    # Rank by score descending
    ranked = sorted(zip(labels, options_padded, scores), key=lambda x: -x[2])
    top_label, top_text, top_score = ranked[0]

    # Build formatted result
    answer_md = f"## 🏆 Top Answer: **Option {top_label}**\n\n"
    answer_md += f"> {top_text}\n\n"
    answer_md += f"*Confidence score: `{top_score:.4f}`*\n"

    ranking_md = "### Full Ranking\n\n| Rank | Option | Score | Text Preview |\n|------|--------|-------|--------------|\n"
    for rank_i, (lbl, txt, sc) in enumerate(ranked, 1):
        preview = (txt[:60] + "…") if len(txt) > 60 else txt
        medal   = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank_i - 1]
        ranking_md += f"| {medal} | **{lbl}** | `{sc:.4f}` | {preview} |\n"

    return answer_md, ranking_md

# ── Gradio UI ──────────────────────────────────────────────────────────────────
HEADER = """
<div style="text-align:center;padding:1.5rem 0 0.5rem">
  <h1 style="font-size:2rem;font-weight:700;margin:0">🧠 Smart MCQ Solver</h1>
  <p style="color:#6b7280;margin:0.4rem 0 0">
    LightGBM + TF-IDF + MiniLM ensemble — no external LLM APIs
  </p>
</div>
"""

EXAMPLES = [
    [
        "What is the powerhouse of the cell?",
        "The nucleus, which controls cell activity",
        "The mitochondria, which produces ATP via cellular respiration",
        "The ribosome, which synthesises proteins",
        "The cell membrane, which regulates ion transport",
        "The endoplasmic reticulum, which folds proteins",
    ],
    [
        "Pick the best answer: Which planet is closest to the Sun?",
        "Venus", "Earth", "Mercury", "Mars", "Jupiter",
    ],
]

with gr.Blocks(title="Smart MCQ Solver", theme=gr.themes.Soft()) as demo:
    gr.HTML(HEADER)

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label="📝 Question / Prompt",
                placeholder="Enter the MCQ question here…",
                lines=3,
                elem_id="prompt_input",
            )
            with gr.Row():
                opt_a = gr.Textbox(label="Option A", placeholder="Answer A…", elem_id="opt_a")
                opt_b = gr.Textbox(label="Option B", placeholder="Answer B…", elem_id="opt_b")
            with gr.Row():
                opt_c = gr.Textbox(label="Option C", placeholder="Answer C…", elem_id="opt_c")
                opt_d = gr.Textbox(label="Option D", placeholder="Answer D…", elem_id="opt_d")
            opt_e = gr.Textbox(label="Option E (optional)", placeholder="Answer E…", elem_id="opt_e")

            submit_btn = gr.Button("🔍 Solve MCQ", variant="primary", elem_id="solve_btn")

        with gr.Column(scale=2):
            answer_out  = gr.Markdown(label="Top Answer", elem_id="answer_output")
            ranking_out = gr.Markdown(label="Full Ranking", elem_id="ranking_output")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_box, opt_a, opt_b, opt_c, opt_d, opt_e],
        label="📚 Try a sample question",
    )

    gr.Markdown(
        """---
        **Model**: LightGBM binary ranker trained on 2,000 MCQs.
        **Features**: Word/Char TF-IDF cosine similarity, Jaccard overlap, relative ranking,
        and MiniLM-L6-v2 semantic similarity (computed live).
        **No internet inference** — all computation runs inside this Space.
        """,
        elem_id="footer_note",
    )

    submit_btn.click(
        fn=predict_mcq,
        inputs=[prompt_box, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[answer_out, ranking_out],
    )

if __name__ == "__main__":
    demo.launch()
