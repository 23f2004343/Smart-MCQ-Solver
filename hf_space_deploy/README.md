---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
hardware: cpu-basic
short_description: LightGBM + TF-IDF + MiniLM MCQ solver
---

# Smart MCQ Solver

An end-to-end MCQ ranking pipeline combining lexical and semantic features via a LightGBM ensemble.

## Features
- **Word & Char TF-IDF** cosine similarity (trained on the competition corpus)
- **MiniLM-L6-v2** semantic embeddings for live inference
- **Relative ranking** features (diff-from-mean, within-question rank)
- **LightGBM** binary ranker selects the best option(s)

## No External LLM APIs
All inference runs inside this Space — no OpenAI, Gemini, or any hosted LLM.

## Dataset
Trained on the [Smart MCQ Solver Challenge](https://www.kaggle.com/competitions/smart-mcq-solver-challenge) dataset (2,000 MCQs).
