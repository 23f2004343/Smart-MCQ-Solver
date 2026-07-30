# Smart MCQ Solver

End-to-end MCQ answer ranking project for the Smart MCQ Solver Kaggle challenge. Each question has five options (`A`-`E`), and the final submission predicts exactly three labels in ranked order using MAP@3 as the competition metric.

## Project Setup

1. Create a Python environment with Python 3.10+.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

3. Place Kaggle data in either the competition mount or a local `data/` folder:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

The notebooks also resolve Kaggle paths under `/kaggle/input/competitions/smart-mcq-solver-challenge/`.

## Notebook Execution Guide

Run the final notebooks independently so each model remains isolated for viva review:

| Model | Notebook | Purpose |
|---|---|---|
| Model 1 | `nb/scratch/base_model.ipynb` | Custom PyTorch transformer-style encoder built from scratch. |
| Model 2 | `nb/pretrained/pre_trained.ipynb` | Fine-tuned Hugging Face BERT multiple-choice model. |
| Model 3 | `nb/custom/model3.ipynb` | Lexical (TF-IDF/BM25) and semantic (MiniLM) similarity feature blending with LightGBM/XGBoost ensemble. |

For local checks, keep runs small and CPU-safe. Production training and final validation should be executed on Kaggle GPUs through `kaggle kernels push` using the notebook-specific `kernel-metadata.json` files.

## Final Results

The final qualifying solution achieved **0.74397 MAP@3** (Kaggle Leaderboard Score), crossing the required 0.73 viva cutoff. Model 3 combines lexical (TF-IDF/BM25) and semantic (MiniLM) similarity features blended with a LightGBM/XGBoost ensemble — no external vector-search RAG pipeline is used. All final notebooks generate a `submission.csv` in the Kaggle format:

```csv
ID,Prediction
1,A B C
2,C A D
```

W&B tracking is configured in each final notebook. Credentials must be supplied through Kaggle Secrets or the local `WANDB_API_KEY` environment variable; API keys are not stored in source files.
