# ORIENTATION_SUMMARY.md

## 1. Core Project Philosophy
The 'DL-GenAI-Project' evaluates an end-to-end Deep Learning pipeline focused on Retrieval-Augmented Generation (RAG). The project demands a balance between custom algorithmic architecture, robust experiment tracking, and rigorous software engineering practices. All development must adhere to strict local-inference rules while validating execution in cloud GPU environments.

## 2. Technical Architecture & Pipeline Requirements
- **Core Objective:** Build a functional RAG pipeline capable of context retrieval and generation, processing local data (`trn.csv` / `ts.csv`).
- **The Three-Model Mandate:** 1. A DL model built entirely from scratch (Custom PyTorch/TensorFlow architecture).
  2. A pre-trained DL model (e.g., Hugging Face Transformer) fine-tuned on the dataset.
  3. A custom experimental model (can be an ensemble or advanced ML/DL hybrid).
- **Execution Boundary:** All local testing must fall back to CPU compute to maintain stability, while final training and inference must be executed via Kaggle cloud GPUs.

## 3. Key Success Metrics & Compliance
- **Leaderboard Cutoff:** Achieve a submission score of at least ~0.73 on the Kaggle leaderboard.
- **Experiment Tracking:** Mandatory use of Weights & Biases (W&B) to log all hyperparameters, epochs, and loss metrics.
- **Repository Maturity:** A consistent 3-week commit history across dedicated milestone branches, with a pristine `main` branch reserved for final models.
- **Viva Readiness (Zero-Malpractice):** Absolute comprehension of every line of code. No external LLM APIs (OpenAI, Gemini) are permitted for inference generation.