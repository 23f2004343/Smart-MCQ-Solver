# PROJECT_ROADMAP.md

## Phase 1: Data Integration & RAG Baseline (Milestone 1)
- **Objective:** Establish the dynamic path loader for `data/trn.csv` and `data/ts.csv`.
- **Task:** Build the text chunking and vector indexing pipeline. 
- **Compute:** Force CPU overrides for local sanity checks.
- **Git Strategy:** Commit all exploratory work and the baseline pipeline to the `milestone-1` branch. Push to the remote repository.

## Phase 2: The Three-Model Architecture (Milestones 2 & 3)
- **Objective:** Develop the core model files.
- **Task:** 1. Draft `model_scratch.ipynb` (defining custom `nn.Module` classes).
  2. Draft `model_pretrained.ipynb` (integrating local Hugging Face transformers).
  3. Draft `model_custom.ipynb` (experimental architecture).
- **Tracking:** Integrate `wandb.init()` boilerplate into all training loops.
- **Git Strategy:** Commit progress to respective milestone branches. Do not pollute the final notebook files with Q&A blocks.

## Phase 3: Cloud Execution & Optimization
- **Objective:** Secure a Kaggle leaderboard score > 0.73.
- **Task:** Push the models to Kaggle via `kaggle kernels push`. Train using cloud GPUs, fetch the logs, and evaluate W&B metrics to tune hyperparameters.
- **Git Strategy:** Once the best-performing models are finalized, clean the code and commit *only* the polished `model_*.ipynb` files to the `main` branch. Ensure `README.md` and `requirements.txt` are updated for potential bonus deployment.