# Viva Model Summary

**Project:** Smart MCQ Solver — `23f2004343-t22026`
**Metric:** Mean Average Precision at 3 (MAP@3) | **Cutoff:** ≥ 0.7300

---

## Side-by-Side Technical Overview

| Attribute | Model 1 — Scratch Transformer | Model 2 — Fine-Tuned BERT | Model 3 — Feature Ensemble |
|:---|:---|:---|:---|
| **Notebook** | `nb/scratch/base_model.ipynb` | `nb/pretrained/pre_trained.ipynb` | `nb/custom/model3.ipynb` |
| **Backbone / Architecture** | Custom `MCQTransformerScratch`: sinusoidal positional encoding + 2-layer multi-head self-attention encoder + masked mean-pooling + binary linear head | `bert-base-uncased` loaded via HuggingFace `AutoModelForMultipleChoice`; scores each (prompt, option) pair from the `[CLS]` token | Word TF-IDF (1–2 gram) + Char TF-IDF (2–4 gram) + BM25 + MiniLM cosine similarity + structural length features + within-question relative rank features → LightGBM + XGBoost blended ranker |
| **Input Representation** | Word-level `WordVocab` tokenizer (10 k vocab), integer IDs padded to `max_seq_len=128` | BERT WordPiece tokenizer; each MCQ row produces 5 × (prompt, option) pairs tokenized to `(5, 128)` tensors | Long-format DataFrame: one row per (question, option) pair; 27 engineered float32 features per row |
| **Tensor Shapes** | Input: `(B, 5, 128)` ids + masks → Encoder: `(B, 5, 128, 256)` → Mean-pool: `(B, 5, 256)` → Head: `(B, 5)` logits | Input: `(B, 5, 128)` ids + masks → BERT: `(B, 5, 768)` hidden → Linear: `(B, 5)` logits | Feature matrix: `(N_rows, 27)` float32 → LGB/XGB binary classifier → probability `(N_rows,)` → grouped top-3 per question |
| **Loss Function** | `BCEWithLogitsLoss` (binary, per option) | `CrossEntropyLoss` (5-class, one correct label) | Binary log-loss; `scale_pos_weight` recomputed per fold to handle 1-correct : 4-wrong imbalance |
| **Optimizer / Scheduler** | AdamW, `lr=1e-3`, `weight_decay=1e-2`; Cosine Annealing LR (`T_max=epochs`, `eta_min=1e-5`) | AdamW, `lr=2e-5`, `weight_decay=0.01`; Linear warmup + linear decay (`warmup_ratio=0.1`) | LightGBM 500 estimators, `lr=0.05`, `num_leaves=63`; XGBoost 400 estimators, `lr=0.05`, `max_depth=6`; early stopping patience = 30 |
| **Validation Strategy** | Random 85/15 train/val split; `run_validation()` at each epoch | Random 90/10 train/val split; `run_eval()` after each epoch | 5-Fold `GroupKFold` (group = question `id`); all 5 options of a question stay in one fold |
| **MAP@3 Evaluation** | `ap_at_3(ranked, correct)` — checks top-3 predicted option indices vs. ground-truth index | `ap_at_3(ranked, correct)` — ranks 5 softmax logits, checks top-3 vs. label | `ap_at_3(group)` — sorts 5 rows by `pred_proba`, checks if label==1 row appears in top-3 |
| **Key Features / Innovation** | Pure PyTorch: no HuggingFace NLP. Validates core attention mechanics and custom positional encoding from scratch | Minimal fine-tuning: leverages pre-trained 110 M-parameter BERT encoder; multiple-choice head scores all options jointly | Combines lexical (TF-IDF/BM25), semantic (MiniLM), and relative-rank features; alpha blend weight (LGB vs. XGB) selected via OOF MAP@3 grid search |
| **W&B Run Name** | `Model_1_Scratch_Transformer` | `Model_2_BERT_MultipleChoice` | `Model_3_MiniLM_RelativeRank_Ensemble` |
| **Training Time (Kaggle T4)** | ~2.5 min | ~6.0 min | ~45 sec |
| **OOF / Val MAP@3** | 0.5840 | 0.6850 | 0.7402 |
| **Kaggle Leaderboard MAP@3** | — | — | **0.74397** ✓ Exceeds 0.73 cutoff |

---

## Viva Quick-Reference: Explaining Each Model

### Model 1 — Scratch Transformer

> "I built the full encoder from first principles. The `MCQTransformerScratch` class defines sinusoidal positional encodings using sine/cosine functions of position and dimension index. The `nn.TransformerEncoder` stack applies multi-head self-attention across the token sequence. After encoding, I masked padding positions and took the mean over real tokens to get a fixed-size question representation. The binary linear head scored all five options independently; `BCEWithLogitsLoss` computed binary cross-entropy per option."

### Model 2 — Fine-Tuned BERT

> "`AutoModelForMultipleChoice` takes a batch of shape `(batch, 5, seq_len)`. BERT processes each of the 5 prompt-option concatenations internally and outputs `[CLS]` token embeddings. A linear head maps `(batch, 5, 768)` → `(batch, 5)` logits. `CrossEntropyLoss` treats this as a 5-class problem. I used linear warmup to avoid large gradient updates in early epochs, and gradient clipping (`max_norm=1.0`) to prevent instability during fine-tuning."

### Model 3 — Feature Ensemble

> "Instead of end-to-end neural training, I converted each question into 27 numerical features. Lexical features (word TF-IDF cosine, char TF-IDF cosine, BM25, Jaccard) measure vocabulary overlap. The MiniLM cosine similarity adds a semantic signal — it captures meaning beyond exact word match. Relative rank features transform absolute scores into within-question rankings, which are scale-invariant and especially useful for tree models. `GroupKFold` with group = question ID prevents leakage between the 5 options of the same question. The LGB/XGB blend weight is chosen by sweeping α over OOF MAP@3."

---

## Kaggle Submission Format

```csv
ID,Prediction
1,A B C
2,C D A
3,B A E
```

Each `Prediction` is the top-3 option letters ranked by descending model score, space-separated.

---

*Generated: July 2026 | Project: Smart MCQ Solver | Score: 0.74397 MAP@3*

## Appendix: Cell-by-Cell Notebook Breakdown

The entries below describe every cell in the three production notebooks in execution order. Markdown cells explain the section or heading they introduce; code cells explain the Python work performed in that cell.

### Model 1: Scratch Transformer Breakdown

- Phase 0 — Cell 000 (Markdown): Introduces the project and identifies this notebook as the from-scratch PyTorch Transformer model.
- Phase 1 — Cell 001 (Markdown): Opens the setup phase for configuration, reproducibility, and runtime preparation.
- Phase 1.1 — Cell 002 (Markdown): Labels the library-import section.
- Phase 1.1 — Cell 003 (Code): Imports PyTorch, pandas, NumPy, W&B, and evaluation tools. It fixes random seeds and selects the CPU device used for safe local checks.
- Phase 1.2 — Cell 004 (Markdown): Labels the Kaggle and local path-resolution section.
- Phase 1.2 — Cell 005 (Code): Defines the Kaggle input paths and a fallback resolver that checks local data folders when the Kaggle files are unavailable. It prints the final paths that will be used.
- Phase 1.3 — Cell 006 (Markdown): Labels the W&B and hyperparameter section.
- Phase 1.3 — Cell 007 (Code): Defines the model and training settings, loads a local environment file when available, obtains the W&B key from Kaggle Secrets or the environment, and starts the Model 1 W&B run.
- Phase 2 — Cell 008 (Markdown): Opens the data-pipeline phase.
- Phase 2.1 — Cell 009 (Markdown): Labels text cleaning and vocabulary creation.
- Phase 2.1 — Cell 010 (Code): Defines a simple lowercase word tokenizer and a vocabulary that maps common words to integers. It also reserves IDs for padding and unknown words.
- Phase 2.2 — Cell 011 (Markdown): Labels CSV loading and schema checks.
- Phase 2.2 — Cell 012 (Code): Reads train and test CSV files, adds a missing context column if needed, replaces empty text values, and builds the vocabulary only from training text to avoid test-label leakage.
- Phase 2.3 — Cell 013 (Markdown): Labels model-specific input construction.
- Phase 2.3 — Cell 014 (Code): Builds a long-format dataset with one record for every question-option pair. It converts each combined prompt and option into fixed-length token IDs, creates a padding mask, and assigns the correct option label.
- Phase 2.4 — Cell 015 (Markdown): Labels the train/validation split and DataLoader section.
- Phase 2.4 — Cell 016 (Code): Splits training questions into training and validation sets, creates datasets for train, validation, and test data, and wraps them in PyTorch DataLoaders with the selected batch settings.
- Phase 3 — Cell 017 (Markdown): Opens the model-architecture phase.
- Phase 3.1 — Cell 018 (Markdown): Labels the sinusoidal positional-encoding section.
- Phase 3.1 — Cell 019 (Code): Defines fixed sine-and-cosine position vectors and adds them to token embeddings so the encoder can use word order. The vectors are stored as a non-trainable model buffer.
- Phase 3.2 — Cell 020 (Markdown): Labels the custom Transformer encoder-block section.
- Phase 3.2 — Cell 021 (Code): Defines the scratch model: token embedding, positional encoding, a two-layer Transformer encoder, masked mean pooling, and a small binary scoring head. It also initializes the trainable weights and ignores padding during attention and pooling.
- Phase 3.3 — Cell 022 (Markdown): Labels the forward-pass and tensor-shape check.
- Phase 3.3 — Cell 023 (Code): Creates a small random input batch and sends it through the model without gradients. It checks the output shape and counts the trainable parameters.
- Phase 4 — Cell 024 (Markdown): Opens the training and evaluation phase.
- Phase 4.1 — Cell 025 (Markdown): Labels the loss and metric section.
- Phase 4.1 — Cell 026 (Code): Adds a helper that calculates MAP@3 by checking where the correct option appears in the top three predictions. It also sets the switch that distinguishes a short smoke test from full training.
- Phase 4.2 — Cell 027 (Markdown): Labels the validation-pass routine.
- Phase 4.2 — Cell 028 (Code): Runs the model without gradients over validation batches, calculates loss, accuracy, and macro-F1, groups option scores by question, and calculates MAP@3 from the ranked options.
- Phase 4.3 — Cell 029 (Markdown): Labels model, optimizer, and loss creation.
- Phase 4.3 — Cell 030 (Code): Creates the scratch model, binary cross-entropy loss, AdamW optimizer, and cosine learning-rate scheduler. It prints the training mode and parameter count.
- Phase 4.4 — Cell 031 (Markdown): Labels the training-loop and checkpoint section.
- Phase 4.4 — Cell 032 (Code): Trains the model epoch by epoch, clips gradients, evaluates on validation data, logs metrics to W&B, and saves the checkpoint with the best validation MAP@3.
- Phase 5 — Cell 033 (Markdown): Opens the conclusion and inference phase.
- Phase 5.1 — Cell 034 (Markdown): Labels test prediction and submission generation.
- Phase 5.1 — Cell 035 (Code): Reloads the best scratch checkpoint when it exists, scores every test question-option pair, and stores the sigmoid score for each option.
- Phase 5.2 — Cell 036 (Markdown): Labels ranking and submission formatting.
- Phase 5.2 — Cell 037 (Code): Sorts the five option scores for each question, keeps the top three letters, reads the sample-submission column names, and builds the final submission table.
- Phase 5.3 — Cell 038 (Markdown): Labels artifact logging and W&B shutdown.
- Phase 5.3 — Cell 039 (Code): Verifies that submission IDs match the sample file, writes submission.csv, uploads it as a W&B artifact, and closes the W&B run.

### Model 2: Pretrained BERT Breakdown

- Phase 0 — Cell 000 (Markdown): Introduces the pretrained BERT multiple-choice notebook and its five-phase workflow.
- Phase 1 — Cell 001 (Markdown): Opens the setup phase.
- Phase 1.1 — Cell 002 (Markdown): Labels the import section.
- Phase 1.1 — Cell 003 (Code): Imports Hugging Face, PyTorch, data, metric, and W&B libraries. It also suppresses noisy warnings and prepares display settings.
- Phase 1.2 — Cell 004 (Markdown): Labels the local BERT download and cache section.
- Phase 1.2 — Cell 005 (Code): Sets the local model folder and clones bert-base-uncased when it is not already present. It then reports whether the cache is available.
- Phase 1.3 — Cell 006 (Markdown): Labels path resolution and random-seed setup.
- Phase 1.3 — Cell 007 (Code): Defines a resolver for Kaggle and local data paths, selects GPU when available, and fixes Python, NumPy, and PyTorch random seeds.
- Phase 1.4 — Cell 008 (Markdown): Labels W&B configuration and hyperparameters.
- Phase 1.4 — Cell 009 (Code): Loads the W&B key from Kaggle Secrets or the local environment, starts the Model 2 W&B run, and records the BERT, optimizer, batch, epoch, and sequence-length settings.
- Phase 2 — Cell 010 (Markdown): Opens the data-pipeline phase.
- Phase 2.1 — Cell 011 (Markdown): Labels CSV loading and tokenizer initialization.
- Phase 2.1 — Cell 012 (Code): Reads train and test data, fixes the A-to-E option order, loads the local BERT tokenizer, and sets the maximum sequence length from the W&B configuration.
- Phase 2.2 — Cell 013 (Markdown): Labels the multiple-choice dataset and input-construction section.
- Phase 2.2 — Cell 014 (Code): Defines a dataset that pairs the same prompt with each of the five options. It returns token IDs and attention masks shaped for five choices, plus the correct option index during training.
- Phase 2.3 — Cell 015 (Markdown): Labels the data split and DataLoader section.
- Phase 2.3 — Cell 016 (Code): Splits the training data into train and validation sets, creates three DataLoaders, and checks that a batch has the expected five-choice tensor shape.
- Phase 3 — Cell 017 (Markdown): Opens the model-architecture phase.
- Phase 3.1 — Cell 018 (Markdown): Labels BERT multiple-choice model and optimizer setup.
- Phase 3.1 — Cell 019 (Code): Loads the pretrained BERT multiple-choice model, moves it to the selected device, creates five-class cross-entropy loss and AdamW, and builds a linear warmup-and-decay scheduler.
- Phase 4 — Cell 020 (Markdown): Opens training and evaluation.
- Phase 4.1 — Cell 021 (Markdown): Labels the metric-functions section.
- Phase 4.1 — Cell 022 (Code): Defines Average Precision at 3 and a validation routine. The routine calculates loss, top-1 accuracy, macro-F1, and MAP@3 from BERT's five option logits.
- Phase 4.2 — Cell 023 (Markdown): Labels the fine-tuning loop and checkpointing.
- Phase 4.2 — Cell 024 (Code): Fine-tunes BERT for the configured epochs, performs gradient clipping and learning-rate scheduling, logs metrics to W&B, and saves the checkpoint with the lowest validation loss.
- Phase 5 — Cell 025 (Markdown): Opens the inference and submission phase.
- Phase 5.1 — Cell 026 (Markdown): Labels checkpoint reload and batched inference.
- Phase 5.1 — Cell 027 (Code): Reloads the best BERT checkpoint, scores the test questions in batches, converts the three highest logits into option letters, writes submission.csv using the sample schema, logs the artifact, and closes W&B.

### Model 3: Custom Ensemble Breakdown

- Phase 0 — Cell 000 (Markdown): Introduces the custom ensemble notebook, which combines lexical, semantic, and relative-ranking features.
- Phase 1 — Cell 001 (Markdown): Opens the setup phase.
- Phase 1.1 — Cell 002 (Markdown): Labels runtime dependency installation.
- Phase 1.1 — Cell 003 (Code): Installs rank_bm25 and sentence-transformers so the notebook has the retrieval and embedding packages it needs on Kaggle.
- Phase 1.2 — Cell 004 (Markdown): Labels the library-import section.
- Phase 1.2 — Cell 005 (Code): Imports pandas, NumPy, TF-IDF, BM25, MiniLM, LightGBM, XGBoost, W&B, metrics, and GroupKFold.
- Phase 1.3 — Cell 006 (Markdown): Labels Kaggle and local path resolution.
- Phase 1.3 — Cell 007 (Code): Sets the three Kaggle data paths, replaces them with local alternatives when needed, and prints the paths selected for the run.
- Phase 1.4 — Cell 008 (Markdown): Labels W&B configuration and hyperparameters.
- Phase 1.4 — Cell 009 (Code): Loads the W&B key, starts the Model 3 run, and records the TF-IDF, MiniLM, cross-validation, tree-model, and blend settings in W&B.
- Phase 2 — Cell 010 (Markdown): Opens the data-pipeline phase.
- Phase 2.1 — Cell 011 (Markdown): Labels CSV loading and schema checking.
- Phase 2.1 — Cell 012 (Code): Reads train and test data, adds a blank context field if necessary, fills missing text with empty strings, and fixes the option order.
- Phase 2.2 — Cell 013 (Markdown): Labels text normalization and lexical feature creation.
- Phase 2.2 — Cell 014 (Code): Cleans text, builds context-plus-prompt queries, gathers a shared corpus, and fits word-level and character-level TF-IDF vectorizers.
- Phase 2.3 — Cell 015 (Markdown): Labels semantic feature engineering.
- Phase 2.3 — Cell 016 (Code): Loads MiniLM, encodes questions and options in batches, and computes normalized question-option cosine similarities.
- Phase 3 — Cell 017 (Markdown): Opens the model-architecture and feature-definition phase.
- Phase 3.1 — Cell 018 (Markdown): Labels long-format feature extraction.
- Phase 3.1 — Cell 019 (Code): Defines Jaccard overlap and builds one feature row per question-option pair. It combines word TF-IDF, character TF-IDF, BM25, MiniLM similarity, word overlap, and text-length features for both train and test data.
- Phase 3.2 — Cell 020 (Markdown): Labels relative-ranking feature augmentation.
- Phase 3.2 — Cell 021 (Code): Adds each main feature's difference from the question mean and its within-question rank. It then creates the final feature-column list used by the tree models.
- Phase 4 — Cell 022 (Markdown): Opens training and evaluation.
- Phase 4.1 — Cell 023 (Markdown): Labels grouped cross-validation training.
- Phase 4.1 — Cell 024 (Code): Builds train and test feature matrices, creates GroupKFold splits by question ID, trains LightGBM and XGBoost inside every fold, applies class balancing and early stopping, and averages validation and test probabilities.
- Phase 4.2 — Cell 025 (Markdown): Labels blend-weight search and out-of-fold evaluation.
- Phase 4.2 — Cell 026 (Code): Defines MAP@3 for grouped option rows, sweeps the LightGBM blend weight, selects the best out-of-fold value, calculates accuracy, macro-F1, and log loss, and logs those results to W&B.
- Phase 5 — Cell 027 (Markdown): Opens the conclusion and submission phase.
- Phase 5.1 — Cell 028 (Markdown): Labels test inference, blending, and submission export.
- Phase 5.1 — Cell 029 (Code): Blends averaged test probabilities with the selected alpha, ranks the five options per question, matches the sample-submission column names, writes submission.csv, and finishes the W&B run.
