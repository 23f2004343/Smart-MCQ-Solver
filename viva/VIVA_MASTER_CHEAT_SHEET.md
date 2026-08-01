# VIVA MASTER CHEAT SHEET

## Locked Final Results

- **Model 2:** `0.74397 MAP@3`, using a 3-epoch early-stopping cap and minimum-`val_loss` checkpoint selection to limit overfitting.
- **Model 3:** `0.74608 MAP@3`, the highest final Kaggle score from the LightGBM/XGBoost feature ensemble.

## Current Notebook Spine

- **Phase 1: Setup & Configuration** covers imports, path resolution, W&B configuration, and hyperparameters.
- **Phase 2: Data Pipeline** covers CSV loading, cleaning, tokenization, feature construction, and data splits.
- **Phase 3: Model Architecture & Definition** covers the scratch Transformer, BERT multiple-choice model, or Model 3 feature frame.
- **Phase 4: Training and Evaluation** covers losses, metrics, validation, cross-validation, checkpointing, and blend selection.
- **Phase 5: Conclusion / Inference & Submission** covers top-three ranking, `ID,Prediction` formatting, artifact logging, and W&B teardown.

## Data Pipeline

- **Raw rows** contain `id`, `prompt`, options `A-E`, and `answer` for training; the current developer comment is `# fill NaN text fields with empty string so vectorisers never see NaN tokens`.
- **Model 1** concatenates `context + prompt + option`, tokenizes with `simple_tokenize`, maps words through `WordVocab`, and returns `input_ids` plus an `attention_mask`.
- **Model 1 tensor shape** is `input_ids: (batch, 128)`, `attention_mask: (batch, 128)`, and `label: (batch,)` because each row represents one question-option pair.
- **Model 2** tokenizes five `(prompt, option)` pairs per question using the BERT tokenizer with padding and truncation to `max_seq_length=128`.
- **Model 2 tensor shape** is `input_ids: (batch, 5, 128)`, `attention_mask: (batch, 5, 128)`, and `label: (batch,)`, where the label is the correct option index from `0` to `4`.
- **Model 3** converts each question into five feature rows, one per option, so the final matrices are `X_all: (num_train_questions * 5, 27)` and `X_test: (num_test_questions * 5, 27)`.

## Model 1 (Scratch) Core Logic

- **WordVocab** is built from scratch with `collections.Counter`, using `<PAD>=0`, `<UNK>=1`, truncation to `max_seq_len=128`, and right-padding for shorter text.
- **MCQOptionDataset** creates one binary training example per option, so the correct option has label `1.0` and the four distractors have label `0.0`.
- **nn.Embedding** converts `(batch, seq_len)` token IDs into `(batch, seq_len, d_model)`, with `d_model=256` from the config.
- **SinusoidalPositionalEncoding** adds fixed position information so self-attention can understand token order without using pretrained weights.
- **nn.TransformerEncoderLayer** and **nn.TransformerEncoder** form a 2-layer transformer encoder with `nhead=4`, `dim_feedforward=512`, dropout, residual connections, and layer normalization internally.
- **Masked mean pooling** averages only real tokens using the `attention_mask`, then the classifier `Linear -> GELU -> Dropout -> Linear` outputs one scalar logit per option.

## Model 2 (Pretrained) Flow

- **AutoTokenizer** builds five BERT inputs per question by pairing the same prompt with each option `A-E`.
- **AutoModelForMultipleChoice** receives tensors shaped `(batch, 5, seq_len)` and internally evaluates the five choices with the shared BERT encoder.
- **BERT output** is reduced by its multiple-choice classification head into `logits: (batch, 5)`, one score for each option.
- **CrossEntropyLoss** compares those five logits against the correct option index, not a one-hot vector.
- **Overfitting control** is a 3-epoch limit with the best checkpoint selected by minimum validation loss; MAP@3 remains a logged evaluation metric.
- **W&B security** checks Kaggle Secrets first, then a local `.env`-provided `WANDB_API_KEY`, and raises if unavailable; no key is hardcoded in the notebook.
- **MAP@3 evaluation** sorts the five logits descending, keeps the top three option indices, and rewards the model more when the true answer appears earlier.
- **Inference** uses `torch.topk(logits, 3, dim=1)` and maps the resulting indices back to option letters for the `Prediction` column.

## Model 3 (Ensemble) Strategy

- **Text cleaning** lowercases text and collapses whitespace, while preserving punctuation for character n-gram features.
- **TF-IDF features** include word-level `1-2` grams and character-level `2-4` grams, then compute cosine similarity between the question query and each option.
- **BM25** scores all five option texts against the prompt tokens and adds both raw `bm25` and normalized `bm25_norm` features.
- **MiniLM cosine similarity** adds a semantic feature from 384-dimensional sentence embeddings, which catches paraphrases that sparse TF-IDF can miss.
- **Relative rank features** add per-question `diff` and descending `rank` columns for the main similarity features, making the model compare options inside the same question.
- **LightGBM and XGBoost** are trained with `GroupKFold` by question `id`, so all five options from one question stay in the same fold and avoid leakage.
- **Blending** computes `pred_proba = alpha * LGB + (1 - alpha) * XGB`, sweeps `alpha` on out-of-fold MAP@3, then uses the best alpha for final test predictions.
- **Submission generation** sorts each question's five option probabilities descending, joins the top three letters with spaces, and writes the Kaggle schema `ID,Prediction`.
