# VIVA QUESTIONS PREP

Source used: live Google Sheet CSV export from the provided viva-question sheet, plus the three final notebooks:
`nb/scratch/base_model.ipynb`, `nb/pretrained/pre_trained.ipynb`, and `nb/custom/model3.ipynb`.

The current production notebooks use the same five-phase spine: **Phase 1: Setup & Configuration**, **Phase 2: Data Pipeline**, **Phase 3: Model Architecture & Definition**, **Phase 4: Training and Evaluation**, and **Phase 5: Conclusion / Inference & Submission**. Numbered subheadings such as `## 1.1 Library Imports` are dedicated markdown cells; code comments are not part of the Kaggle TOC.

The locked final results are **0.74397 MAP@3 for Model 2** and **0.74608 MAP@3 for Model 3**. Model 2 uses a 3-epoch early-stopping cap and saves the best checkpoint by minimum `val_loss` to reduce overfitting.

## Identity, Ownership, And Project Walkthrough

### Q1. What are the three models you implemented?

I implemented **Model 1 from scratch**, **Model 2 with pretrained BERT**, and **Model 3 as a custom LightGBM/XGBoost ensemble**. The final notebooks are isolated in `nb/scratch/base_model.ipynb`, `nb/pretrained/pre_trained.ipynb`, and `nb/custom/model3.ipynb`.

```python
model_name      = 'MCQTransformerScratch',
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
```

### Q2. What is the problem statement?

The task is to rank the top three correct answer labels for each MCQ with options **A-E**. My code turns every question into option-level scores, then writes a Kaggle submission with exactly three labels per row.

```python
OPTIONS = ['A', 'B', 'C', 'D', 'E']
top3_idx = torch.topk(logits, 3, dim=1).indices.cpu().numpy()  # (B, 3) descending
```

### Q3. How do you prove this is your work?

I can walk through the repo history, W&B runs, Kaggle notebook versions, and explain each final notebook cell-by-cell. The code has model-specific paths, run names, and project-specific preprocessing rather than generic copied placeholders.

```python
run = wandb.init(
    project = '23f2004343-t22026',
```

### Q4. What files should you show first in viva?

Show the three final notebooks, then W&B, GitHub commits, Kaggle submission history, README, and report. If asked for code ownership, start with the scratch model because it exposes the most custom code.

```python
CONFIG = dict(
    model_name      = 'MCQTransformerScratch',
```

## Data Pipeline And Preprocessing

### Q5. How do you load data in Kaggle and locally?

The notebooks use dynamic path resolution so the same code works on Kaggle and local folders. The resolver checks the Kaggle competition mount first, then local `data/` paths.

```python
def _resolve(kaggle_abs, *local_candidates):
    """Return the first existing path; default to kaggle_abs for cloud runs."""
```

### Q6. How do you handle missing values?

In Model 3, I fill all text columns before feature construction so `"nan"` is not treated as a real word. This is important because TF-IDF, BM25, and MiniLM all consume text directly.

```python
TEXT_COLS = ["context", "prompt", *OPTION_COLS]
# fill NaN text fields with empty string so vectorisers never see NaN tokens
df[col] = df[col].fillna("")
```

### Q7. What preprocessing happens in the scratch model?

The scratch model lowercases text, removes non-alphanumeric characters, splits on whitespace, and maps words into a custom vocabulary. This keeps Model 1 independent of pretrained tokenizers.

```python
text = re.sub(r'[^a-z0-9\s]', ' ', text)
return text.split()
```

### Q8. What is tokenization in your pretrained model?

BERT tokenization creates five prompt-option pairs per question and pads/truncates them to a fixed length. This gives `input_ids` and `attention_mask` tensors shaped like five choices per sample.

```python
encodings = tokenizer(
    [prompt] * 5,
```

### Q9. What are `input_ids`?

`input_ids` are integer token IDs produced by either my `WordVocab` or the BERT tokenizer. They are the model-readable version of text.

```python
ids   = torch.tensor(self.records[idx], dtype=torch.long)
'input_ids'     : encodings['input_ids'],
```

### Q10. What is `attention_mask`?

The **attention mask** marks real tokens as `1` and padding as `0`. In the scratch model, it prevents padding from affecting masked mean pooling and transformer attention.

```python
mask  = (ids != PAD_IDX).long()          # 1=real token, 0=pad
pad_mask = (attention_mask == 0) if attention_mask is not None else None
```

### Q11. What are the exact tensor shapes before Model 1?

Model 1 uses long format, so one batch contains many individual question-option pairs. The shapes are `input_ids: (batch, 128)`, `attention_mask: (batch, 128)`, and `label: (batch,)`.

```python
input_ids      : (batch, seq_len)  integer token ids
attention_mask : (batch, seq_len)  1=real, 0=pad (optional)
```

### Q12. What are the exact tensor shapes before Model 2?

Model 2 keeps all five options inside one sample, so the tensors are `input_ids: (batch, 5, 128)` and `attention_mask: (batch, 5, 128)`. The label is one integer from `0` to `4`.

```python
input_ids      : (5, max_len)
attention_mask : (5, max_len)
```

### Q13. Why do you convert MCQs into long format in Model 1 and Model 3?

Long format makes each option a separate scored candidate. That lets binary classifiers or tree models produce one relevance score per option before regrouping by question for top-3 ranking.

```python
for opt_key in OPTION_COLS:
    combined = q_text + ' ' + str(row[opt_key])
```

## Model 1: Scratch Transformer

### Q14. Explain your scratch model architecture.

Model 1 uses **nn.Embedding**, fixed sinusoidal positional encoding, a 2-layer **nn.TransformerEncoder**, masked mean pooling, and a small MLP classifier. It outputs one scalar logit for each question-option pair.

```python
self.embedding = nn.Embedding(
self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)
```

### Q15. Is Model 1 really from scratch?

Yes, Model 1 does not use Hugging Face or pretrained tokenizers for the model architecture. The vocabulary is built with `collections.Counter`, and all neural weights are initialized and trained inside PyTorch.

```python
freq = collections.Counter()
self._init_weights()
```

### Q16. What does `nn.Embedding` do?

`nn.Embedding` maps each integer token ID into a dense vector of size `d_model`. In my config, that hidden dimension is **256**.

```python
d_model         = 256,     # transformer hidden dimension
x = self.embedding(input_ids)    # (batch, seq_len, d_model)
```

### Q17. Why do you need positional encoding?

Self-attention alone does not know token order, so sinusoidal positional encoding injects position information into embeddings. I used a fixed non-learnable positional encoding to keep the scratch model simple and explainable.

```python
self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
x = self.pos_enc(x)
```

### Q18. What does `batch_first=True` mean?

It means tensors use `(batch, seq_len, features)` ordering instead of `(seq_len, batch, features)`. My scratch transformer uses this so the embedding output can go directly into the encoder.

```python
dropout=dropout, batch_first=True,
x = self.embedding(input_ids)    # (batch, seq_len, d_model)
```

### Q19. What activation functions are used in Model 1?

The classifier uses **GELU**, which is common in transformer-style models because it is smooth and works well with deep representations. The final layer produces raw logits, not probabilities.

```python
nn.GELU(),
nn.Linear(d_model // 2, 1),
```

### Q20. Why do you use masked mean pooling?

Masked mean pooling converts `(batch, seq_len, d_model)` into `(batch, d_model)` while ignoring padding positions. Without the mask, padded zeros could dilute the representation.

```python
m      = attention_mask.unsqueeze(-1).float()
pooled = (enc * m).sum(1) / m.sum(1).clamp(min=1e-9)
```

### Q21. What loss does Model 1 use and why?

Model 1 scores each option independently as correct or incorrect, so **BCEWithLogitsLoss** is appropriate for binary option-level labels. The sigmoid is only used later for interpreting logits as probabilities.

```python
criterion = nn.BCEWithLogitsLoss()
pb = (torch.sigmoid(logits) > 0.5).cpu().numpy()
```

### Q22. Why do you call `zero_grad()`?

PyTorch accumulates gradients by default, so `zero_grad()` clears old gradients before each new batch. It belongs inside the batch loop before `loss.backward()`.

```python
optimizer.zero_grad()
loss.backward()
```

### Q23. Why do you use gradient clipping?

Gradient clipping prevents very large gradient updates from destabilizing training. I clip gradients with `max_norm=1.0` in both scratch and pretrained training loops.

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Q24. What scheduler do you use in the scratch model?

The scratch model uses a cosine annealing scheduler so the learning rate gradually changes over epochs. In my loop, `scheduler.step()` is called after each epoch.

```python
scheduler.step()
wandb.log({
```

## Architecture Defense & Mathematics

**Scaled Dot-Product Attention** maps a query and a set of key-value pairs to a weighted value mixture:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right)V.
$$

- **Queries ($Q$)** are learned projections of the current token representation: they express what contextual information that token is looking for.
- **Keys ($K$)** are learned projections used to measure each token's compatibility with a query; $QK^{\mathsf{T}}$ produces the attention-score matrix.
- **Values ($V$)** carry the information that is mixed by the normalized attention weights to form each contextualized output.
- **The $\sqrt{d_k}$ scaling factor** keeps score magnitudes controlled as the key dimension grows, preventing softmax saturation and the extremely small (vanishing) gradients that saturation can create.

**Multi-Head Attention** applies this operation in parallel after distinct learned projections, then concatenates and projects the results:

$$
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O, \qquad
\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V).
$$

This lets different heads attend to different representation subspaces and token positions. A Transformer encoder layer applies multi-head self-attention, a position-wise feed-forward block, residual connections, and layer normalization.

**Positional Encoding** is necessary because self-attention has no inherent token order. Model 1 adds fixed sinusoidal vectors to token embeddings:

$$
\operatorname{PE}(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
\operatorname{PE}(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).
$$

**Model relationship:** Model 1 is a from-scratch implementation of this encoder pattern: its vocabulary, embeddings, attention projections, and classifier weights are initialized and learned within this project, with fixed sinusoidal positions. Model 2 is BERT, a heavily pre-trained, bidirectional version of the Transformer encoder: its encoder self-attention can use both left and right context, it starts from learned pre-trained weights, and its multiple-choice head scores each prompt-option sequence through the final `[CLS]` representation. BERT uses learned position embeddings rather than Model 1's fixed sinusoidal encoding.

Basis: Vaswani et al. (2017), *Attention Is All You Need*; Devlin et al. (2018), *BERT*.

## Model 2: Pretrained BERT

### Q25. Why did you use `AutoModelForMultipleChoice`?

This class is designed for exactly this setup: one question with multiple candidate choices. It returns one logit per option, giving `logits: (batch, 5)`.

```python
# AutoModelForMultipleChoice: takes (batch, 5, seq_len), returns logits (batch, 5)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
```

### Q26. How does BERT process all five options?

The tokenizer builds five `(prompt, option)` pairs, and BERT scores each choice using the same shared encoder. The multiple-choice head then produces five raw scores per question.

```python
out    = model(input_ids=ids, attention_mask=masks)
logits = out.logits  # (batch, 5) - raw scores before softmax
```

### Q27. Why is `CrossEntropyLoss` used for Model 2?

Model 2 is a 5-class choice problem where the target is the correct option index. `CrossEntropyLoss` directly compares the five logits against that integer label.

```python
criterion = nn.CrossEntropyLoss()
loss   = criterion(logits, labs)
```

### Q28. Why did you not pass `token_type_ids`?

The current BERT setup scores prompt-option pairs directly with `input_ids` and `attention_mask`, matching the final notebook logic. If asked, say the model still receives the pair structure from the tokenizer inputs and validation selected this stable setup.

```python
out    = model(input_ids=ids, attention_mask=masks)
logits = out.logits  # (batch, 5) - raw scores before softmax
```

### Q29. What is fine-tuning here?

Fine-tuning means the pretrained BERT weights are loaded first, then updated on the MCQ training data. I did not freeze layers in this final notebook, so trainable parameters are optimized with AdamW.

```python
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
optimizer = AdamW(
```

### Q30. How do you generate top-3 predictions in Model 2?

I sort or top-k the five logits for each question and select the highest three indices. Those indices are mapped back to option letters.

```python
top3_idx = torch.topk(logits, 3, dim=1).indices.cpu().numpy()  # (B, 3) descending
all_top3.append(' '.join(OPTIONS_ARR[row_idx]))
```

## Model 3: TF-IDF, BM25, MiniLM, And Ensemble

### Q31. What is TF-IDF and why did you use it?

**TF-IDF** gives high weight to terms that are important in one text but not common everywhere. I used word TF-IDF for keywords and character TF-IDF for spelling variants, abbreviations, and partial word patterns.

```python
word_tfidf = TfidfVectorizer(
char_tfidf = TfidfVectorizer(
```

### Q32. What is BM25 and why is it useful?

**BM25** is a retrieval scoring method that improves over plain term frequency by adding saturation and length normalization. In my Model 3, it scores each question prompt against all five option texts.

```python
bm25_raw  = BM25Okapi([t.split() for t in opt_texts]).get_scores(query_tokens)
bm25_norm = bm25_raw / bm25_max
```

### Q33. Why use MiniLM if you already have TF-IDF and BM25?

TF-IDF and BM25 capture lexical overlap, while **MiniLM** captures semantic similarity. This helps when the option uses a paraphrase instead of the same words as the question.

```python
minilm = SentenceTransformer("all-MiniLM-L6-v2")
train_q_embs = minilm.encode(train_queries,   batch_size=64, show_progress_bar=True,
```

### Q34. What are relative rank features?

Relative rank features compare options inside the same question, so the tree model knows which option is strongest for that question. A rank of `1` means that option scored highest for that feature among the five options.

```python
df[f"{feat}_diff"] = df[feat] - grp[feat].transform("mean")
df[f"{feat}_rank"] = grp[feat].rank(ascending=False, method="min")
```

### Q35. Why use GroupKFold?

Each original question creates five option rows, so random splitting could leak sibling options across train and validation. `GroupKFold` keeps all five options for the same `id` in the same fold.

```python
gkf     = GroupKFold(n_splits=N_FOLDS)
groups = train_feat["id"].values   # group identifier = question id
```

### Q36. How do you handle class imbalance in Model 3?

There is one correct option and four wrong options per question, so Model 3 uses balancing in both tree models. LightGBM uses `class_weight="balanced"` and XGBoost uses `scale_pos_weight`.

```python
class_weight     = "balanced",
scale_pos_weight      = spw,
```

### Q37. Why use both LightGBM and XGBoost?

Both are gradient-boosted tree models, but they can learn slightly different decision boundaries and handle feature interactions differently. Blending their probabilities can be more stable than relying on one model.

```python
lgb_clf = lgb.LGBMClassifier(
xgb_clf = xgb.XGBClassifier(
```

### Q38. How is the ensemble blended?

The final probability is a weighted average of LightGBM and XGBoost probabilities. The best alpha is selected using out-of-fold MAP@3, so it is validation-driven.

```python
oof_df["pred_proba"] = alpha * oof_lgb + (1 - alpha) * oof_xgb
best_alpha = round(float(alpha), 2)
```

### Q39. How does Model 3 make the final submission?

It computes blended probabilities, sorts the five options per question, and joins the top three option letters. The final columns are copied from `sample_submission.csv` or default to `ID,Prediction`.

```python
test_feat["pred_proba"] = best_alpha * test_lgb_preds + (1 - best_alpha) * test_xgb_preds
submission.columns = [id_col, pred_col]
```

## Evaluation, Metrics, And W&B

### Q40. What is MAP@3?

**MAP@3** measures whether the correct answer appears in the top three predictions and gives more credit when it appears earlier. If the correct answer is first, the score is `1`; second gives `1/2`; third gives `1/3`.

```python
for k, pred in enumerate(ranked[:3], start=1):
    score += hits / k
```

### Q41. How do you calculate MAP@3 in Model 2?

Model 2 sorts the five logits, takes the top three, and compares those ranked indices with the true class index. The mean AP@3 over validation rows becomes `val_map3`.

```python
top_3  = torch.argsort(logits, dim=1, descending=True)[:, :3].cpu().tolist()
'val_map3'    : float(np.mean(all_ap3)) if all_ap3 else 0.0,
```

### Q42. Why checkpoint on validation loss instead of MAP@3 in Model 2?

MAP@3 is the competition metric, but it can jump around because it depends on ranking positions. Model 2 uses an early-stopping cap of 3 epochs and saves the checkpoint using minimum `val_loss`, which gives a more stable defense against overfitting while still logging MAP@3.

```python
if val_m['val_loss'] < best_val_loss:
    best_val_loss = val_m['val_loss']
```

### Q43. What metrics do you log to W&B?

I log loss, accuracy, macro-F1, MAP@3, learning rate, and model-specific settings. W&B authentication checks Kaggle Secrets first, then the local `.env` environment variable, and raises if no key is available, so the production notebooks contain zero hardcoded keys.

```python
# Kaggle Secrets -> local .env -> raise if unavailable
wandb.log({'epoch': epoch, 'train_loss': avg_trn, **val_m})
"oof_f1_macro":   oof_f1,
```

### Q44. Why use macro-F1?

Macro-F1 treats each class equally, which is useful when class distribution is imbalanced. In this MCQ setup, it gives another view beyond accuracy and MAP@3.

```python
f1  = f1_score(labels_b, preds_b, average='macro', zero_division=0)
'val_f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
```

### Q45. How do you prevent gradient updates during evaluation?

I use `model.eval()` and `torch.no_grad()` so dropout-like behavior is disabled and gradients are not tracked. This makes validation and inference faster and deterministic.

```python
model.eval()
with torch.no_grad():
```

## Optimizers, Schedulers, And Training Loops

### Q46. What optimizer did you use?

Both scratch and pretrained notebooks use **AdamW**, which decouples weight decay from Adam's adaptive updates. It is a strong default for transformer-style training.

```python
optimizer = AdamW(
    model.parameters(),
```

### Q47. What is the difference between Adam and AdamW?

Adam includes adaptive moment estimates, while AdamW applies weight decay separately from the gradient update. That separate decay often regularizes neural networks more cleanly.

```python
weight_decay    = 1e-2,
'optimizer'     : 'AdamW',
```

### Q48. Where does backpropagation happen?

Backpropagation happens after computing loss in the batch loop. The pattern is clear: zero old gradients, compute loss, call `backward`, then update parameters.

```python
loss.backward()
optimizer.step()
```

### Q49. Why call `scheduler.step()`?

The scheduler updates the learning rate according to the selected schedule. In Model 2 it steps each batch after the optimizer, while in Model 1 it steps once per epoch.

```python
optimizer.step()
scheduler.step()
```

### Q50. How would you write a basic training loop live?

Use the same structure as the notebooks: loop over epochs, loop over batches, move tensors to device, compute logits/loss, backpropagate, step optimizer, and log metrics. Start with `model.train()` and clear gradients inside the batch loop.

```python
model.train()
optimizer.zero_grad()
```

## Theory Questions Proctors Frequently Ask

### Q51. What is overfitting and how would you detect it?

Overfitting means training metrics improve while validation metrics stop improving or degrade. In this project, I would inspect W&B train loss versus validation loss/MAP@3 curves.

```python
wandb.log({'epoch': epoch, 'train_loss': avg_trn, **val_m})
print(f"OOF MAP@3               : {best_map3:.4f}")
```

### Q52. What is dropout?

Dropout randomly disables some activations during training to reduce co-adaptation. In my scratch model, dropout appears inside positional encoding, transformer layers, and the classifier head.

```python
nn.Dropout(dropout),
dropout=dropout, batch_first=True,
```

### Q53. What is self-attention?

Self-attention lets each token compare itself with every other token in the sequence. In my scratch model, PyTorch's `TransformerEncoderLayer` implements this internally with `nhead=4`.

```python
nhead           = 4,       # attention heads (must divide d_model)
enc_layer = nn.TransformerEncoderLayer(
```

### Q54. What is an embedding?

An embedding is a dense vector representation of a token or sentence. Model 1 learns word embeddings from scratch, while Model 3 uses MiniLM sentence embeddings.

```python
self.embedding = nn.Embedding(
minilm = SentenceTransformer("all-MiniLM-L6-v2")
```

### Q55. What is the difference between tokens and embeddings?

Tokens are discrete IDs or text pieces; embeddings are continuous vectors used by models. In Model 1, token IDs become embeddings through `nn.Embedding`.

```python
ids   = torch.tensor(self.records[idx], dtype=torch.long)
x = self.embedding(input_ids)    # (batch, seq_len, d_model)
```

### Q56. Why not use an external LLM API?

The project rule bans hosted inference APIs, so all inference must run inside Kaggle/local execution. My notebooks use local PyTorch, Hugging Face model loading, sentence-transformers, and tree models instead.

```python
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
minilm = SentenceTransformer("all-MiniLM-L6-v2")
```

### Q57. How would you improve the project?

I would add stronger cached retrieval, tune a larger local transformer on Kaggle GPU, improve calibration of the ensemble, and add better error analysis by question type. I would keep the three-model isolation and W&B comparison intact.

```python
"best_lgb_alpha": best_alpha,
wandb.log({'epoch': epoch, 'train_loss': avg_trn, **val_m})
```

### Q58. What should you say if asked why your Model 3 is not a neural network?

Model 3 is the custom experimental model, allowed by the rubric as an ensemble or hybrid approach. It combines semantic embeddings, lexical retrieval-style features, and boosted trees to improve ranking.

```python
FEAT_COLS = BASE_COLS + RANK_COLS
lgb_clf = lgb.LGBMClassifier(
```

### Q59. What is your final Kaggle output format?

The output has one row per test question with `ID` and a space-separated top-three `Prediction`. Model 3 explicitly preserves the exact sample submission columns.

```python
submission.columns = [id_col, pred_col]
submission.to_csv("submission.csv", index=False)
```

### Q60. What is the safest 30-second summary of the whole project?

I built three isolated MCQ rankers: a scratch PyTorch transformer, a fine-tuned BERT multiple-choice model, and a custom TF-IDF/BM25/MiniLM plus LightGBM/XGBoost ensemble. All of them score five options, rank them, log comparable W&B metrics, and generate top-three Kaggle predictions.

```python
top3_idx = torch.topk(logits, 3, dim=1).indices.cpu().numpy()  # (B, 3) descending
submission.columns = [id_col, pred_col]
```
