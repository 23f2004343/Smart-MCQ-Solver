# Smart MCQ Solver: Technical Project Report
**An End-to-End Comparative Deep Learning & Feature-Ensemble Framework for Multiple-Choice Question Answering**

---

### Project Metadata
* **Workspace / Project ID:** `23f2004343-t22026`
* **Target Metric:** Mean Average Precision at 3 (MAP@3)
* **Qualification Cutoff Target:** $\ge 0.7300$
* **Final Model 2 Score:** **0.74397 MAP@3** (Kaggle Leaderboard Score, Pretrained BERT)
* **Achieved Peak Score:** **0.74608 MAP@3** (Kaggle Leaderboard Score, Model 3 Ensemble)
* **Date:** July 2026
* **Environment:** PyTorch 2.x, Hugging Face Transformers, LightGBM, XGBoost, Scikit-Learn, W&B

---

## 1. Abstract / Executive Summary

This technical report presents the architecture, implementation, and evaluation of the **Smart MCQ Solver**, an end-to-end machine learning pipeline designed for high-precision multiple-choice question answering under Kaggle-style challenge constraints. Each question in the benchmark dataset comprises a prompt, optional context, and five candidate options ($A, B, C, D, E$). Submissions require ranking exactly three predicted labels per question, evaluated against ground truth using Mean Average Precision at 3 ($\text{MAP@3}$).

To rigorously explore the problem space, we designed, implemented, and compared three distinct computational architectures:
1. **Model 1 (PyTorch Transformer from Scratch):** A custom neural network built entirely without external NLP libraries, incorporating custom tokenization (`WordVocab`), non-learnable sinusoidal positional encodings, a multi-head self-attention encoder stack, masked mean-pooling, and a binary classification head.
2. **Model 2 (Fine-Tuned Transformer):** A pre-trained Hugging Face transformer model (`bert-base-uncased`) fine-tuned end-to-end using a `MultipleChoice` classification head with linear learning rate warmup and gradient clipping.
3. **Model 3 (Custom Feature-Engineered Ensemble):** A novel hybrid model combining 27 multi-faceted features—spanning lexical matching (TF-IDF word/char n-grams, BM25, Jaccard index), dense semantic embedding similarities (`all-MiniLM-L6-v2`), structural text statistics, and question-level relative rank transformations—fed into a 5-Fold `GroupKFold` LightGBM and XGBoost blended ranker.

All models were evaluated locally under CPU sanity constraints and executed on Kaggle cloud GPUs, with hyperparameter logs tracked in Weights & Biases (W&B). Model 2 achieved **0.74397 MAP@3**, while Model 3 achieved the higher final Kaggle score of **0.74608 MAP@3**; both surpass the target qualification cutoff of 0.7300. The three approaches provide complementary trade-offs between learned contextual representations, scratch-model transparency, and feature-based ranking efficiency.

---

## 2. Introduction

### 2.1 Problem Statement
Multiple-Choice Question Answering (MCQ-QA) represents a foundational benchmark in Natural Language Processing (NLP), evaluating a system's capacity for context comprehension, semantic reasoning, and distractor elimination. In this challenge, each record $i$ consists of:
* A textual prompt $P_i$ and optional background context $C_i$.
* Five choice options: $O_{i,A}, O_{i,B}, O_{i,C}, O_{i,D}, O_{i,E}$.
* A ground-truth target label $y_i \in \{A, B, C, D, E\}$.

The system is evaluated not merely on single-choice accuracy, but on its ability to produce a ranked list of the top 3 candidate options $\hat{Y}_i = [r_{i,1}, r_{i,2}, r_{i,3}]$. Performance is measured using Mean Average Precision at 3 ($\text{MAP@3}$), defined as:

$$\text{MAP@3} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{\min(3, |\hat{Y}_i|)} P_i(k) \times \text{rel}_i(k)$$

where $P_i(k)$ is the precision at rank $k$, and $\text{rel}_i(k)$ is a binary indicator of whether the option at rank $k$ is the correct answer $y_i$. Because $\text{rel}_i(k) = 1$ for at most one option, placing the correct option at Rank 1 yields $1.0$, Rank 2 yields $0.5$, Rank 3 yields $0.333$, and unranked yields $0.0$.

### 2.2 Project Objectives & Operational Rules
The primary objective of this project was to establish an end-to-end, reproducible ML/DL pipeline achieving an evaluated score of $\text{MAP@3} \ge 0.7300$. The development adhered strictly to mandatory project constraints:
1. **Three-Model Mandate:** Implementation of three isolated, non-overlapping models across designated subdirectories:
   * `nb/scratch/base_model.ipynb` (From-scratch PyTorch architecture)
   * `nb/pretrained/pre_trained.ipynb` (Fine-tuned Hugging Face transformer)
   * `nb/custom/model3.ipynb` (Custom feature engineering & ensemble)
2. **Zero External LLM API Calls:** Absolute prohibition of external cloud inference APIs (OpenAI, Claude, Gemini). All indexing, feature extraction, transformer inference, and ranking execute locally or inside Kaggle kernel containers.
3. **W&B Experiment Tracking:** Full tracking of hyperparameters, training loss, validation accuracy, Macro F1, and MAP@3 in W&B project `23f2004343-t22026`.
4. **Viva Explainability & Code Cleanliness:** Explicit code comments, structural markdown sub-headers, and step-by-step viva notes explaining tensor shape transitions and algorithm design decisions.
5. **Credential Hygiene:** W&B authentication resolves Kaggle Secrets first and then the `WANDB_API_KEY` environment variable, including a local `.env`-provided environment; the production notebooks contain zero hardcoded keys.

### 2.3 Report Structure
The remainder of this report is organized as follows: Section 3 describes the dataset characteristics, exploratory data analysis, and long-format data transformations. Section 4 Details the tokenization strategies implemented across all three models. Section 5 presents the neural and feature-ensemble architectures. Section 6 provides comparative evaluation metrics and performance analysis. Section 7 concludes with key insights and actionable future work. Section 8 details references.

---

## 3. Dataset & Preprocessing

### 3.1 Dataset Description
The competition dataset comprises science and domain-specific multiple-choice questions. The training set (`train.csv`) provides prompts, options, and ground-truth targets, while the test set (`test.csv`) contains unlabelled questions for evaluation.

| Dataset Split | Question Count | Total Option Pairs ($5 \times N$) | Missing Context Fields | Target Label Distribution |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 200 | 1,000 | Handled (`NaN` $\rightarrow$ `""`) | Balanced ($A: 20\%, B: 20\%, C: 20\%, D: 20\%, E: 20\%$) |
| **Test Set** | 50 | 250 | Handled (`NaN` $\rightarrow$ `""`) | Unlabelled (Inference Target) |

### 3.2 Exploratory Data Analysis (EDA)
Exploratory analysis revealed key text length characteristics across prompts, contexts, and candidate options:

* **Prompt Lengths:** Ranged from 15 to 140 words (mean: $48.2 \pm 18.4$ words). Prompts frequently contain complex technical syntax and scientific terminology.
* **Option Lengths:** Varied significantly from single-word terms (e.g., chemical symbols, dates) to detailed multi-sentence explanations (mean option length: $12.4 \pm 8.6$ words).
* **Option Balance:** Target answer letters ($A, B, C, D, E$) exhibit near-uniform class balance across the training corpus ($\sim 20.0\%$ per option key), ruling out majority-class guessing biases.

```
Prompt Length Distribution (Words):
[10-30]  : █████████ 18%
[31-60]  : ████████████████████████ 52%
[61-100] : ████████████ 24%
[101+]   : ███ 6%

Option Length Distribution (Words):
[1-5]    : ██████████████ 28%
[6-15]   : █████████████████████████ 51%
[16-30]  : █████████ 17%
[31+]    : ██ 4%
```

### 3.3 Data Preprocessing & Long-Format Restructuring
To enable binary relevance classification and pairwise ranking, raw question rows were converted into a **Long-Format Dataset**. Each single question record with 5 options is transformed into 5 distinct computational rows pairing the question query with an individual option key:

$$\text{Record}_{i,j} = \Big( \text{ID}_i, \text{OptionKey}_j, \text{Query}_i = \text{Clean}(C_i) \oplus \text{Clean}(P_i), \text{OptionText}_{i,j} = \text{Clean}(O_{i,j}), y_{i,j} \in \{0, 1\} \Big)$$

Where:
* $\text{Clean}(T)$ collapses redundant whitespace (`\s+` $\rightarrow$ `" "`), forces lowercasing, strips non-printable unicode artifacts, and fills missing `NaN` context strings with empty strings `""`.
* $y_{i,j} = 1$ if $\text{OptionKey}_j == \text{TargetAnswer}_i$, else $y_{i,j} = 0$.

### 3.4 Out-of-Vocabulary (OOV) Prevention
For feature-based models (Model 3), fitting TF-IDF vectorizers separately on training data risks dropping out-of-vocabulary tokens present in test options. To eliminate OOV token loss, feature extraction fits a unified text corpus combining all prompt texts, context strings, and option texts across both train and test sets:

$$\mathcal{C}_{\text{unified}} = \mathcal{T}_{\text{train\_queries}} \cup \mathcal{T}_{\text{test\_queries}} \cup \bigcup_{j \in \{A..E\}} \mathcal{T}_{\text{train\_options}, j} \cup \bigcup_{j \in \{A..E\}} \mathcal{T}_{\text{test\_options}, j}$$

---

## 4. Tokenization Strategy

### 4.1 Tokenizer Selection Across Models
Different tokenization strategies were customized for each model class to balance representational power, subword handling, and runtime execution constraints:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                TOKENIZATION PIPELINES                                 │
├──────────────────────────────┬───────────────────────────┬─────────────────────────────┤
│ Model 1 (Scratch PyTorch)    │ Model 2 (Pretrained BERT) │ Model 3 (Feature Ensemble)  │
├──────────────────────────────┼───────────────────────────┼─────────────────────────────┤
│ • Custom WordVocab           │ • WordPiece Tokenizer     │ • Dual TF-IDF Vectorizers   │
│ • Python Counter (top 10k)   │ • bert-base-uncased       │   - Word (1-2 n-grams)      │
│ • Custom regex split         │ • Subword vocabulary      │   - Char_wb (2-4 n-grams)   │
│ • Reserved <pad>=0, <unk>=1  │   (30,522 tokens)         │ • SentenceTransformers BPE  │
│                              │ • [CLS] / [SEP] markers   │   (all-MiniLM-L6-v2)        │
└──────────────────────────────┴───────────────────────────┴─────────────────────────────┘
```

1. **Custom `WordVocab` (Model 1):** Built entirely from scratch using `collections.Counter`. Words are extracted via string normalization and whitespace/punctuation split filtering. The top 10,000 most frequent tokens are retained in index mappings. Reserved tokens `<pad>` (index 0) and `<unk>` (index 1) ensure consistent sequence padding and unknown word handling.
2. **WordPiece Tokenizer (Model 2):** Utilizes `AutoTokenizer.from_pretrained('bert-base-uncased')`. Text pairs $(P_i \oplus C_i, O_{i,j})$ are encoded into candidate input sequences formatted as `[CLS] Prompt + Context [SEP] Option Text [SEP]`, bounded by maximum sequence length $L_{\max} = 128$. Attention masks distinguish actual tokens (1) from padding tokens (0).
3. **Dual TF-IDF & SentenceTransformer Tokenizer (Model 3):**
   * *Word TF-IDF Vectorizer:* 1–2 n-grams capturing specific keyword pairs (max features: 10,000, sublinear TF scaling).
   * *Character-wb TF-IDF Vectorizer:* 2–4 character n-grams bounded by word boundaries (`char_wb`), preserving morphological prefixes/suffixes and technical acronyms (max features: 6,000).
   * *MiniLM Subword Tokenizer:* Hugging Face WordPiece tokenizer associated with `all-MiniLM-L6-v2` for 384-dimensional dense semantic embedding generation.

### 4.2 Tokenizer Implementation Summary Table

| Metric / Parameter | Model 1 (`WordVocab`) | Model 2 (`bert-base-uncased`) | Model 3 (`TF-IDF + MiniLM`) |
| :--- | :--- | :--- | :--- |
| **Vocabulary Size** | 10,000 words | 30,522 subwords | 16,000 features + 30,522 subwords |
| **Subword Handling** | No (Full word match) | Yes (WordPiece `##` prefix) | N-gram character boundaries + Subwords |
| **Max Sequence Length** | 128 tokens | 128 tokens | 128 tokens (MiniLM) |
| **Padding / Truncation** | Post-padding with 0 | Post-padding with `[PAD]` | Sparse matrix padding + Dense pooling |
| **Special Tokens** | `<pad>`, `<unk>` | `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`| `[CLS]`, `[SEP]` (MiniLM) |

---

## 5. Modeling & Experimentation

### 5.1 Model 1: PyTorch Transformer Built From Scratch (`base_model.ipynb`)

#### Architecture Overview
Model 1 implements a custom PyTorch Transformer Encoder (`MCQTransformerScratch`) built directly from primitive PyTorch modules without relying on Hugging Face or high-level abstractions.

#### Transformer Encoder Mathematics
The encoder's core operation is scaled dot-product self-attention. For token-state matrices $Q$, $K$, and $V$, the contextualized representation is

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right)V.
$$

Here, queries specify the context sought by each token, keys determine query-token compatibility, and values are combined using the resulting attention distribution. Dividing by $\sqrt{d_k}$ prevents large dot products from saturating the softmax and producing extremely small gradients. Multi-head self-attention uses separate learned projections per head,

$$
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O, \qquad
\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V),
$$

so distinct heads can model different token relations and representation subspaces. Each encoder layer pairs this self-attention block with a position-wise feed-forward block, with residual connections and layer normalization around both sublayers. Since attention itself is permutation-equivariant, Model 1 adds fixed sinusoidal vectors to its token embeddings:

$$
\operatorname{PE}(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
\operatorname{PE}(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).
$$

Mathematically, Model 1 learns its embedding table and all encoder projection parameters from random initialization on this task; it is therefore a from-scratch implementation of the Transformer encoder pattern rather than a transferred language model.

```
                           +------------------------+
                           | Prompt + Option Text   |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           | Custom WordVocab Index |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           | Embedding Layer (d=256)|
                           +-----------+------------+
                                       |
                                       v  <-- + Sinusoidal Positional Encoding
                           +------------------------+
                           |  TransformerEncoder    |
                           |  Layer 1 (4 heads)     |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           |  TransformerEncoder    |
                           |  Layer 2 (4 heads)     |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           | Masked Mean Pooling    |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           | MLP Classifier Head    |
                           | Linear(256->64) + GELU |
                           | Linear(64->1) Logit    |
                           +------------------------+
```

#### Key Architecture Components
1. **Embedding & Positional Encoding:** Converts token IDs to continuous representations $\mathbf{E} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$, where $d_{\text{model}} = 256$. Fixed non-learnable sinusoidal positional encodings $\mathbf{PE}$ are added:
   $$\mathbf{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \mathbf{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
2. **Transformer Encoder Stack:** 2 layers of `nn.TransformerEncoderLayer` configured with $d_{\text{model}} = 256$, Multi-Head Self-Attention heads $n_{\text{head}} = 4$, feedforward dimension $d_{\text{ff}} = 512$, and dropout rate $p = 0.1$.
3. **Masked Mean-Pooling:** Collapses the sequence length dimension by averaging hidden state representations strictly over non-padding tokens:
   $$\mathbf{h}_{\text{pooled}} = \frac{\sum_{i=1}^{L} \mathbf{h}_i \cdot M_i}{\sum_{i=1}^{L} M_i}$$
   where $M_i \in \{0, 1\}$ represents the binary attention mask value.
4. **Classification MLP Head:** Two-layer projection MLP: $\text{Linear}(256 \to 64) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(64 \to 1)$.

#### Fine-Tuning Strategy & Hyperparameters
* **Optimizer:** AdamW ($\text{lr} = 1 \times 10^{-3}$, $\text{weight\_decay} = 1 \times 10^{-2}$).
* **Loss Function:** `BCEWithLogitsLoss` operating on binary long-format targets.
* **Gradient Management:** Gradient clipping enforced at $\text{max\_norm} = 1.0$.
* **Epochs & Batch Size:** 5 epochs, batch size 32.

---

### 5.2 Model 2: Fine-Tuned Transformer Model (`pre_trained.ipynb`)

#### Architecture Overview
Model 2 fine-tunes `bert-base-uncased` (Devlin et al., 2018) instantiated via `AutoModelForMultipleChoice`. The architecture consists of 12 transformer encoder layers, 768 hidden units, 12 self-attention heads, and a total of 110M parameters.

#### Relation to the From-Scratch Encoder
BERT applies the same multi-head scaled-dot-product attention operator, but its 12-layer encoder starts with parameters pre-trained bidirectionally before MCQ fine-tuning. In contrast with Model 1's fixed sinusoidal positions, BERT forms each input state from learned token, segment, and position embeddings. Its unmasked encoder self-attention allows every token to use both left and right context. For each prompt-option sequence $j$, the final classification score is produced from the special-token state, $z_j = \mathbf{w}^{\mathsf{T}}\mathbf{h}^{(j)}_{[\mathrm{CLS}]} + b$; Model 1 instead aggregates non-padding token states with masked mean pooling before its binary option head. Thus the shared attention mathematics is the same, while the initialization, position representation, sequence pooling, and pretraining regime are materially different.

```
 Input Options (A, B, C, D, E) paired with Prompt
                       |
                       v
 [CLS] Prompt [SEP] Option_A [SEP]  -->  BERT Encoder  -->  Pooled Logit A
 [CLS] Prompt [SEP] Option_B [SEP]  -->  BERT Encoder  -->  Pooled Logit B
 [CLS] Prompt [SEP] Option_C [SEP]  -->  BERT Encoder  -->  Pooled Logit C
 [CLS] Prompt [SEP] Option_D [SEP]  -->  BERT Encoder  -->  Pooled Logit D
 [CLS] Prompt [SEP] Option_E [SEP]  -->  BERT Encoder  -->  Pooled Logit E
                       |
                       v
            Softmax / CrossEntropy Loss
```

#### Fine-Tuning Strategy & Parameters
* **Model Checkpoint:** `bert-base-uncased` (cloned locally to eliminate runtime download timeouts).
* **Classifier Head:** Linear dropout layer ($p=0.1$) mapping the `[CLS]` token state $\mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^{768}$ to a single scalar relevance logit per option pair.
* **Optimization:** AdamW optimizer with initial learning rate $\eta = 2 \times 10^{-5}$, weight decay $0.01$, and linear warmup schedule with $\text{warmup\_ratio} = 0.1$.
* **Loss Function:** Multi-class `CrossEntropyLoss` across 5 choice logits per question group:
  $$\mathcal{L}_{\text{CE}} = -\log \frac{\exp(z_{i, y_i})}{\sum_{j \in \{A..E\}} \exp(z_{i, j})}$$
* **Training Dynamics:** Model 2 uses an early-stopping cap of 3 epochs with batch size 8 per GPU to limit overfitting on the small MCQ dataset. The best model checkpoint is saved based on minimum validation loss rather than the final epoch, while MAP@3 remains a passive logged ranking metric.

---

### 5.3 Model 3: Custom Feature Engineering & Tree Ensemble (`model3.ipynb`)

#### Feature Engineering Pipeline (27 Multi-Faceted Features)
Model 3 transforms option scoring into a rich feature engineering and gradient-boosted decision tree ranking task. For every question-option pair $(i, j)$, a total of 27 numerical features are extracted across four feature subsets:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             FEATURE EXTRACTION PIPELINE                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. LEXICAL MATCHING FEATURES                                                           │
│    • word_cos     : Cosine similarity over Word TF-IDF vectors (1-2 n-grams)          │
│    • char_cos     : Cosine similarity over Char TF-IDF vectors (2-4 char_wb n-grams)   │
│    • bm25         : BM25Okapi retrieval score between query & option tokens            │
│    • bm25_norm    : BM25 score normalized to [0, 1] within question group              │
│    • jaccard      : Word set Jaccard similarity (|A ∩ B| / |A ∪ B|)                    │
│    • word_intersect: Proportion of option tokens present in query token set            │
│                                                                                        │
│ 2. DENSE SEMANTIC SIMILARITY                                                           │
│    • minilm_cos   : Cosine similarity between 384-d dense embeddings extracted via     │
│                     SentenceTransformer('all-MiniLM-L6-v2')                            │
│                                                                                        │
│ 3. STRUCTURAL LENGTH STATISTICS                                                        │
│    • opt_len, prompt_len, len_ratio (opt_len / prompt_len)                             │
│    • opt_wc, prompt_wc, wc_ratio   (opt_wc / prompt_wc)                                │
│                                                                                        │
│ 4. QUESTION-LEVEL RELATIVE RANKING TRANSFORMATIONS (CRITICAL FOR TREES)               │
│    • For each feature F in {word_cos, char_cos, bm25, bm25_norm, jaccard,              │
│                           word_intersect, minilm_cos}:                                 │
│      - {F}_diff = F - Mean(F_question)    (Deviation from question average)           │
│      - {F}_rank = Rank_Descending(F_question) (1 = Best option in question group)      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Significance of Relative Rank Features
In standard GBDT classification, absolute feature values vary across questions depending on prompt length and domain topic. Adding relative transformations ($\{F\}_{\text{rank}}$ and $\{F\}_{\text{diff}}$) renders the tree splitter invariant to question-level scale shifts: an option with $\{F\}_{\text{rank}} = 1$ is explicitly recognized as the top-scoring candidate within its question group regardless of absolute numerical magnitude.

```
Raw BM25 Values:               Per-Question Grouping:         Relative Rank Features:
Q1 Opt A: 12.4                 Q1 Group Mean: 8.2             Q1 Opt A Rank: 1  Diff: +4.2
Q1 Opt B:  4.1     ─────────>  Q1 Group Max: 12.4  ─────────> Q1 Opt B Rank: 2  Diff: -4.1
Q2 Opt A: 48.9                 Q2 Group Mean: 35.0            Q2 Opt A Rank: 1  Diff: +13.9
Q2 Opt B: 21.1                 Q2 Group Max: 48.9             Q2 Opt B Rank: 2  Diff: -13.9
```

#### 5-Fold GroupKFold Cross-Validation & Ensemble Blending
To eliminate data leakage, validation splits employ a 5-Fold `GroupKFold` strategy grouped strictly by question `id`. All 5 candidate options of any given question remain strictly inside the same fold.

Two distinct gradient-boosted decision tree architectures were trained on each fold:
1. **LightGBM Classifier (`LGBMClassifier`):** Configured with 500 trees, learning rate $\eta = 0.05$, `num_leaves` $= 63$, `colsample_bytree` $= 0.7$, `subsample` $= 0.8$, and class weighting balanced.
2. **XGBoost Classifier (`XGBClassifier`):** Configured with 400 trees, learning rate $\eta = 0.05$, `max_depth` $= 6$, `colsample_bytree` $= 0.7$, `scale_pos_weight` adjusted for distractor imbalance.

Out-Of-Fold (OOF) predictions $\mathbf{P}_{\text{LGB}}$ and $\mathbf{P}_{\text{XGB}}$ were blended using a grid-searched weighting parameter $\alpha \in [0.05, 0.95]$:

$$\mathbf{P}_{\text{blend}} = \alpha \cdot \mathbf{P}_{\text{LGB}} + (1 - \alpha) \cdot \mathbf{P}_{\text{XGB}}$$

Sweeping $\alpha$ identified an optimal blend weight of **$\alpha = 0.55$** (55% LightGBM, 45% XGBoost), maximizing OOF $\text{MAP@3}$.

---

## 6. Performance & Comparative Analysis

### 6.1 Evaluation Metrics
Models were evaluated across four core performance metrics:
1. **Mean Average Precision at 3 ($\text{MAP@3}$):** Competition primary metric measuring ranking quality of top-3 choices.
2. **Binary Cross-Entropy Loss ($\text{LogLoss}$):** Calibration quality of option relevance probabilities.
3. **Classification Accuracy:** Proportion of total option pairs correctly classified as distractor (0) vs correct answer (1).
4. **Macro F1-Score:** Unweighted harmonic mean of F1-score across positive and negative binary classes.

### 6.2 Training Dynamics & W&B Validation Tracking
Training and validation progression were monitored in real-time via W&B under project `23f2004343-t22026`.

```
Validation MAP@3 Score Progression Across Epochs / Folds:

  MAP@3
   0.80 │                                                   ┌─── Model 3 (Leaderboard: 0.74608)
   0.75 │................................................─┼── Qualification Cutoff (0.7300)
   0.70 │                                          ┌──────┘
   0.65 │                           ┌──────────────┘          ├── Model 2 (BERT: 0.74397)
   0.60 │            ┌──────────────┘
   0.55 │  ┌─────────┘                                        ├── Model 1 (Scratch: 0.5840)
   0.50 └──┴──────────┴──────────────┴──────────────┴─────────
        Epoch 1       Epoch 2        Epoch 3       Final OOF
```

### 6.3 Comprehensive Model Comparison Table

| Metric / Attribute | Model 1: Scratch PyTorch Transformer | Model 2: Fine-Tuned `bert-base-uncased` | Model 3: Custom Feature Ensemble (LGBM+XGB) |
| :--- | :--- | :--- | :--- |
| **Primary Metric ($\text{MAP@3}$)** | `0.5840` | `0.74397` (Kaggle LB, Exceeds 0.73 Cutoff) | **`0.74608`** (Kaggle LB, Exceeds 0.73 Cutoff) |
| **Validation Accuracy** | 0.4620 | 0.5980 | **0.6650** |
| **Macro F1-Score** | 0.4410 | 0.5750 | **0.6480** |
| **Validation Loss ($\text{LogLoss}$)**| 0.6120 | 0.4850 | **0.3920** |
| **Training Time (Kaggle T4)** | ~2.5 minutes | ~6.0 minutes | **~45 seconds** |
| **Parameters / Features** | 10k Vocab, $d=256$, 2 Layers | 110M Transformer Weights | 27 Engineered Features |
| **Ensemble / Fold Strategy** | Train/Val Split (85/15) | Train/Val Split (85/15) | 5-Fold `GroupKFold` ($\alpha = 0.55$) |
| **W&B Run Name** | `Model_1_Scratch_Transformer` | `Model_2_BERT_MultipleChoice` | `Model_3_MiniLM_RelativeRank_Ensemble` |

### 6.4 Discussion & Technical Insights
1. **Comparative behavior:** Model 3 achieved the highest reported Kaggle score at **0.74608 MAP@3** through its fixed dense sentence embedder (`all-MiniLM-L6-v2`), BM25 lexical signals, and explicit relative-rank features ($\{F\}_{\text{rank}}$). Model 2 achieved **0.74397 MAP@3**; its 3-epoch cap and minimum-`val_loss` checkpoint reduce overfitting risk.
2. **Value of the Scratch Transformer (Model 1):** Building `MCQTransformerScratch` from raw PyTorch validated core architectural understanding of multi-head attention, sinusoidal positional encodings, and masked mean-pooling. While its MAP@3 score (0.5840) was limited by vocabulary size and dataset scale, it provided a essential baseline for deep learning diagnostics.
3. **Computational Efficiency:** Model 3 executed full 5-fold cross-validation and feature extraction in under 45 seconds on Kaggle CPU/GPU, whereas fine-tuning BERT required multi-minute GPU execution epochs.

---

## 7. Conclusion & Future Work

### 7.1 Summary of Accomplishments
In this project, we successfully developed, evaluated, and documented an end-to-end multiple-choice question solver pipeline. Key milestones achieved include:
* Full compliance with the **Three-Model Mandate** (`base_model.ipynb`, `pre_trained.ipynb`, and `model3.ipynb`).
* Implementation of strict zero-LLM-API local inference pipelines.
* Achieving final Kaggle Leaderboard Scores of **0.74397 MAP@3** with Model 2 (Pretrained BERT) and **0.74608 MAP@3** with Model 3 (feature ensemble), both surpassing the required qualification benchmark of $0.7300$.
* Complete experiment logging across all runs in Weights & Biases (`23f2004343-t22026`).

### 7.2 Engineering Challenges & Solutions
1. **Preventing Test Set OOV Tokens:** Resolved by constructing a unified fitting corpus across train queries, test queries, and all candidate option texts prior to TF-IDF vectorization.
2. **Intra-Question Data Leakage:** Prevented by implementing `GroupKFold` cross-validation grouped strictly by question `id`, ensuring all 5 option rows of a question reside in the same validation fold.
3. **Cross-Platform Path Portability:** Implemented dynamic path resolvers checking Kaggle cloud mount points (`/kaggle/input/competitions/...`) before falling back to local relative paths (`data/train.csv`).
4. **Credential Hygiene:** Replaced embedded W&B credentials with Kaggle Secrets and `WANDB_API_KEY` environment-variable resolution, including local `.env`-provided environments; no hardcoded W&B key remains in the production notebooks.

### 7.3 Actionable Future Work
1. **Larger Semantic Backbones:** Incorporate 1024-dimensional embeddings from `deberta-v3-large` or `bge-large-en-v1.5` as additional semantic feature columns in Model 3.
2. **Pairwise Learning-to-Rank (LTR):** Transition tree objectives from binary classification (`binary:logistic`) to explicit LambdaMART pairwise ranking (`rank:pairwise`) using XGBoost LTR modules.
3. **Synthetic MCQ Augmentation:** Generate synthetic distractor options using offline local models (e.g., Mistral-7B via llama.cpp) to expand the training set scale.

---

## 8. References

1. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).** BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.
2. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.
3. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017).** LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 3146–3154.
4. **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 785–794.
5. **Reimers, N., & Gurevych, I. (2019).** Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Empirical Methods in Natural Language Processing (EMNLP)*.
6. **Robertson, S., & Zaragoza, H. (2009).** The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
7. **Biewald, L. (2020).** Experiment tracking with Weights & Biases. *Software available from wandb.com*.

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
