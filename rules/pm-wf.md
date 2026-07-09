# 🏗️ DL-GENAI PROJECT: RAG & 3-MODEL DEVELOPMENT WORKFLOW

## OVERVIEW: THE RAG MCQ PIPELINE
The objective of this project is to answer Multiple Choice Questions (A, B, C, D, E) by retrieving relevant context and scoring the probability of each answer. 
- **Metric:** Mean Average Precision @ 3 (MAP@3). The model must output the top 3 most likely answers.
- **Input Format:** `[Retrieved Context] + [Question] + [Option]`
- **Output:** A probability score for that specific option.

---

## PHASE 1: DATA PIPELINE & RETRIEVAL (Shared Across All Models)
Before building the models, the agent must construct a robust data loader and context retriever.
1. **Document Chunking:** Parse the provided raw text/Wikipedia data into manageable overlapping chunks.
2. **Vectorization:** Use a lightweight embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to convert chunks into vector embeddings.
3. **Indexing:** Build a FAISS (Facebook AI Similarity Search) or simple Cosine Similarity index.
4. **Retrieval Function:** Given an MCQ prompt, retrieve the top-K most relevant text chunks to serve as context.

---

## PHASE 2: MODEL 1 - "FROM SCRATCH" (The PyTorch Baseline)
**Constraint:** NO pre-trained weights. NO `transformers` library imports for the model architecture. 
1. **Architecture Definition:** Write a custom `nn.Module` in PyTorch. 
   - *Example:* An `nn.Embedding` layer followed by an `nn.LSTM` or `nn.GRU`, feeding into a fully connected `nn.Linear` classification head.
2. **Input Processing:** Tokenize the `Context + Question + Option` using a standard vocabulary mapping (e.g., TF-IDF or a custom word-to-index dictionary).
3. **Training Loop:** - Initialize `wandb.init(project="DL-GenAI-Project", name="Model_1_Scratch")`.
   - Train using CrossEntropyLoss (or Binary Cross Entropy per option).
   - Log epoch loss and validation MAP@3 to Weights & Biases.

---

## PHASE 3: MODEL 2 - "PRE-TRAINED" (Hugging Face Integration)
**Constraint:** Must utilize downloaded open-source weights via the Hugging Face `transformers` library. No external API calls.
1. **Model Selection:** Load a lightweight, robust transformer designed for classification.
   - *Recommendation:* `distilbert-base-uncased` or `deberta-v3-small`.
2. **Implementation:** - Load the tokenizer: `AutoTokenizer.from_pretrained('distilbert-base-uncased')`.
   - Load the model: `AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)`.
3. **Fine-Tuning:**
   - Pass the tokenized `[CLS] Context [SEP] Question + Option [SEP]` into the model.
   - Use the Hugging Face `Trainer` API or a custom PyTorch training loop.
   - Wrap the run in `wandb.init(name="Model_2_Pretrained")`.

---

## PHASE 4: MODEL 3 - "CUSTOM EXPERIMENT" (The Wildcard)
**Constraint:** Must be structurally different from Models 1 and 2. 
1. **Strategy A (The ML Ensemble):** Extract embeddings from the `Context + Question + Option` using a pre-trained sentence transformer, then feed those embeddings into a **LightGBM** or **XGBoost** ranking model. (Use `wandb.sklearn` for tracking).
2. **Strategy B (The Advanced DL Hybrid):** Combine a Convolutional Neural Network (1D-CNN) for local feature extraction with an LSTM for sequential context. 
3. **Execution:** Train, validate, log metrics to W&B, and generate the final predictions.

---

## PHASE 5: KAGGLE EXECUTION & SUBMISSION
1. **Notebook Isolation:** Ensure Model 1, Model 2, and Model 3 are saved as three distinct `.ipynb` files.
2. **Deployment:** Use `kaggle kernels push` (or manual upload) to execute the notebooks on Kaggle GPUs (T4x2).
3. **Inference & CSV Generation:**
   - Run the trained model against `test.csv`.
   - Format the output precisely: `id, prediction` (e.g., `id_123, A C B`).
   - Submit the `.csv` from the highest-performing model to the Kaggle leaderboard.

- Development files are nested: `nb/pretrained/pre_trained.ipynb`
- Relative execution root for datasets: Look for `../../data/` during local CPU smoke tests.