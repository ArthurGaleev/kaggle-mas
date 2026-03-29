# Architecture — Multi-Agent System for Kaggle Rental Property Regression

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Agent Descriptions](#2-agent-descriptions)
3. [Communication Protocol — LangGraph State Flow](#3-communication-protocol--langgraph-state-flow)
4. [RAG Integration](#4-rag-integration)
5. [Guardrails and Safety Mechanisms](#5-guardrails-and-safety-mechanisms)
6. [Feedback Loop Design](#6-feedback-loop-design)
7. [LLM Provider Configuration](#7-llm-provider-configuration)
8. [Model Selection Rationale](#8-model-selection-rationale)
9. [Memory Optimization Strategies](#9-memory-optimization-strategies)

---

## 1. System Overview

The pipeline is a **multi-agent ML system** designed for the `mws-ai-agents-2026` Kaggle competition (rental property price regression, MSE metric).

Five specialized agents are wired together using [LangGraph](https://github.com/langchain-ai/langgraph) into a directed graph with a conditional feedback loop. Each agent has a **deterministic execution path** (pure Python/pandas/sklearn) and an **LLM reasoning layer** that guides non-trivial decisions (what to clean, which features to build, when to stop iterating).

```
┌────────────────────────────────────────────────────────────────────────┐
│                       LangGraph Pipeline                               │
│                                                                        │
│  START                                                                 │
│    │                                                                   │
│    ▼                                                                   │
│  ┌──────────────────┐                                                  │
│  │  [RAG: DataCtx]  │  ← retrieves Kaggle rental data best practices   │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │   DataAgent      │  profile → LLM cleaning plan → execute           │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │  [InputGuard]    │  validate rows, dtypes, target column            │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ [RAG: FeatCtx]   │  ← retrieves feature engineering papers          │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │  FeatureAgent    │  LLM feature plan → execute (datetime/geo/text)  │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ [FeatureGuard]   │  check feature count, NaN, inf, constants        │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ [RAG: ModelCtx]  │  ← retrieves GBDT tuning recipes                 │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │   ModelAgent     │  LightGBM + XGBoost + CatBoost + Optuna          │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ [RAG: EvalCtx]   │  ← retrieves Kaggle evaluation guidance          │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │  EvaluatorAgent  │  OOF MSE + RMSE + MAE; LLM interpretation        │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │  [OutputGuard]   │  validate predictions (NaN, range, constant)     │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ [RAG: OrchCtx]   │  ← retrieves decision best practices             │
│  └────────┬─────────┘                                                  │
│           ▼                                                            │
│  ┌──────────────────┐                                                  │
│  │ OrchestratorAgent│ ←─────────────────────────────────────────────┐  │
│  └────────┬─────────┘                                               │  │
│   ACCEPT  │    IMPROVE                                              │  │
│           │       └── next_agent = feature|model|data_agent ────────┘  │
│           ▼                                                            │
│         END                                                            │
└────────────────────────────────────────────────────────────────────────┘
```

**Key design principles:**

- **Separation of concerns**: each agent owns one phase; no agent reaches into another's logic.
- **LLM as a planner, not an executor**: the LLM outputs JSON plans; deterministic Python executes them. This ensures reproducibility and testability.
- **Graceful degradation**: if the LLM call fails or returns invalid JSON, every agent falls back to a hard-coded sensible default plan.
- **Stateless nodes**: each LangGraph node receives the full `PipelineState` dict and returns an updated copy. No shared mutable state.

---

## 2. Agent Descriptions

### 2.1 DataAgent (`agents/data_agent.py`)

**Responsibilities:**
- Load `train.csv` and `test.csv` from `cfg.project.data_dir`.
- Compute a comprehensive **data profile** (shape, dtypes, null counts, descriptive stats, skewness, cardinality, duplicate rows).
- Ask the LLM to produce a **JSON cleaning plan** (columns to drop, imputation strategies, outlier clipping, type conversions).
- Execute the cleaning plan deterministically (never drop `_id` or `target`).

**Key methods:**
| Method | Description |
|---|---|
| `profile_data(state)` | Loads CSVs and builds the profile dict |
| `clean_data(state)` | Calls LLM for plan; executes via `_execute_cleaning_plan` |
| `_compute_profile(train, test)` | Pure function → JSON-serialisable dict |
| `_execute_cleaning_plan(train, test, plan)` | Applies type casts, drops, imputation, clipping |

**State keys produced:** `train_df`, `test_df`, `data_profile`, `cleaning_plan`

---

### 2.2 FeatureAgent (`agents/feature_agent.py`)

**Responsibilities:**
- Ask the LLM which **feature groups** to activate and with what parameters.
- Execute each enabled group deterministically:
  - **Datetime features**: year, month, day-of-week, quarter, days-since-review from `last_dt` (relative to fixed `REFERENCE_DATE = 2026-01-01` for reproducibility)
  - **Geo features**: KMeans clusters on (lat, lon) + haversine distance to city centroid + distance to assigned cluster centroid
  - **Text features**: TF-IDF + TruncatedSVD on `name` and `location` columns
  - **Target encoding**: Flag only — actual smoothed mean encoding is deferred to ModelAgent (applied inside each CV fold to prevent leakage)
  - **Frequency encoding**: Relative frequency of each category value (computed on full train set; minor pre-CV leak, see module docstring)
  - **Interaction features**: Pairwise products of selected numeric pairs
  - **Ratio features**: Pairwise ratios of selected numeric pairs
  - **Log transforms**: `log1p` of right-skewed numeric columns
  - **Label encoding**: Ordinal encoding for categoricals (fit on train, unseen test categories get -1)
- **Scaling deliberately omitted**: GBDT models are invariant to monotonic feature transformations, and fitting a scaler before CV splits leaks validation-fold statistics.
- On feedback iterations, prunes low-importance features (< 1% of max importance) using ensemble-weighted importances from the previous iteration.
- Hard guardrail: never exceeds `MAX_FEATURES = 500` columns.

**Key methods:**
| Method | Description |
|---|---|
| `execute(state)` | Orchestrates all feature groups |
| `_add_datetime_features(train, test)` | Temporal signal extraction (fixed reference date) |
| `_add_geo_features(train, test, n_clusters)` | KMeans + haversine + cluster centroid distance |
| `_add_text_features(train, test, n_components)` | TF-IDF + TruncatedSVD |
| `_add_frequency_encoding(train, test)` | Relative frequency encoding |
| `_add_interaction_features(train, test, pairs)` | Pairwise product features |
| `_add_ratio_features(train, test, pairs)` | Pairwise ratio features |
| `_add_log_transforms(train, test, columns)` | log1p of skewed columns |
| `_add_label_encoding(train, test, columns)` | Ordinal label encoding |

**State keys produced:** `train_feat`, `test_feat`, `feature_names`, `target_series`, `test_ids`, `feature_plan`

---

### 2.3 ModelAgent (`agents/model_agent.py`)

**Responsibilities:**
- Ask the LLM for a model plan: which algorithms to enable, Optuna trial counts, and search-space adjustments.
- Run **Optuna TPE hyperparameter search** for each enabled algorithm using a disjoint KFold (seed + 7919) to avoid optimistic bias. Each algorithm uses a unique TPE sampler seed to explore different parameter regions.
- Run **K-fold cross-validation** (default 5 folds, regular `KFold` — not stratified, appropriate for regression) for each enabled model.
- Apply **fold-level target encoding** inside each CV fold (smoothed mean encoding using only training fold labels) to prevent leakage.
- Train LightGBM, XGBoost, and CatBoost with early stopping. GPU acceleration is handled by `ModelTools` with automatic fallback to CPU.
- Build a **stacking ensemble**: attempts RidgeCV meta-learner first, falls back to inverse-MSE weighted average if stacking weights are unstable (any weight < -0.2).
- Clip predictions to `[0, 99.9th percentile × 3.0]` to block negative and absurd outlier predictions.
- Generate out-of-fold (OOF) and test predictions.
- Produce feature importance rankings.

**State keys produced:** `models`, `oof_predictions`, `cv_scores`, `feature_importances`, `ensemble_weights`, `ensemble_oof`, `test_predictions`, `ensemble_test`, `submission_df`, `ensemble_cv_mse`, `model_plan`, `use_log_target`

---

### 2.4 EvaluatorAgent (`agents/evaluator_agent.py`)

**Responsibilities:**
- Compute structured evaluation metrics: OOF MSE, RMSE, MAE, R², and per-fold CV scores for each model and the ensemble.
- **CV stability analysis**: mean, std, and coefficient of variation of per-fold MSEs.
- **Residual analysis**: skewness, kurtosis, Shapiro-Wilk normality test, percentile statistics.
- **Overfitting check**: computes train MSE by averaging predictions from all fold models on the full training set; flags overfitting when OOF/train MSE ratio > 1.5.
- **Feature importance analysis**: weighted top-K features across algorithms.
- Ask the LLM to produce a **natural-language interpretation** of the results, including identified weaknesses and improvement recommendations.

**State keys produced:** `evaluation_report`, `llm_interpretation`

---

### 2.5 OrchestratorAgent (`agents/orchestrator.py`)

**Responsibilities:**
- Review `evaluation_report` and `llm_interpretation` from EvaluatorAgent.
- **Adaptive early stop**: if relative MSE improvement from the previous iteration is < 1%, force ACCEPT regardless of the LLM’s recommendation.
- Ask the LLM to decide **ACCEPT** (done) or **IMPROVE** (continue).
- If IMPROVE: specify which agent to route back to (`feature_agent`, `model_agent`, or `data_agent`) and provide an `improvement_plan` dict with targeted suggestions.
- Clear stale downstream results when routing back to an earlier agent (prevents stale data from bleeding into re-evaluation).
- Enforce `cfg.max_iterations` hard cap to prevent infinite loops.
- Maintain a `decision_history` list for transparency.

**Routing logic:**
```
pipeline_complete=True → END
next_agent="feature_agent" → feature_rag node → FeatureAgent → ModelAgent → Evaluator
next_agent="model_agent"   → model_rag node   → ModelAgent → Evaluator
next_agent="data_agent"    → data_rag node    → DataAgent → FeatureAgent → ModelAgent → Evaluator
```

**State keys produced/updated:** `decision`, `reasoning`, `next_agent`, `improvement_plan`, `iteration`, `decision_history`, `pipeline_complete`

---

## 3. Communication Protocol — LangGraph State Flow

All inter-agent communication happens through a single **`PipelineState`** TypedDict that is passed by value through every node. No agent holds references to other agents.

```python
class PipelineState(TypedDict, total=False):
    # Bookkeeping
    config: Any                          # OmegaConf DictConfig
    data_dir: str
    output_dir: str
    iteration: int
    pipeline_complete: bool
    decision: str                        # "ACCEPT" | "IMPROVE"
    next_agent: Optional[str]
    reasoning: str                       # orchestrator reasoning text
    decision_history: List[Dict]
    improvement_plan: Dict
    agent_timings: Dict[str, float]      # per-agent elapsed seconds
    errors: List[str]

    # Data phase
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    data_profile: Dict
    cleaning_plan: Dict
    data_validation_issues: List[str]    # issues from InputValidator

    # Feature phase
    train_feat: pd.DataFrame
    test_feat: pd.DataFrame
    feature_names: List[str]
    target_series: pd.Series
    test_ids: pd.Series                  # _id column from test
    feature_plan: Dict
    feature_validation_issues: List[str] # issues from InputValidator (features)

    # Model phase
    models: Dict
    oof_predictions: Dict                # per-algorithm OOF predictions
    cv_scores: Dict                      # per-fold metrics by model name
    feature_importances: Dict
    ensemble_weights: Dict
    ensemble_oof: np.ndarray
    test_predictions: Dict
    ensemble_test: np.ndarray
    submission_df: pd.DataFrame
    ensemble_cv_mse: float
    model_plan: Dict                     # LLM-generated model training plan
    use_log_target: bool                 # always False (disabled)

    # Evaluation phase
    evaluation_report: Dict
    llm_interpretation: str
    output_validation_issues: List[str]  # issues from OutputValidator

    # RAG context (injected before each agent)
    rag_context_data: str
    rag_context_feature: str
    rag_context_model: str
    rag_context_evaluator: str
    rag_context_orchestrator: str
```

**Message format between LLM and agents:**

Every LLM call returns a **JSON object** with a pre-defined schema. The agent's `_ask_llm_json()` method:
1. Sends a structured prompt with schema documentation.
2. Strips Markdown code fences from the response.
3. Attempts `json.loads()`.
4. Falls back to the `default=` dict on parse failure.
5. Logs warnings on fallback for observability.

---

## 4. RAG Integration

The RAG (Retrieval-Augmented Generation) system augments each agent's LLM prompt with relevant domain knowledge retrieved from a curated knowledge base.

### Knowledge Base (`rag/knowledge_base.py`)

Built from three source types:
- **Competition-specific documents**: competition description, data dictionary, sample notebooks
- **Domain papers**: AutoKaggle (Liu et al., 2024), rental market literature, feature engineering surveys
- **Code recipes**: sklearn preprocessing patterns, LightGBM/XGBoost tuning guides

Documents are split into 512-token chunks with 50-token overlap, embedded with `sentence-transformers/all-MiniLM-L6-v2`, and indexed in a FAISS flat-L2 index.

### Retriever (`rag/retriever.py`)

```
User query (agent name + data profile)
         ↓
  Embed query with all-MiniLM-L6-v2
         ↓
  FAISS top-k search (default k=5)
         ↓
  Rerank by relevance score
         ↓
  Format as context string: "CONTEXT:\n[chunk1]\n[chunk2]..."
         ↓
  Injected into agent's system prompt via rag_context_* state key
```

**Context injection per agent:**
| Agent | Query used for retrieval |
|---|---|
| DataAgent | "rental property data cleaning missing values" + profile summary |
| FeatureAgent | "rental price feature engineering geospatial text" + profile |
| ModelAgent | "LightGBM XGBoost CatBoost hyperparameter tuning regression" |
| EvaluatorAgent | "Kaggle competition evaluation MSE improvement strategies" |
| OrchestratorAgent | "machine learning pipeline iteration strategy acceptance" |

RAG can be disabled with `pipeline.enable_rag: false` for faster debug runs.

---

## 5. Guardrails and Safety Mechanisms

Three guardrail modules protect the pipeline from data quality issues, unsafe LLM outputs, and resource exhaustion.

### 5.1 InputValidator (`guardrails/input_validator.py`)

Validates DataFrames at two checkpoints:

**Dataset validation** (after DataAgent):
- Row/column count within configured limits (`max_rows`, `max_cols`)
- No completely empty columns
- Target column present and numeric
- No duplicate IDs
- In-memory size within `max_file_size_mb`

**Feature matrix validation** (after FeatureAgent):
- Feature count ≤ `max_features`
- No constant (zero-variance) features
- No features with >95% missing values
- No duplicate column names
- No infinite values

### 5.2 OutputValidator (`guardrails/output_validator.py`)

Validates model outputs:
- No NaN or infinite predictions
- Predictions within `[pred_min, pred_max]`
- Prediction length matches expected test set size
- Predictions are not suspiciously constant (std > threshold)

**Submission validation** (against sample_submission.csv):
- Column names match exactly
- Row count matches
- IDs match the sample submission
- No NaN in the target column

### 5.3 SafetyGuard (`guardrails/safety.py`)

Sanitizes all LLM responses before they influence pipeline behavior:

**Blocked patterns (replaced with `[BLOCKED:...]`):**
- Code execution: `os.system()`, `subprocess.*`, `eval()`, `exec()`, `__import__()`, `compile()`
- File deletion: `os.remove()`, `os.unlink()`, `shutil.rmtree()`, `Path.unlink()`
- Write operations: `open(..., 'w')`
- Prompt injection: "ignore all previous instructions", "disregard previous instructions", "system prompt", DAN jailbreak patterns

**Resource checks:**
- Available RAM vs. `min_free_ram_mb` threshold (default 500 MB)
- Available disk space vs. `min_free_disk_mb` threshold (default 1000 MB)
- Reads via `psutil` (preferred) or `/proc/meminfo` (fallback)

**Config validation:**
- Required keys present (`target_col`)
- Numeric parameters in sane ranges (`n_folds` in [2,20], `max_rows` in [1, 10M])
- No shell metacharacters in string values

---

## 6. Feedback Loop Design

```
EvaluatorAgent produces:
  evaluation_report = {
    "ensemble": {"oof_metrics": {"mse": X, "rmse": Y, ...}},
    "per_algorithm": {...}
  }
  llm_interpretation = "text narrative"

          ↓

OrchestratorAgent asks LLM:
  "Current MSE: X. Threshold: T. History: [...].
   Decide ACCEPT or IMPROVE. If IMPROVE, specify
   next_agent and improvement_plan as JSON."

          ↓

  Decision = ACCEPT → pipeline_complete = True → END

  Decision = IMPROVE →
    state["next_agent"] = "feature_agent" | "model_agent"
    state["improvement_plan"] = {
      "target_agent": "feature_agent",
      "suggestions": ["try geo features", "increase n_clusters"],
      "reasoning": "..."
    }
    → route back to RAG node → re-run agent with plan in state
```

**Improvement plan consumption:**

Each agent checks `state.get("improvement_plan", {})` at the start of its execution. If the plan targets this agent (e.g., `target_agent == "feature_agent"`), the LLM prompt is augmented with the improvement suggestions, biasing the next plan toward the recommended changes.

**Iteration cap:**

The orchestrator enforces `cfg.max_iterations` (default 3). After the cap is reached, it forces ACCEPT regardless of the LLM's recommendation. This prevents runaway API costs and ensures the pipeline always terminates.

**Decision history:**

Every decision is appended to `state["decision_history"]`:
```json
{
  "iteration": 1,
  "decision": "IMPROVE",
  "next_agent": "feature_agent",
  "reasoning": "MSE 145000 > threshold 150000; geo features not yet added",
  "metrics": {"ensemble_mse": 145000, "best_algo": "lgbm"}
}
```

---

## 7. LLM Provider Configuration

All providers are accessed via an **OpenAI-compatible API** (`openai` Python SDK). The `LLMClient` (`utils/llm_client.py`) wraps provider differences.

### Supported Providers

| Provider | Config file | Model (default) | Speed | Cost |
|---|---|---|---|---|
| **OpenRouter** | `configs/llm/openrouter.yaml` | `nvidia/nemotron-3-super-120b-a12b:free` | Fast | Free (default) |
| **Groq** | `configs/llm/groq.yaml` | `llama-3.3-70b-versatile` | Very fast | Free tier |
| **HuggingFace** | `configs/llm/huggingface.yaml` | `Qwen/Qwen2.5-72B-Instruct` | Moderate | Free tier |
| **Local** | `configs/llm/local_hf.yaml` | `meta-llama/Llama-3.3-70B-Instruct` | Moderate | Free (on-device) |

### Switching Providers

Hydra config override on the command line:

```bash
python main.py llm=groq
python main.py llm=huggingface
python main.py llm=local_hf
python main.py llm=local_hf llm.model=Qwen/Qwen3-80B-A3B-Instruct
```

Or set in `configs/config.yaml`:
```yaml
defaults:
  - llm: openrouter  # change to groq, huggingface, or local_hf
```

### Local Provider

The `local` provider loads models directly on Kaggle’s 2×T4 GPUs using `transformers` + `bitsandbytes` 4-bit NF4 quantization and HuggingFace Accelerate’s `device_map="auto"` for automatic multi-GPU layer splitting. Memory budget per GPU is capped at 14 GiB to leave headroom for activations. Flash Attention is disabled by default (T4 is Turing architecture, not Ampere+).

### LLMClient Features

- **Retry with exponential backoff**: 3 retries, base delay 1 s, max delay 30 s
- **Fallback model**: if the primary model fails, retries with `fallback_model`
- **Token usage tracking**: logs prompt/completion token counts per call via `TokenTracker`
- **Structured JSON output**: passes `response_format={"type": "json_object"}` when the provider supports it
- **Thread-safe**: uses a per-instance lock for token counter updates

### API Key Configuration

Set via environment variable (`.env` file or shell export):
```bash
export OPENROUTER_API_KEY="sk-or-..." # for OpenRouter (default)
export GROQ_API_KEY="gsk_..."        # for Groq
export HF_TOKEN="hf_..."             # for HuggingFace and Local (gated model downloads)
```

---

## 8. Model Selection Rationale

### Why gradient-boosted decision trees?

Rental price regression is a **tabular regression problem** with:
- Mixed feature types (numeric, categorical, datetime, text-derived)
- Moderate dataset size (typical Kaggle competition: 10K–500K rows)
- Interpretability requirements (feature importances for competition analysis)

Deep learning methods (MLP, TabNet, NODE) rarely outperform well-tuned GBDT on tabular data of this scale ([Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815)).

### LightGBM

- **Strength**: fastest training among the three; excellent on high-cardinality categoricals via `cat_feature` support; leaf-wise tree growth captures complex interactions
- **Configuration**: `num_leaves=127`, `min_child_samples=20`, `learning_rate=0.03`, early stopping at 100 rounds (config defaults; LLM plan may adjust)
- **GPU**: `device='gpu'` + OpenCL ICD auto-setup; multi-GPU via `num_gpu` when >1 GPU detected
- **Use case**: primary model; also the fastest for Optuna sweeps

### XGBoost

- **Strength**: level-wise tree growth tends to generalize better on noisier data; strong regularization (`reg_alpha`, `reg_lambda`) prevents overfitting
- **Configuration**: `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.7`, early stopping at 100 rounds
- **GPU**: `device='cuda'` (XGBoost ≥2.0) with `tree_method='hist'`; multi-GPU via `n_gpus=-1`
- **Use case**: diversity in the ensemble (different inductive bias from LightGBM)

### CatBoost

- **Strength**: native ordered target encoding avoids target leakage; handles raw categorical columns without preprocessing; often best on text-heavy or high-cardinality datasets
- **Configuration**: `depth=8`, `l2_leaf_reg=3`, `verbose=200`, early stopping at 100 rounds
- **GPU**: `task_type='GPU'`; multi-GPU via `devices='0:N-1'` range notation
- **Use case**: handles `host_name`, `location_cluster`, `type_house` natively; adds orthogonal diversity to the ensemble

### Ensemble Strategy

The ensemble uses a **two-tier strategy**:

1. **RidgeCV stacking** (primary): When ≥2 algorithms are enabled, a RidgeCV meta-learner is fit on the OOF prediction matrix. Alpha is selected from `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]` via 5-fold CV. If any weight is strongly negative (< -0.2), stacking is considered unstable and the fallback is used.

2. **Inverse-MSE weighted average** (fallback): Weights are computed from inverse OOF MSE:
```python
weights = {algo: 1/mse for algo, mse in oof_mse.items()}
total = sum(weights.values())
weights = {algo: w/total for algo, w in weights.items()}
```

Final predictions are clipped to `[0, 99.9th percentile of target × 3.0]` to block negative and absurd outlier values.

### Optuna Integration

Optuna TPE hyperparameter search is run for **each** enabled algorithm (trial counts come from the LLM model plan; defaults: LightGBM 12, XGBoost 10, CatBoost 6).

**Key design decisions:**
- **Disjoint tuning fold**: tuning uses a KFold with `random_state = seed + 7919` so the tuning validation fold never coincides with any CV validation fold, eliminating optimistic bias.
- **Per-algorithm TPE seeds**: `lgbm=seed`, `xgb=seed+1`, `catboost=seed+2` ensure each algorithm explores a different initial region.
- **Limited tuning folds**: only the first 2 folds per trial are used (`_MAX_TUNE_FOLDS = 2`) to limit tuning time.

The search space covers:
- `learning_rate`: [0.01, 0.3] log-scale
- `max_depth` / `num_leaves` / `depth`: integer ranges
- `subsample` / `feature_fraction` / `colsample_bytree`: [0.5, 1.0]
- `reg_alpha`, `reg_lambda` / `l2_leaf_reg`: [1e-8, 10.0] log-scale
- `min_child_samples`: [20, 200] (LightGBM only)

---

## 9. Memory Optimization Strategies

Running on Colab T4 (15 GB RAM) or Kaggle P100 (13 GB RAM) requires careful memory management.

### DataFrame Downcasting

After loading CSVs, DataAgent applies dtype downcasting:
- `int64` → `int32` (halves integer memory)
- `float64` → `float32` (halves float memory, negligible precision loss for tree models)

Expected saving: ~40–50% for a 200K-row dataset.

### Feature Matrix Garbage Collection

After each feature group is built, temporary intermediate DataFrames are deleted and `gc.collect()` is called. FeatureAgent does this after:
- TF-IDF matrix construction (can be large)
- KMeans fitting (cluster assignment arrays)

### Sparse Matrix Handling

TF-IDF produces sparse matrices. The SVD step (`TruncatedSVD`) consumes the sparse matrix directly without densifying, reducing peak memory usage.

### In-Process Chunk Processing for Large Datasets

If `train.shape[0] > 200_000`, DataAgent can be configured to load data in chunks (via `pandas.read_csv(chunksize=...)`), profile each chunk, and concatenate only after profiling. Controlled by `pipeline.chunk_size` config key.

### LightGBM/XGBoost Dataset Objects

Training data is converted to `lgb.Dataset` / `xgb.DMatrix` **once** before the cross-validation loop and re-used across folds via `reference=` parameter. This avoids re-creating the dataset structure on every fold.

### CatBoost Pool

Similarly, `catboost.Pool` is constructed once with explicit `cat_features` parameter to avoid internal re-encoding on every fold.

### Early Stopping

All models use early stopping (50 rounds) to avoid training past the point of diminishing returns. This reduces peak training time and memory by 30–60% compared to training for the full `n_estimators`.

### Memory Monitoring

`PipelineTracker.log_memory_snapshot(tag)` records RSS (resident set size) before and after each agent runs. If memory grows unexpectedly between phases, the tracker report highlights the culprit agent.

Threshold alert: if available RAM drops below `guardrails.min_free_ram_mb` (default 500 MB), `SafetyGuard.check_resource_limits()` logs a WARNING. The pipeline does not abort but reduces `n_folds` in the next iteration if the orchestrator detects the warning in the tracker report.
