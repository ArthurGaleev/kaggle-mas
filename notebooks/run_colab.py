# %% [markdown]
# # Multi-Agent System for Kaggle Competition
# ## mws-ai-agents-2026: Rental Property Occupancy Prediction
#
# This notebook runs the full multi-agent pipeline on **Google Colab** (T4/A100 GPU).
#
# **Architecture**: LangGraph orchestrates 5 specialized agents:
# - `DataAgent` — data loading, profiling, and LLM-guided cleaning
# - `FeatureAgent` — LLM-planned feature engineering (datetime, geo, text, encodings)
# - `ModelAgent` — trains LightGBM, XGBoost, and CatBoost with cross-validation
# - `EvaluatorAgent` — computes MSE metrics and LLM interpretation
# - `OrchestratorAgent` — decides ACCEPT or IMPROVE in a feedback loop
#
# **LLM providers supported**: Groq (free, fast), HuggingFace Inference, OpenRouter
#
# **Estimated runtime**: ~15–30 min on Colab T4 (depends on data size and iterations)

# %% [markdown]
# ## 1. Install dependencies

# %%
# Install all project dependencies
# faiss-cpu is used for the RAG knowledge base vector search
!pip install -q \
    lightgbm xgboost catboost optuna \
    openai langgraph langchain-core \
    sentence-transformers faiss-cpu \
    hydra-core omegaconf \
    pydantic pydantic-settings \
    psutil tqdm kaggle python-dotenv

# %% [markdown]
# ## 2. Clone repository

# %%
# Replace YOUR_USERNAME with your GitHub username
!git clone https://github.com/YOUR_USERNAME/kaggle-mas.git
%cd kaggle-mas

# %% [markdown]
# ## 3. Configure API keys
#
# Choose one LLM provider:
# - **Groq** (recommended — free tier, very fast): https://console.groq.com
# - **HuggingFace**: https://huggingface.co/settings/tokens
# - **OpenRouter**: https://openrouter.ai

# %%
import os

# --- Option A: Groq (free tier, recommended) ---
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"  # ← replace with your key

# --- Option B: HuggingFace Inference API ---
# os.environ["HF_TOKEN"] = "hf_your_token_here"

# --- Option C: OpenRouter ---
# os.environ["OPENROUTER_API_KEY"] = "sk-or-your_key_here"

# %% [markdown]
# ## 4. Download competition data
#
# Choose one download method:

# %%
# --- Method 1: Kaggle API (requires Kaggle credentials) ---
# os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
# os.environ["KAGGLE_KEY"]      = "your_kaggle_key"
# !kaggle competitions download -c mws-ai-agents-2026
# !unzip -o mws-ai-agents-2026.zip -d data/
# print("Data downloaded via Kaggle API")

# %%
# --- Method 2: Google Drive (public dataset mirror) ---
!pip install -q gdown
!mkdir -p data
!gdown 1Xkag8BW9Q9phWyz1uQWyRVT311rG4tqp -O data.zip
!unzip -o data.zip -d data/
!ls data/
print("Data downloaded via Google Drive")

# %% [markdown]
# ## 5. (Optional) Inspect data

# %%
import pandas as pd

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")
print("\nTrain head:")
train.head()

# %%
print("Target statistics:")
print(train["target"].describe())

# %% [markdown]
# ## 6. Run the full pipeline
#
# The pipeline is configured via `configs/config.yaml`.
# Default LLM provider is Groq — override with `llm=huggingface` or `llm=openrouter`.

# %%
# Run with default config (Groq LLM)
!python main.py

# %%
# --- Alternatives ---

# Use HuggingFace Inference API
# !python main.py llm=huggingface

# Use OpenRouter
# !python main.py llm=openrouter

# Fast debug run (2 folds, no RAG, max 1 iteration)
# !python main.py pipeline=fast

# Custom data directory
# !python main.py project.data_dir=data/

# %% [markdown]
# ## 7. Inspect results

# %%
import json

with open("outputs/pipeline_report.json") as f:
    report = json.load(f)

print(json.dumps(report, indent=2))

# %%
# View the submission file
sub = pd.read_csv("outputs/submission.csv")
print(f"Submission shape: {sub.shape}")
print(f"Prediction range: [{sub['target'].min():.2f}, {sub['target'].max():.2f}]")
print(f"Prediction mean : {sub['target'].mean():.2f}")
print(sub.head(10))

# %% [markdown]
# ## 8. View monitoring dashboard

# %%
from monitoring.dashboard import MetricsDashboard
from monitoring.tracker import PipelineTracker
import os

os.makedirs("outputs/plots", exist_ok=True)

tracker = PipelineTracker.load("outputs/pipeline_report.json")
dashboard = MetricsDashboard()
dashboard.generate_report(tracker, "outputs/plots/")

print("Dashboard plots saved to outputs/plots/")

# %%
# Display all generated plots inline
from IPython.display import Image, display
import glob

plot_files = sorted(glob.glob("outputs/plots/*.png"))
print(f"Found {len(plot_files)} plots:")
for img_path in plot_files:
    print(f"  {img_path}")
    display(Image(filename=img_path))

# %% [markdown]
# ## 9. (Optional) Hyperparameter sweep with Optuna
#
# The ModelAgent uses Optuna internally, but you can run an explicit sweep:

# %%
# !python main.py pipeline=fast pipeline.n_optuna_trials=50

# %% [markdown]
# ## 10. Submit to Kaggle

# %%
# After verifying the submission looks correct, submit:
# !kaggle competitions submit -c mws-ai-agents-2026 -f outputs/submission.csv -m "MAS pipeline v1"

# %% [markdown]
# ---
# ## Troubleshooting
#
# | Problem | Solution |
# |---------|----------|
# | `GROQ_API_KEY not set` | Set `os.environ["GROQ_API_KEY"]` in cell 3 |
# | OOM on T4 | Run with `pipeline=fast` to reduce memory usage |
# | `data/train.csv not found` | Re-run cell 4 (download data) |
# | Low MSE score | Try `pipeline.max_feedback_loops=3` for more iterations |
# | Slow LLM calls | Switch to `llm=groq` for fastest inference |
