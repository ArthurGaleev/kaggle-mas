# %% [markdown]
# # Multi-Agent System — Kaggle Kernel Version
# ## mws-ai-agents-2026: Rental Property Occupancy Prediction
#
# This notebook is adapted for the **Kaggle Kernel** environment:
# - Data is pre-mounted at `/kaggle/input/` (no download step needed)
# - Internet access is available during training (for LLM API calls)
# - GPU: P100 (16 GB VRAM) available via Settings → Accelerator
#
# **How to use:**
# 1. Fork this notebook in the competition
# 2. Add your API key as a Kaggle Secret (Settings → Secrets → Add)
# 3. Click "Run All" — submission is written to `/kaggle/working/submission.csv`

# %% [markdown]
# ## 1. Install dependencies
#
# Kaggle kernels have many packages pre-installed; we only add what's missing.

# %%
!pip install -q \
    openai langgraph langchain-core \
    sentence-transformers faiss-cpu \
    hydra-core omegaconf \
    pydantic pydantic-settings \
    python-dotenv

# %% [markdown]
# ## 2. Clone repository

# %%
import subprocess, os

repo_url = "https://github.com/YOUR_USERNAME/kaggle-mas.git"  # ← update
result = subprocess.run(
    ["git", "clone", repo_url, "/kaggle/working/kaggle-mas"],
    capture_output=True, text=True
)
print(result.stdout or result.stderr)
os.chdir("/kaggle/working/kaggle-mas")
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# ## 3. Configure API key via Kaggle Secrets
#
# Add your Groq (or other provider) key as a Kaggle Secret:
# **Settings → Secrets → Add new secret → Name: `GROQ_API_KEY`**

# %%
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

# --- Option A: Groq (recommended — free tier, fast) ---
try:
    groq_key = secrets.get_secret("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = groq_key
    print("Groq API key loaded from Kaggle Secrets.")
except Exception:
    print("GROQ_API_KEY not found in secrets — set it in Settings → Secrets")

# --- Option B: HuggingFace ---
# try:
#     hf_token = secrets.get_secret("HF_TOKEN")
#     os.environ["HF_TOKEN"] = hf_token
#     print("HF token loaded.")
# except Exception:
#     pass

# --- Option C: OpenRouter ---
# try:
#     or_key = secrets.get_secret("OPENROUTER_API_KEY")
#     os.environ["OPENROUTER_API_KEY"] = or_key
# except Exception:
#     pass

# %% [markdown]
# ## 4. Locate competition data
#
# In a Kaggle kernel, data is pre-mounted at `/kaggle/input/<competition-slug>/`.

# %%
import os
import glob

COMPETITION_SLUG = "mws-ai-agents-2026"
DATA_ROOT = f"/kaggle/input/{COMPETITION_SLUG}"

# Verify data is accessible
csv_files = glob.glob(f"{DATA_ROOT}/*.csv")
print(f"Found {len(csv_files)} CSV files at {DATA_ROOT}:")
for f in csv_files:
    size_mb = os.path.getsize(f) / (1024 ** 2)
    print(f"  {os.path.basename(f)}  ({size_mb:.1f} MB)")

# Symlink into the project's expected data/ directory
os.makedirs("data", exist_ok=True)
for src in csv_files:
    dst = f"data/{os.path.basename(src)}"
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"Linked {src} → {dst}")

# %% [markdown]
# ## 5. (Optional) Quick data sanity check

# %%
import pandas as pd

train = pd.read_csv(f"{DATA_ROOT}/train.csv")
test  = pd.read_csv(f"{DATA_ROOT}/test.csv")

print(f"Train: {train.shape}  |  Test: {test.shape}")
print("\nTarget stats:")
print(train["target"].describe().to_string())
print("\nMissing values (train):")
missing = train.isnull().sum()
print(missing[missing > 0].to_string())

# %% [markdown]
# ## 6. Run the pipeline

# %%
# Set output directory to Kaggle working directory
os.makedirs("/kaggle/working/outputs", exist_ok=True)
os.makedirs("/kaggle/working/outputs/plots", exist_ok=True)

# Run the full pipeline
# Override data_dir and output_dir to point to Kaggle paths
!python main.py \
    project.data_dir=/kaggle/input/{COMPETITION_SLUG} \
    project.output_dir=/kaggle/working/outputs

# %%
# --- Alternatives ---

# Fast debug run (2 folds, skip RAG)
# !python main.py pipeline=fast \
#     project.data_dir=/kaggle/input/{COMPETITION_SLUG} \
#     project.output_dir=/kaggle/working/outputs

# Use HuggingFace provider
# !python main.py llm=huggingface \
#     project.data_dir=/kaggle/input/{COMPETITION_SLUG} \
#     project.output_dir=/kaggle/working/outputs

# %% [markdown]
# ## 7. Inspect results

# %%
import json

report_path = "/kaggle/working/outputs/pipeline_report.json"
with open(report_path) as f:
    report = json.load(f)

print("=== Pipeline Report ===")
print(json.dumps(report, indent=2))

# %%
# Read and verify submission
sub = pd.read_csv("/kaggle/working/outputs/submission.csv")
print(f"Submission rows : {len(sub)}")
print(f"Submission cols : {list(sub.columns)}")
print(f"Prediction range: [{sub['target'].min():.2f}, {sub['target'].max():.2f}]")
print(f"NaN count       : {sub['target'].isna().sum()}")
print(sub.head(10))

# %%
# Copy submission to Kaggle output (auto-detected for submission)
import shutil
shutil.copy(
    "/kaggle/working/outputs/submission.csv",
    "/kaggle/working/submission.csv"
)
print("submission.csv copied to /kaggle/working/ — ready to submit!")

# %% [markdown]
# ## 8. View monitoring plots

# %%
from monitoring.dashboard import MetricsDashboard
from monitoring.tracker import PipelineTracker
from IPython.display import Image, display
import glob

tracker = PipelineTracker.load(report_path)
dashboard = MetricsDashboard()
dashboard.generate_report(tracker, "/kaggle/working/outputs/plots/")

plot_files = sorted(glob.glob("/kaggle/working/outputs/plots/*.png"))
print(f"Generated {len(plot_files)} plots")
for img_path in plot_files:
    print(img_path)
    display(Image(filename=img_path))

# %% [markdown]
# ## 9. Evaluation metrics summary

# %%
if "evaluation_report" in report:
    eval_rep = report["evaluation_report"]

    ensemble = eval_rep.get("ensemble", {}).get("oof_metrics", {})
    print("=== Ensemble OOF Metrics ===")
    for k, v in ensemble.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Per-Algorithm OOF MSE ===")
    for algo, metrics in eval_rep.get("per_algorithm", {}).items():
        mse = metrics.get("oof_metrics", {}).get("mse", "N/A")
        print(f"  {algo:20s}: MSE = {mse}")

# %% [markdown]
# ---
# ## Troubleshooting (Kaggle Kernel)
#
# | Problem | Solution |
# |---------|----------|
# | `Secret GROQ_API_KEY not found` | Add secret in Settings → Secrets |
# | `No such file: data/train.csv` | Re-run cell 4 to set up symlinks |
# | Session crashes (OOM) | Enable P100 GPU; run with `pipeline=fast` |
# | LLM rate limit errors | Wait 60 s and re-run; or switch to `llm=huggingface` |
# | Submission file is empty | Check `/kaggle/working/outputs/pipeline_report.json` for errors |
