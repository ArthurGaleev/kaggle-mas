#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Kaggle Competition System.

Usage:
  # Default (Groq LLM):
  python main.py

  # With HuggingFace:
  python main.py llm=huggingface

  # With OpenRouter:
  python main.py llm=openrouter

  # Fast pipeline (debug):
  python main.py pipeline=fast

  # Override specific params:
  python main.py pipeline.max_feedback_loops=5 models.lightgbm.params.n_estimators=2000

Install dependencies (Colab/Kaggle):
  !pip install langgraph omegaconf hydra-core lightgbm xgboost catboost \
               optuna scikit-learn pandas numpy sentence-transformers faiss-cpu \
               matplotlib psutil openai
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("kaggle-mas.main")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _download_kaggle_data(competition: str, data_dir: str) -> bool:
    """
    Attempt to download competition data using the Kaggle CLI.

    Parameters
    ----------
    competition:
        Kaggle competition slug, e.g. ``"mws-ai-agents-2026"``.
    data_dir:
        Target directory for downloaded files.

    Returns
    -------
    bool
        ``True`` if the download succeeded, ``False`` otherwise.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if kaggle CLI is available
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        logger.warning(
            "kaggle CLI not found.  Install with: pip install kaggle\n"
            "Then place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY env vars."
        )
        return False

    cmd = [
        "kaggle", "competitions", "download",
        "-c", competition,
        "-p", str(data_path),
    ]
    logger.info("Downloading data: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("Kaggle download failed:\n%s\n%s", result.stdout, result.stderr)
            return False
        logger.info("Kaggle download complete.")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Kaggle download timed out after 300 s.")
        return False
    except Exception as exc:
        logger.error("Kaggle download error: %s", exc)
        return False


def _extract_zip(data_dir: str) -> None:
    """
    Extract any ``.zip`` files found in *data_dir* into the same directory.

    Skips files that appear to have already been extracted (based on the
    presence of a ``train.csv`` or ``test.csv``).

    Parameters
    ----------
    data_dir:
        Directory to scan for zip archives.
    """
    data_path = Path(data_dir)

    # Skip extraction if CSV files already exist
    csv_files = list(data_path.glob("*.csv"))
    if csv_files:
        logger.info("CSV files already present (%d files) — skipping extraction.", len(csv_files))
        return

    zip_files = list(data_path.glob("*.zip"))
    if not zip_files:
        logger.info("No zip files found in %s — assuming data is ready.", data_dir)
        return

    for zf_path in zip_files:
        logger.info("Extracting: %s", zf_path)
        try:
            with zipfile.ZipFile(zf_path, "r") as zf:
                zf.extractall(data_path)
            logger.info("Extracted: %s", zf_path.name)
        except Exception as exc:
            logger.error("Failed to extract %s: %s", zf_path, exc)


def _ensure_data(cfg: DictConfig) -> str:
    """
    Ensure competition data is present in *cfg.project.data_dir*.

    Strategy:
    1. If ``train.csv`` / ``test.csv`` already exist → nothing to do.
    2. Attempt Kaggle CLI download.
    3. Extract any zip archives.
    4. If still missing, log a helpful message pointing to manual upload.

    Parameters
    ----------
    cfg:
        Full Hydra config.

    Returns
    -------
    str
        Resolved path to the data directory.
    """
    data_dir    = str(OmegaConf.select(cfg, "project.data_dir",    default="./data"))
    competition = str(OmegaConf.select(cfg, "project.competition", default="mws-ai-agents-2026"))

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_csv = data_path / "train.csv"
    test_csv  = data_path / "test.csv"

    if train_csv.exists() and test_csv.exists():
        logger.info("Data already present at %s.", data_dir)
        return data_dir

    # Try Kaggle download
    downloaded = _download_kaggle_data(competition, data_dir)
    if downloaded:
        _extract_zip(data_dir)

    # Final check
    if not train_csv.exists() or not test_csv.exists():
        missing = []
        if not train_csv.exists():
            missing.append(str(train_csv))
        if not test_csv.exists():
            missing.append(str(test_csv))
        logger.warning(
            "Expected data files not found: %s\n"
            "Manual upload: place train.csv and test.csv in '%s'.\n"
            "  Colab: from google.colab import files; files.upload()\n"
            "  Kaggle notebook: competition data is mounted at /kaggle/input/",
            missing, data_dir,
        )

    return data_dir


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def _print_results_summary(final_state: Dict[str, Any]) -> None:
    """Print a human-readable results summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  MULTI-AGENT ML PIPELINE — RESULTS SUMMARY")
    print(sep)

    # Evaluation report
    eval_report = final_state.get("evaluation_report", {})
    if eval_report:
        ens = eval_report.get("ensemble", {}).get("oof_metrics", {})
        print(f"\n  Ensemble OOF metrics:")
        print(f"    MSE  : {ens.get('mse',  'N/A')}")
        print(f"    RMSE : {ens.get('rmse', 'N/A')}")
        print(f"    MAE  : {ens.get('mae',  'N/A')}")
        print(f"    R²   : {ens.get('r2',   'N/A')}")

        per_algo = eval_report.get("per_algorithm", {})
        if per_algo:
            print(f"\n  Per-algorithm OOF MSE:")
            for algo, info in per_algo.items():
                mse = info.get("oof_metrics", {}).get("mse", "N/A")
                print(f"    {algo:<20}: {mse}")

    # Orchestrator decision
    decision = final_state.get("decision", "N/A")
    iteration = final_state.get("iteration", 0)
    print(f"\n  Orchestrator decision : {decision}")
    print(f"  Feedback iterations   : {iteration}")

    reasoning = final_state.get("reasoning", "")
    if reasoning:
        print(f"  Reasoning             : {reasoning}")

    # Timings
    timings = final_state.get("agent_timings", {})
    if timings:
        print(f"\n  Agent timings (s):")
        for agent, t in sorted(timings.items(), key=lambda x: -x[1]):
            print(f"    {agent:<25}: {t:.2f}s")

    # Errors
    errors = final_state.get("errors", [])
    if errors:
        print(f"\n  Non-fatal errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Submission save
# ---------------------------------------------------------------------------

def _save_submission(final_state: Dict[str, Any], output_dir: str) -> Optional[str]:
    """
    Save ``submission_df`` from the final state to *output_dir/submission.csv*.

    Returns the file path on success, ``None`` otherwise.
    """
    submission_df = final_state.get("submission_df")
    if submission_df is None:
        logger.warning("No submission_df in final state — skipping submission save.")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "submission.csv"

    try:
        submission_df.to_csv(csv_path, index=False)
        logger.info("Submission saved to %s (%d rows).", csv_path, len(submission_df))
        print(f"  Submission saved → {csv_path}")
        return str(csv_path)
    except Exception as exc:
        logger.error("Failed to save submission: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

def _generate_dashboard(final_state: Dict[str, Any], output_dir: str) -> None:
    """
    Use :class:`~monitoring.dashboard.MetricsDashboard` to generate diagnostic plots.
    """
    try:
        from monitoring.dashboard import MetricsDashboard

        tracker = final_state.get("tracker")
        if tracker is None:
            logger.warning("No tracker in final state — skipping dashboard.")
            return

        dashboard = MetricsDashboard()

        importance_dict = final_state.get("feature_importances")
        y_true = None
        y_pred = None

        # Try to get OOF predictions for residual plot
        target_series = final_state.get("target_series")
        ensemble_oof  = final_state.get("ensemble_oof")
        if target_series is not None and ensemble_oof is not None:
            try:
                import numpy as np
                y_true = np.asarray(target_series)
                y_pred = np.asarray(ensemble_oof)
            except Exception:
                pass

        saved_plots = dashboard.generate_report(
            tracker=tracker,
            output_dir=output_dir,
            importance_dict=importance_dict,
            y_true=y_true,
            y_pred=y_pred,
        )
        print(f"  Dashboard plots saved ({len(saved_plots)} files) → {output_dir}")

    except ImportError as exc:
        logger.warning("matplotlib not available — skipping dashboard: %s", exc)
    except Exception as exc:
        logger.error("Dashboard generation failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra-powered main entry point.

    Orchestrates:
    1. Data download / extraction.
    2. Pipeline execution via :func:`~pipeline.run_pipeline`.
    3. Results summary printing.
    4. Submission CSV saving.
    5. Monitoring dashboard generation.
    """
    # Configure root logger level from config
    log_level_name = str(OmegaConf.select(cfg, "monitoring.log_level", default="INFO"))
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # 1. Ensure data is present
    # ------------------------------------------------------------------
    data_dir = _ensure_data(cfg)

    # Update config with resolved data_dir in case Hydra changed cwd
    try:
        OmegaConf.update(cfg, "project.data_dir", data_dir, merge=True)
    except Exception:
        pass

    output_dir = str(OmegaConf.select(cfg, "project.output_dir", default="./outputs"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Run pipeline
    # ------------------------------------------------------------------
    from pipeline import run_pipeline

    try:
        final_state = run_pipeline(cfg)
    except Exception as exc:
        logger.critical("Pipeline failed with unrecoverable error: %s", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Print results summary
    # ------------------------------------------------------------------
    _print_results_summary(final_state)

    # ------------------------------------------------------------------
    # 4. Save submission.csv
    # ------------------------------------------------------------------
    sub_path = _save_submission(final_state, output_dir)
    if sub_path:
        logger.info("Competition submission file: %s", sub_path)

    # ------------------------------------------------------------------
    # 5. Generate monitoring dashboard
    # ------------------------------------------------------------------
    dashboard_dir = os.path.join(output_dir, "dashboard")
    _generate_dashboard(final_state, dashboard_dir)

    # ------------------------------------------------------------------
    # 6. Save final tracker report (redundant safety save)
    # ------------------------------------------------------------------
    tracker = final_state.get("tracker")
    if tracker is not None:
        report_path = os.path.join(output_dir, "pipeline_report.json")
        tracker.save_report(report_path)
        logger.info("Final tracker report: %s", report_path)

    logger.info("main.py complete.  Output dir: %s", output_dir)


# ---------------------------------------------------------------------------
# Direct execution guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
