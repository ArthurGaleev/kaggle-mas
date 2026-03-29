"""
Local HuggingFace model client for large models on Kaggle's 2×T4 (30 GB VRAM total).

Strategy
--------
* Uses ``transformers`` + ``bitsandbytes`` for 4-bit NF4 quantization so that
  models like Llama-3.3-70B or Qwen3-80B-A3B fit within 28–30 GB.
* Uses ``device_map="auto"`` (HuggingFace Accelerate) to automatically split
  model layers across both T4 GPUs — no manual CUDA device management needed.
* MoE models like Qwen3-80B-A3B have only ~3B *active* parameters per forward
  pass, so they load in ~10–12 GB at 4-bit and fit on a single T4 to spare.

Memory budget on Kaggle 2×T4
----------------------------
  GPU 0 (T4):  15 GB  — capped to 14 GiB in config (1 GB headroom)
  GPU 1 (T4):  15 GB  — capped to 14 GiB in config
  CPU RAM:     32 GiB — overflow buffer for Accelerate
  Total VRAM:  ~28 GiB usable

Model size estimates (NF4 + double-quant)
-----------------------------------------
  Llama-3.3-70B-Instruct  :  ~28–30 GB  → needs both T4s (tight)
  Qwen3-80B-A3B-Instruct  :  ~10–12 GB  → easily fits on one T4
  Qwen2.5-72B-Instruct    :  ~28–30 GB  → needs both T4s (tight)
  Qwen2.5-7B-Instruct     :  ~4  GB     → fits on one T4 in FP16

CLI usage
---------
  # Default model from local_hf.yaml:
  python main.py llm=local_hf

  # Override model at runtime:
  python main.py llm=local_hf llm.model=meta-llama/Llama-3.3-70B-Instruct
  python main.py llm=local_hf llm.model=Qwen/Qwen3-80B-A3B-Instruct
  python main.py llm=local_hf llm.model=Qwen/Qwen2.5-72B-Instruct

Install (Kaggle notebook cell)
------------------------------
  !pip install -q transformers>=4.40.0 accelerate>=0.30.0 bitsandbytes>=0.43.0
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kaggle-mas.local_llm_client")

# ---------------------------------------------------------------------------
# Lazy imports — fail gracefully if deps not installed
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def _require_deps() -> None:
    """Raise a helpful ImportError if required packages are missing."""
    missing = []
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers>=4.40.0")
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if missing:
        raise ImportError(
            "Missing packages for local model loading: " + ", ".join(missing) + "\n"
            "Install with:\n"
            "  pip install transformers>=4.40.0 accelerate>=0.30.0 bitsandbytes>=0.43.0"
        )


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------

def log_gpu_info() -> None:
    """Log per-GPU free/total VRAM — useful to debug OOM errors before loading."""
    if not _TORCH_AVAILABLE:
        logger.warning("PyTorch not available — cannot inspect GPU memory.")
        return
    n = torch.cuda.device_count()
    if n == 0:
        logger.warning(
            "No CUDA GPUs detected. The model will run on CPU — expect very slow inference."
        )
        return
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        free_b, total_b = torch.cuda.mem_get_info(i)
        logger.info(
            "GPU %d: %-20s | total=%5.1f GB | free=%5.1f GB",
            i, props.name, total_b / 1e9, free_b / 1e9,
        )


# ---------------------------------------------------------------------------
# Model registry — known model families with recommended settings
# ---------------------------------------------------------------------------

_MODEL_HINTS: Dict[str, Dict[str, Any]] = {
    "llama-3.3-70b": {
        "notes": "Llama 3.3 70B Instruct — needs 4-bit NF4 to fit in 2×T4 (28–30 GB).",
        "recommended_4bit": True,
    },
    "llama-3.1-70b": {
        "notes": "Llama 3.1 70B Instruct — needs 4-bit NF4 to fit in 2×T4.",
        "recommended_4bit": True,
    },
    "qwen3-80b": {
        "notes": "Qwen3-80B-A3B MoE — only ~3B active params/token, fits in ~10–12 GB at 4-bit.",
        "recommended_4bit": True,
    },
    "qwen2.5-72b": {
        "notes": "Qwen2.5 72B — needs 4-bit NF4 to fit in 2×T4.",
        "recommended_4bit": True,
    },
    "qwen2.5-7b": {
        "notes": "Qwen2.5 7B — fits comfortably in FP16 on a single T4.",
        "recommended_4bit": False,
    },
    "mixtral-8x7b": {
        "notes": "Mixtral 8x7B MoE — 4-bit recommended for safety.",
        "recommended_4bit": True,
    },
    "mistral-7b": {
        "notes": "Mistral 7B — fits in FP16 on a single T4.",
        "recommended_4bit": False,
    },
}


def _get_model_hints(model_name: str) -> Dict[str, Any]:
    """Return known tuning hints for a model name (case-insensitive substring match)."""
    lower = model_name.lower()
    for key, hints in _MODEL_HINTS.items():
        if key in lower:
            return hints
    return {}


# ---------------------------------------------------------------------------
# LocalLLMClient
# ---------------------------------------------------------------------------

class LocalLLMClient:
    """
    Loads a HuggingFace causal LM locally with optional 4-bit NF4 quantization
    and automatic multi-GPU layer splitting via HuggingFace Accelerate.

    The interface mirrors the remote :class:`~utils.llm_client.LLMClient` so
    it can be swapped in transparently — same ``generate()`` signature.

    Parameters
    ----------
    model_name:
        HuggingFace model ID, e.g. ``"meta-llama/Llama-3.3-70B-Instruct"``.
    hf_token:
        HuggingFace access token.  Required for gated models (Llama, etc.).
        Falls back to the ``HF_TOKEN`` environment variable if ``None``.
    load_in_4bit:
        Enable NF4 4-bit quantization via bitsandbytes.  **Must be True** for
        70B+ models on 2×T4 (30 GB total VRAM).
    bnb_4bit_compute_dtype:
        Dtype for 4-bit compute kernels.  ``"float16"`` is fastest on T4
        (Turing architecture).  Use ``"bfloat16"`` on Ampere+.
    bnb_4bit_use_double_quant:
        Enable nested/double quantization — saves ~0.4 extra bits per param.
    bnb_4bit_quant_type:
        Quantization type. ``"nf4"`` (NormalFloat4) gives the best
        quality/compression trade-off.
    device_map:
        Accelerate device map strategy.  ``"auto"`` is the right choice for
        Kaggle: it automatically fills GPU 0, then GPU 1, then overflows to CPU.
    max_memory_per_gpu:
        Hard cap on VRAM usage per GPU device, e.g. ``"14GiB"``.
        Leave ~1 GB headroom below the physical 15 GB.
    use_flash_attention:
        Enable Flash Attention 2 (requires ``pip install flash-attn``).
        **Keep False on Kaggle T4** — T4 is Turing (sm75), Flash Attention 2
        requires Ampere (sm80+).
    temperature:
        Default sampling temperature.
    max_new_tokens:
        Default maximum tokens to generate per call.
    do_sample:
        Use sampling (vs. greedy decoding).  Should be ``True`` when
        ``temperature > 0``.
    top_p:
        Nucleus sampling probability threshold.
    repetition_penalty:
        Token repetition penalty (> 1.0 reduces repeated phrases).
    """

    def __init__(
        self,
        model_name: str,
        *,
        hf_token: Optional[str] = None,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        device_map: str = "auto",
        max_memory_per_gpu: str = "14GiB",
        use_flash_attention: bool = False,
        temperature: float = 0.1,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> None:
        _require_deps()

        self.model_name        = model_name
        self.temperature       = temperature
        self.max_new_tokens    = max_new_tokens
        self.do_sample         = do_sample
        self.top_p             = top_p
        self.repetition_penalty = repetition_penalty

        # Log hints for known model families
        hints = _get_model_hints(model_name)
        if hints:
            logger.info("[LocalLLM] %s", hints.get("notes", ""))
            if hints.get("recommended_4bit") and not load_in_4bit:
                logger.warning(
                    "[LocalLLM] Model '%s' is recommended to run with load_in_4bit=True "
                    "on 2×T4, but load_in_4bit=False was specified. Risk of OOM.",
                    model_name,
                )

        # ------------------------------------------------------------------
        # Log GPU state before loading so we can see available VRAM
        # ------------------------------------------------------------------
        logger.info("[LocalLLM] GPU state before model load:")
        log_gpu_info()

        # ------------------------------------------------------------------
        # Build BitsAndBytes 4-bit quantization config
        # ------------------------------------------------------------------
        bnb_config = None
        if load_in_4bit:
            import torch as _torch
            _dtype_map = {
                "float16":  _torch.float16,
                "bfloat16": _torch.bfloat16,
                "float32":  _torch.float32,
            }
            compute_dtype = _dtype_map.get(bnb_4bit_compute_dtype, _torch.float16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
            )
            logger.info(
                "[LocalLLM] 4-bit NF4 quant enabled "
                "(compute_dtype=%s, double_quant=%s, quant_type=%s)",
                bnb_4bit_compute_dtype, bnb_4bit_use_double_quant, bnb_4bit_quant_type,
            )

        # ------------------------------------------------------------------
        # Build per-device max_memory dict for Accelerate
        # Accelerate's device_map="auto" uses this to greedily fill GPUs
        # from first to last, then overflow to CPU.
        # ------------------------------------------------------------------
        n_gpus = torch.cuda.device_count() if _TORCH_AVAILABLE else 0
        max_memory: Dict[Any, str] = {}
        if n_gpus > 0:
            for i in range(n_gpus):
                max_memory[i] = max_memory_per_gpu
            # Give Accelerate a generous CPU RAM buffer for overflow layers.
            # On Kaggle notebooks ~30 GB of CPU RAM is available.
            max_memory["cpu"] = "30GiB"
            logger.info(
                "[LocalLLM] device_map='auto' | %d GPU(s) @ %s each | CPU overflow=30GiB",
                n_gpus, max_memory_per_gpu,
            )
        else:
            logger.warning("[LocalLLM] No GPUs found — falling back to CPU inference.")
            device_map = "cpu"

        # ------------------------------------------------------------------
        # Resolve HF token
        # ------------------------------------------------------------------
        resolved_token = hf_token or os.environ.get("HF_TOKEN", "").strip() or None
        if resolved_token is None:
            logger.warning(
                "[LocalLLM] HF_TOKEN not set. Gated models (Llama, etc.) will fail. "
                "Set the HF_TOKEN environment variable or api_key_env in your config."
            )

        # ------------------------------------------------------------------
        # Build model kwargs
        # ------------------------------------------------------------------
        model_kwargs: Dict[str, Any] = {
            "device_map":        device_map,
            "trust_remote_code": True,   # required for Qwen and other custom models
            "token":             resolved_token,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            # No quantization — use FP16 to save memory vs FP32
            if _TORCH_AVAILABLE and n_gpus > 0:
                model_kwargs["torch_dtype"] = torch.float16

        if max_memory:
            model_kwargs["max_memory"] = max_memory

        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("[LocalLLM] Flash Attention 2 requested (ensure flash-attn is installed).")

        # ------------------------------------------------------------------
        # Load tokenizer
        # ------------------------------------------------------------------
        logger.info("[LocalLLM] Loading tokenizer for '%s'...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=resolved_token,
        )
        # Some models don't set a pad token — use EOS as fallback
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("[LocalLLM] pad_token was None — set to eos_token.")

        # ------------------------------------------------------------------
        # Load model weights
        # ------------------------------------------------------------------
        logger.info(
            "[LocalLLM] Loading model '%s' — this may take several minutes on first run "
            "(downloading weights from HuggingFace Hub)...",
            model_name,
        )
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        elapsed = time.perf_counter() - t0
        logger.info("[LocalLLM] Model loaded in %.1f s.", elapsed)

        # Log GPU state after loading so the user can see memory consumption
        logger.info("[LocalLLM] GPU state after model load:")
        log_gpu_info()

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------

    def _format_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format a prompt using the tokenizer's built-in chat template if available.

        Most modern HuggingFace models (Llama 3, Qwen2/3, Mistral) ship a
        ``chat_template`` in their tokenizer config, so this will do the right
        thing automatically.  Falls back to a generic format for older models.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prefer the tokenizer's own template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.warning(
                    "[LocalLLM] apply_chat_template failed (%s) — using generic fallback.", exc
                )

        # Generic fallback format
        parts = []
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}")
        parts.append(f"<|user|>\n{prompt}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature:   Optional[float] = None,
        max_tokens:    Optional[int]   = None,
    ) -> str:
        """
        Generate a response from the locally loaded model.

        Parameters
        ----------
        prompt:
            User instruction / question.
        system_prompt:
            Optional system message (prepended before the user turn).
        temperature:
            Sampling temperature override.  Pass ``0.0`` for greedy decoding.
        max_tokens:
            Maximum number of new tokens to generate.

        Returns
        -------
        str
            The generated assistant response (prompt stripped, special tokens removed).
        """
        _temp = temperature if temperature is not None else self.temperature
        _max  = max_tokens  if max_tokens  is not None else self.max_new_tokens

        formatted = self._format_prompt(prompt, system_prompt)

        # Tokenize — truncate context to avoid OOM on very long prompts
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        )

        # Move input tensors to the same device as the first model parameter
        # (Accelerate places the embedding layer on the first GPU in device_map=auto)
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=_max,
                do_sample=self.do_sample and _temp > 0.0,
                temperature=_temp if _temp > 0.0 else None,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the echoed prompt)
        new_token_ids = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return response.strip()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LocalLLMClient("
            f"model={self.model_name!r}, "
            f"device_map='auto', "
            f"4bit={hasattr(self, '_bnb_config')}"
            f")"
        )


__all__ = ["LocalLLMClient", "log_gpu_info"]
