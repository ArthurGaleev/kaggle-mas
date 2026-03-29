"""
Unified LLM client for the Kaggle MAS pipeline.

Supports Groq, HuggingFace Inference, OpenRouter (remote, via OpenAI-compatible
API), and local HuggingFace model loading (provider="local").

A single :class:`LLMClient` instance wraps all provider differences,
implements retry logic with exponential back-off, optional structured JSON output,
token usage tracking, and automatic fallback to a secondary model.

Usage::

    from omegaconf import OmegaConf
    from utils.llm_client import LLMClient

    cfg = OmegaConf.load("configs/config.yaml")
    client = LLMClient(cfg)

    response = client.generate(
        prompt="Describe the dataset.",
        system_prompt="You are a data scientist.",
    )

    structured = client.generate(
        prompt="Return a JSON with keys 'summary' and 'n_rows'.",
        response_format={"type": "json_object"},
    )

Providers
---------
- ``groq``         — fastest remote inference (default)
- ``huggingface``  — HuggingFace Inference API (serverless)
- ``openrouter``   — unified gateway to many remote models
- ``local``        — load model locally with 4-bit NF4 + 2×GPU splitting
                     (for Kaggle 2×T4; see configs/llm/local_hf.yaml)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union

try:
    from omegaconf import DictConfig, OmegaConf
    _OMEGACONF_AVAILABLE = True
except ImportError:
    DictConfig = dict  # type: ignore[misc,assignment]
    _OMEGACONF_AVAILABLE = False

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

from utils.logger import TokenTracker, get_logger, global_token_tracker

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
_log = get_logger("kaggle-mas.llm_client")

# ---------------------------------------------------------------------------
# Retry configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_MAX_RETRIES      = 3
_DEFAULT_BASE_DELAY_S     = 1.0
_DEFAULT_BACKOFF_FACTOR   = 2.0
_DEFAULT_MAX_DELAY_S      = 30.0

# Exceptions that warrant a retry
_RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key(env_var: str) -> str:
    """
    Read an API key from the environment.

    Args:
        env_var: Name of the environment variable to read.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the variable is not set or is empty.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise EnvironmentError(
            f"LLM API key not found.  Please set the environment variable "
            f"'{env_var}' (e.g. in Colab: os.environ['{env_var}'] = '...')."
        )
    return key


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve a value from either OmegaConf DictConfig or a plain dict."""
    try:
        if _OMEGACONF_AVAILABLE and isinstance(cfg, DictConfig):
            return OmegaConf.select(cfg, key, default=default)
        return cfg.get(key, default)  # type: ignore[union-attr]
    except Exception:
        return default


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thread-safe, config-driven LLM client supporting multiple providers.

    Remote providers (all via OpenAI-compatible API):
    - **Groq**        — fastest inference, great for development
    - **HuggingFace** — serverless inference API
    - **OpenRouter**  — unified gateway to many models

    Local provider:
    - **local** — loads the model directly on Kaggle's 2×T4 GPUs using
      ``transformers`` + ``bitsandbytes`` 4-bit NF4 quantization and
      HuggingFace Accelerate's ``device_map="auto"`` for automatic multi-GPU
      layer splitting.  See :class:`~utils.local_llm_client.LocalLLMClient`.

    Args:
        cfg:            Hydra/OmegaConf config object.  Must contain an ``llm``
                        sub-config with at least ``provider``, ``model``,
                        ``base_url``, and ``api_key_env`` keys.
        token_tracker:  :class:`~utils.logger.TokenTracker` instance.
                        Defaults to the global tracker.
        max_retries:    Maximum number of attempts before giving up (remote only).
        base_delay:     Initial wait between retries in seconds (remote only).
        backoff_factor: Multiplier applied to the delay after each retry.
        max_delay:      Upper bound on the retry delay in seconds.

    Example::

        # Remote (Groq, default)
        cfg = OmegaConf.load("configs/config.yaml")
        client = LLMClient(cfg)
        answer = client.generate("What is the MSE metric?")

        # Local large model on 2xT4
        # python main.py llm=local_hf
        # python main.py llm=local_hf llm.model=meta-llama/Llama-3.3-70B-Instruct
        # python main.py llm=local_hf llm.model=Qwen/Qwen3-80B-A3B-Instruct
    """

    def __init__(
        self,
        cfg: Any,
        *,
        token_tracker: Optional[TokenTracker] = None,
        max_retries:    int   = _DEFAULT_MAX_RETRIES,
        base_delay:     float = _DEFAULT_BASE_DELAY_S,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
        max_delay:      float = _DEFAULT_MAX_DELAY_S,
    ) -> None:
        self._lock = threading.Lock()

        # ------------------------------------------------------------------
        # Parse config
        # ------------------------------------------------------------------
        llm_cfg = _cfg_get(cfg, "llm", cfg)   # support both full-config and llm-only

        self.provider:        str = _cfg_get(llm_cfg, "provider", "groq")
        self.model:           str = _cfg_get(llm_cfg, "model",    "llama-3.3-70b-versatile")
        self.fallback_model:  str = _cfg_get(llm_cfg, "fallback_model", self.model)
        self.base_url:        str = _cfg_get(llm_cfg, "base_url", "https://api.groq.com/openai/v1")
        api_key_env:          str = _cfg_get(llm_cfg, "api_key_env", "GROQ_API_KEY")
        self.temperature:   float = float(_cfg_get(llm_cfg, "temperature", 0.1))
        self.max_tokens:      int = int(_cfg_get(llm_cfg, "max_tokens", 4096))

        # ------------------------------------------------------------------
        # Token tracking
        # ------------------------------------------------------------------
        self._token_tracker  = token_tracker or global_token_tracker

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._current_agent: str = "unknown"
        self._current_phase: str = "unknown"

        # Retry settings (used only for remote providers)
        self._max_retries    = max_retries
        self._base_delay     = base_delay
        self._backoff_factor = backoff_factor
        self._max_delay      = max_delay

        # ------------------------------------------------------------------
        # LOCAL provider — delegate everything to LocalLLMClient
        # Loads model weights onto Kaggle's 2×T4 GPUs via Accelerate +
        # bitsandbytes 4-bit NF4 quantization.
        # ------------------------------------------------------------------
        self._local_client: Optional[Any] = None
        if self.provider == "local":
            from utils.local_llm_client import LocalLLMClient

            local_cfg = _cfg_get(llm_cfg, "local", {})

            # HF token: read from the configured env var (same as remote providers)
            hf_token = os.environ.get(api_key_env, "").strip() or None

            self._local_client = LocalLLMClient(
                model_name             = self.model,
                hf_token               = hf_token,
                load_in_4bit           = bool(_cfg_get(local_cfg, "load_in_4bit",              True)),
                bnb_4bit_compute_dtype = str( _cfg_get(local_cfg, "bnb_4bit_compute_dtype",   "float16")),
                bnb_4bit_use_double_quant = bool(_cfg_get(local_cfg, "bnb_4bit_use_double_quant", True)),
                bnb_4bit_quant_type    = str( _cfg_get(local_cfg, "bnb_4bit_quant_type",      "nf4")),
                device_map             = str( _cfg_get(local_cfg, "device_map",               "auto")),
                max_memory_per_gpu     = str( _cfg_get(local_cfg, "max_memory_per_gpu",       "14GiB")),
                use_flash_attention    = bool(_cfg_get(local_cfg, "use_flash_attention",       False)),
                temperature            = self.temperature,
                max_new_tokens         = self.max_tokens,
                do_sample              = bool(_cfg_get(local_cfg, "do_sample",                 True)),
                top_p                  = float(_cfg_get(local_cfg, "top_p",                    0.9)),
                repetition_penalty     = float(_cfg_get(local_cfg, "repetition_penalty",       1.1)),
            )
            _log.info(
                "LLMClient initialised | provider=local model=%s (LocalLLMClient)",
                self.model,
            )
            # Skip OpenAI client setup entirely for local provider
            self._client = None  # type: ignore[assignment]
            self.api_key = ""    # not used
            return

        # ------------------------------------------------------------------
        # REMOTE providers — OpenAI-compatible API
        # ------------------------------------------------------------------
        self.api_key: str = _get_api_key(api_key_env)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        _log.info(
            "LLMClient initialised | provider=%s model=%s base_url=%s",
            self.provider, self.model, self.base_url,
        )

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def set_context(self, agent: str, phase: str) -> None:
        """
        Set the current agent/phase context for token tracking.

        Args:
            agent: Agent name, e.g. ``"DataAgent"``.
            phase: Pipeline phase, e.g. ``"data_profiling"``.
        """
        with self._lock:
            self._current_agent = agent
            self._current_phase = phase

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature:   Optional[float] = None,
        max_tokens:    Optional[int]   = None,
        response_format: Optional[Dict[str, str]] = None,
        agent:  Optional[str] = None,
        phase:  Optional[str] = None,
    ) -> str:
        """
        Generate a text (or structured JSON) response from the configured LLM.

        For the ``local`` provider, delegates directly to
        :class:`~utils.local_llm_client.LocalLLMClient`.  Note that local
        models do not support ``response_format`` — if structured JSON output
        is needed, rely on prompt engineering instead.

        Args:
            prompt:          User message / instruction.
            system_prompt:   Optional system message prepended to the conversation.
            temperature:     Override the default temperature from config.
            max_tokens:      Override the default max_tokens from config.
            response_format: Optional format specifier (remote providers only).
                             Pass ``{"type": "json_object"}`` to request structured
                             JSON output.  Silently ignored for ``local`` provider.
            agent:           Override the agent name for token tracking.
            phase:           Override the phase name for token tracking.

        Returns:
            The model's text response as a string.

        Raises:
            RuntimeError: If all retry attempts (including fallback) fail.
        """
        _agent = agent or self._current_agent
        _phase = phase or self._current_phase
        _temp  = temperature if temperature is not None else self.temperature
        _toks  = max_tokens  if max_tokens  is not None else self.max_tokens

        # ------------------------------------------------------------------
        # Local provider path
        # ------------------------------------------------------------------
        if self._local_client is not None:
            if response_format is not None:
                _log.debug(
                    "[LocalLLM] response_format is not supported for local provider — "
                    "relying on prompt engineering for structured output."
                )
            return self._local_client.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=_temp,
                max_tokens=_toks,
            )

        # ------------------------------------------------------------------
        # Remote provider path (Groq / HuggingFace API / OpenRouter)
        # ------------------------------------------------------------------
        messages = self._build_messages(prompt, system_prompt)

        # First attempt with primary model, then fallback
        for attempt_model, is_fallback in [
            (self.model, False),
            (self.fallback_model, True),
        ]:
            try:
                response_text = self._call_with_retry(
                    model=attempt_model,
                    messages=messages,
                    temperature=_temp,
                    max_tokens=_toks,
                    response_format=response_format,
                    agent=_agent,
                    phase=_phase,
                )
                return response_text
            except Exception as exc:
                if is_fallback:
                    raise RuntimeError(
                        f"LLMClient: all attempts failed (primary={self.model}, "
                        f"fallback={self.fallback_model}).  Last error: {exc}"
                    ) from exc
                _log.warning(
                    "Primary model '%s' failed (%s).  Falling back to '%s'.",
                    attempt_model, exc, self.fallback_model,
                )

        # Should never reach here
        raise RuntimeError("LLMClient: unexpected state after model attempts.")

    # ------------------------------------------------------------------
    # Internal helpers (remote only)
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        """Assemble the messages list for the chat completions API."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_with_retry(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]],
        agent: str,
        phase: str,
    ) -> str:
        """
        Call the OpenAI-compatible API with exponential back-off retry.

        Args:
            model:           Model identifier to use.
            messages:        Chat messages list.
            temperature:     Sampling temperature.
            max_tokens:      Maximum tokens to generate.
            response_format: Optional JSON format specifier.
            agent:           Agent name for token tracking.
            phase:           Phase name for token tracking.

        Returns:
            The text content of the first choice.

        Raises:
            Exception: Re-raises the last exception after *max_retries* attempts.
        """
        delay    = self._base_delay
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model":       model,
                    "messages":    messages,
                    "temperature": temperature,
                    "max_tokens":  max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                t0 = time.perf_counter()
                with self._lock:
                    completion = self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
                elapsed = time.perf_counter() - t0

                # Extract text
                choice = completion.choices[0]
                text   = choice.message.content or ""

                # Track tokens
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    p_toks = getattr(usage, "prompt_tokens",     0) or 0
                    c_toks = getattr(usage, "completion_tokens", 0) or 0
                    cost   = self._token_tracker.record(
                        agent=agent,
                        phase=phase,
                        model=model,
                        prompt_tokens=p_toks,
                        completion_tokens=c_toks,
                    )
                    _log.debug(
                        "LLM call OK | model=%s attempt=%d elapsed=%.2fs "
                        "tokens=%d+%d cost=$%.6f",
                        model, attempt, elapsed, p_toks, c_toks, cost,
                    )
                else:
                    _log.debug(
                        "LLM call OK | model=%s attempt=%d elapsed=%.2fs "
                        "(token usage not reported by provider)",
                        model, attempt, elapsed,
                    )

                return text

            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                _log.warning(
                    "LLM retryable error on attempt %d/%d (model=%s): %s. "
                    "Retrying in %.1fs…",
                    attempt, self._max_retries, model, exc, delay,
                )
                time.sleep(delay)
                delay = min(delay * self._backoff_factor, self._max_delay)

            except APIError as exc:
                _log.error(
                    "LLM non-retryable APIError on attempt %d (model=%s): %s",
                    attempt, model, exc,
                )
                raise

        # Exhausted retries
        raise RuntimeError(
            f"LLM call failed after {self._max_retries} attempts "
            f"(model={model}). Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature:   Optional[float] = None,
        max_tokens:    Optional[int]   = None,
        agent:  Optional[str] = None,
        phase:  Optional[str] = None,
    ) -> Optional[Any]:
        """
        Generate a response and automatically parse it as JSON.

        Requests structured JSON output from the provider (``json_object`` mode
        for remote providers) and passes the raw text through
        :func:`~utils.helpers.safe_json_parse`.

        For the ``local`` provider, ``json_object`` mode is not supported by
        the model backend — the prompt itself should instruct the model to
        respond with JSON, and this method will still attempt to parse the output.

        Args:
            prompt:       User instruction.
            system_prompt: Optional system message.
            temperature:  Sampling temperature override.
            max_tokens:   Max tokens override.
            agent:        Agent name for tracking.
            phase:        Phase name for tracking.

        Returns:
            Parsed Python object (dict, list, etc.), or ``None`` on parse failure.
        """
        from utils.helpers import safe_json_parse  # local import avoids circularity

        # For local provider we skip response_format (not supported)
        fmt = None if self._local_client is not None else {"type": "json_object"}

        raw = self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=fmt,
            agent=agent,
            phase=phase,
        )
        parsed = safe_json_parse(raw)
        if parsed is None:
            _log.warning(
                "generate_json: could not parse response as JSON.  "
                "Raw response (first 200 chars): %s", raw[:200]
            )
        return parsed

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._local_client is not None:
            return f"LLMClient(provider=local, model={self.model!r}, client={self._local_client!r})"
        return (
            f"LLMClient(provider={self.provider!r}, model={self.model!r}, "
            f"fallback={self.fallback_model!r}, base_url={self.base_url!r})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(
    cfg: Any,
    *,
    token_tracker: Optional[TokenTracker] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> LLMClient:
    """
    Convenience factory that creates an :class:`LLMClient` from a Hydra config.

    Args:
        cfg:           Full Hydra/OmegaConf config object.
        token_tracker: Optional token tracker (defaults to global).
        max_retries:   Maximum retry attempts.

    Returns:
        A configured :class:`LLMClient` instance.

    Example::

        import hydra
        from omegaconf import DictConfig
        from utils.llm_client import create_llm_client

        @hydra.main(config_path="configs", config_name="config")
        def main(cfg: DictConfig) -> None:
            client = create_llm_client(cfg)
            print(client)
    """
    return LLMClient(cfg, token_tracker=token_tracker, max_retries=max_retries)


__all__ = [
    "LLMClient",
    "create_llm_client",
]
