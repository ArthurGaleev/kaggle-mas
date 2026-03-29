"""
Base agent class for the multi-agent ML system.
All agents inherit from BaseAgent and implement the execute() method.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from utils.helpers import safe_json_parse


class BaseAgent(ABC):
    """
    Abstract base class for all pipeline agents.

    Each agent receives the full pipeline state dict and returns an updated
    state dict. LLM calls are reserved for decisions that require reasoning;
    all data manipulation is done with deterministic Python code.
    """

    SYSTEM_PROMPT: str = (
        "You are an expert ML engineer working on a Kaggle rental-property "
        "regression competition. Your goal is to maximise predictive accuracy "
        "(minimise MSE) while keeping solutions reproducible and resource-efficient."
    )

    def __init__(
        self,
        cfg: DictConfig,
        llm_client: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Args:
            cfg:        OmegaConf DictConfig with the full pipeline config.
            llm_client: LLMClient instance with a .generate(prompt, system_prompt) method.
            logger:     Optional pre-configured logger; a default one is created if None.
        """
        self.cfg = cfg
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(self.name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the agent's class name."""
        return self.__class__.__name__

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent's main logic.

        Args:
            state: Shared pipeline state dict.

        Returns:
            Updated state dict.
        """

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _ask_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a prompt to the LLM and return the raw text response.

        Args:
            prompt:        User-facing prompt.
            system_prompt: Optional system-level instruction. Defaults to
                           the agent's SYSTEM_PROMPT class attribute.

        Returns:
            LLM response string.
        """
        system_prompt_str = system_prompt if system_prompt is not None else self.SYSTEM_PROMPT
        self._log(f"Calling LLM — prompt length: {len(prompt)} chars")
        response: str = self.llm_client.generate(prompt, system_prompt=system_prompt_str)
        self._log(f"LLM responded — response length: {len(response)} chars")
        return response

    def _ask_llm_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        default: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt to the LLM and parse the JSON response.

        Retries once on parse failure before falling back to *default*.

        Args:
            prompt:        User-facing prompt.
            system_prompt: Optional system-level instruction.
            default:       Fallback dict returned when JSON cannot be parsed.

        Returns:
            Parsed dict from LLM, or *default* on failure.
        """
        system_prompt_str = system_prompt if system_prompt is not None else self.SYSTEM_PROMPT

        for attempt in range(1, 3):  # try twice
            raw = self.llm_client.generate(prompt, system_prompt=system_prompt_str)
            parsed = safe_json_parse(raw)
            if parsed is not None:
                return parsed
            self._log(
                f"JSON parse failed on attempt {attempt}/2 — "
                f"raw snippet: {raw[:200]}",
                level="warning",
            )

        self._log("Using default JSON response after 2 failed parse attempts.", level="warning")
        return default if default is not None else {}

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message at the specified level, prefixed with the agent name.

        Args:
            message: Log message.
            level:   One of 'debug', 'info', 'warning', 'error', 'critical'.
        """
        full_message = f"[{self.name}] {message}"
        log_fn = getattr(self.logger, level, self.logger.info)
        log_fn(full_message)

    # ------------------------------------------------------------------
    # Execution timing decorator / context manager
    # ------------------------------------------------------------------

    def _timed_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper that times execute() and stores elapsed seconds in state.

        Usage: call this instead of execute() when you want automatic timing.
        Subclasses should NOT override this; override execute() instead.
        """
        start = time.time()
        self._log("Starting execution.")
        try:
            result = self.execute(state)
        except Exception as exc:
            elapsed = time.time() - start
            self._log(
                f"Execution FAILED after {elapsed:.2f}s — {exc}",
                level="error",
            )
            raise
        elapsed = time.time() - start
        self._log(f"Execution completed in {elapsed:.2f}s.")

        timings: Dict[str, float] = result.get("agent_timings", {})
        timings[self.name] = round(elapsed, 3)
        result["agent_timings"] = timings
        return result
