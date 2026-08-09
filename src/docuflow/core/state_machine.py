"""Generic state machine for document processing flows.

A document pipeline is a graph of states: navigate to source, download the
document, extract fields, verify, record. Each state runs a handler and
returns the name of the next state (or a terminal marker). The machine
supports resuming from any state via checkpoints, which makes long batch runs
crash-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

START = "__start__"
END = "__end__"


@dataclass
class State:
    """A single pipeline state.

    Attributes:
        name: Unique state identifier.
        handler: Callable ``(context) -> str`` returning the next state name,
            ``END`` to finish, or ``None`` to stay in this state.
        description: Human-readable purpose of the state.
        retries: Number of times the handler may be retried on failure.
    """

    name: str
    handler: Callable[[Dict[str, Any]], Optional[str]]
    description: str = ""
    retries: int = 1


class StateMachine:
    """Run a graph of states with checkpoint support.

    Args:
        states: Ordered mapping of state name -> :class:`State`. The first
            state in the mapping is the entry point.
        checkpoint: Optional :class:`~docuflow.core.checkpoint.CheckpointStore`
            used to persist progress between runs.
        context: Initial execution context shared by all handlers.
    """

    def __init__(
        self,
        states: Dict[str, State],
        checkpoint: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.states = states
        self.checkpoint = checkpoint
        self.context = context or {}
        self.current: Optional[str] = None
        self.history: list = []

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def run(self, start_state: Optional[str] = None) -> Dict[str, Any]:
        """Run the machine until it reaches the terminal state.

        Args:
            start_state: State to begin from (defaults to the first state in
                the mapping, or the checkpointed position when available).

        Returns:
            The execution context after the run.
        """
        self.current = start_state or self._resolve_start()
        if self.current not in self.states:
            raise ValueError(f"Unknown start state: {self.current}")

        logger.info("State machine starting at '%s'", self.current)
        while self.current is not None and self.current != END:
            state = self.states[self.current]
            logger.info("Entering state '%s' (%s)", state.name, state.description)

            next_state: Optional[str] = None
            attempts = 0
            while attempts < max(1, state.retries):
                attempts += 1
                try:
                    next_state = state.handler(self.context)
                    break
                except Exception as exc:  # pragma: no cover - failure path
                    logger.error(
                        "State '%s' failed (attempt %d/%d): %s",
                        state.name, attempts, state.retries, exc,
                    )
                    if attempts >= max(1, state.retries):
                        raise

            self.history.append(state.name)
            self._checkpoint(state.name, next_state)
            self.current = next_state if next_state != END else None

        logger.info("State machine finished after %d states", len(self.history))
        return self.context

    def _resolve_start(self) -> str:
        """Pick the entry state, honoring a saved checkpoint position."""
        if self.checkpoint is not None:
            saved = self.checkpoint.load()
            saved_state = saved.get("current_item")
            if saved_state in self.states:
                logger.info("Resuming from checkpoint state '%s'", saved_state)
                return str(saved_state)
        return next(iter(self.states))

    def _checkpoint(self, state_name: str, next_state: Optional[str]) -> None:
        """Persist progress after each state transition."""
        if self.checkpoint is not None:
            self.checkpoint.save(
                current_item=next_state or END,
                processed_items=set(self.history),
                extra={"last_state": state_name, "history": self.history[-20:]},
            )
