"""Per-refresh LLM token meter (observability only — it enforces no limits).

Lets the agent layer stay free of SQL: judges and the planner only call
``add()``, and the orchestrator persists the totals. The orchestrator calls
``start()`` at the beginning of a run; producer coroutines spawned with
``create_task`` copy the current context and therefore share the same meter.
"""

# A ContextVar (PEP 567) holds one value per asyncio execution context, so
# concurrent refreshes for different users on the same event loop never mix
# their counts — a plain module global would.
from contextvars import ContextVar


class _Meter:
    __slots__ = ("input_tokens", "output_tokens")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, usage) -> None:
        # Tolerate usage objects that lack the attribute or report None.
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0

# Must be module-level (a ContextVar created inside a function would be a new
# variable on every call). default=None makes metering optional: outside a
# started context, add() is a no-op.
_current: ContextVar[_Meter | None] = ContextVar("llm_meter", default=None)


def start() -> _Meter:
    """Install a fresh meter in the current context and return it."""
    meter = _Meter()
    _current.set(meter)
    return meter


def add(usage) -> None:
    """Add a response's token usage to the current meter, if one is active."""
    meter = _current.get()
    if meter is not None:
        meter.add(usage)
