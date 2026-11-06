"""Log redaction helpers.

Foundation-layer module with no project dependencies, importable from any layer.
The string/repr of a googleapiclient ``HttpError`` contains the full request URL
including ``key=<developerKey>``; every place that writes exception text to a
log goes through this module.

Rule: any log call that includes raw exception text uses ``%s`` with
``safe_repr`` (never a bare ``%r``). Redaction only masks secrets and never
swallows information — everything except the secret values is preserved byte
for byte, so troubleshooting detail is not reduced.
"""
import re
import sys
import traceback

# Credentials in URL query form: key=<value>, value runs until &, whitespace or
# a quote. Covers both HttpError.uri and the str(exc) reprinted on the last
# traceback line. Case-insensitive so Key=/KEY= variants are masked too.
# Deliberately broad: do not add \b or narrow the value charset — HttpError's
# URL formats vary, and over-matching (e.g. donkey=x) is an accepted trade-off
# while under-matching would leak a key.
_KEY_RE = re.compile(r"(key=)[^&\s'\"]+", re.IGNORECASE)
# Additional secrets:
#   token=<session token>  (session tokens carried in query strings, e.g. SSE)
#   Bearer <token>         (when downstream request headers end up in an exception)
#   sk-ant-…               (an Anthropic API key repr'd into an exception string)
_MORE_RES = (
    re.compile(r"(token=)[^&\s'\"]+", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),
)


def redact(text: str) -> str:
    """Mask key=/token=/Bearer values as *** and any sk-ant-… key as sk-ant-***."""
    out = _KEY_RE.sub(r"\1***", text)
    out = _MORE_RES[0].sub(r"\1***", out)
    out = _MORE_RES[1].sub(r"\1***", out)
    return _MORE_RES[2].sub("sk-ant-***", out)


def safe_repr(exc: object) -> str:
    """Redacted repr — use with %s in log formatting instead of a bare %r."""
    return redact(repr(exc))


def redacted_traceback() -> str:
    """The current exception's full traceback, redacted.

    Use ``logger.error("... degraded\\n%s", ..., redacted_traceback())`` instead
    of ``logger.exception(...)``: the stack stays visible (degradations must be
    observable) while credentials never reach the log."""
    return redact(traceback.format_exc())


def install_excepthook() -> None:
    """Redact uncaught exceptions in offline scripts.

    When a script's output is redirected to a log file, an uncaught exception's
    traceback goes straight through ``sys.excepthook`` and would write an
    HttpError URL with its key verbatim. This replaces the hook with
    format_exception → redact → stderr, keeping the stack intact. Idempotent.
    Not needed in the web server process, whose error middleware already uses
    ``redacted_traceback``."""
    def _hook(exc_type, exc, tb):
        sys.stderr.write(redact("".join(traceback.format_exception(exc_type, exc, tb))))
    if getattr(sys.excepthook, "__name__", "") != "_hook":
        sys.excepthook = _hook
