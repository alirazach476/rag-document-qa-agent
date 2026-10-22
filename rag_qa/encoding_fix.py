"""Hardening for Streamlit Community Cloud ASCII locales.

Cloud workers often use ASCII stdout. HuggingFace/Rich then crash on
'Downloading…' (ellipsis at index 11). We wrap every write path and, for
Groq/cloud, avoid those libraries entirely via HashEmbeddings.
"""

from __future__ import annotations

import builtins
import logging
import os
import sys


class _SafeWriter:
    """Proxy that never raises UnicodeEncodeError on write."""

    def __init__(self, wrapped: object) -> None:
        object.__setattr__(self, "_wrapped", wrapped)

    def write(self, s: object) -> int:
        wrapped = object.__getattribute__(self, "_wrapped")
        if not isinstance(s, str):
            return wrapped.write(s)  # type: ignore[no-any-return]
        try:
            return wrapped.write(s)  # type: ignore[no-any-return]
        except UnicodeEncodeError:
            safe = s.encode("ascii", "replace").decode("ascii")
            return wrapped.write(safe)  # type: ignore[no-any-return]

    def writelines(self, lines: object) -> None:
        for line in lines:  # type: ignore[attr-defined]
            self.write(line)

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "_wrapped"), name)


def _force_env() -> None:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["LANG"] = "C.UTF-8"
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LC_CTYPE"] = "C.UTF-8"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _wrap_stream(name: str) -> None:
    stream = getattr(sys, name, None)
    if stream is None or isinstance(stream, _SafeWriter):
        return
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    setattr(sys, name, _SafeWriter(stream))


def _wrap_logging_streams() -> None:
    for handler in logging.root.handlers:
        stream = getattr(handler, "stream", None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            try:
                handler.stream = _SafeWriter(stream)
            except Exception:
                pass


def _patch_print() -> None:
    if getattr(builtins.print, "_documind_safe", False):
        return
    real_print = builtins.print

    def safe_print(*args: object, **kwargs: object) -> None:
        try:
            real_print(*args, **kwargs)
        except UnicodeEncodeError:
            safe_args = tuple(
                a.encode("ascii", "replace").decode("ascii") if isinstance(a, str) else a
                for a in args
            )
            real_print(*safe_args, **kwargs)

    safe_print._documind_safe = True  # type: ignore[attr-defined]
    builtins.print = safe_print  # type: ignore[assignment]


def _patch_locale() -> None:
    try:
        import locale

        locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"  # type: ignore[assignment]
    except Exception:
        pass


def apply() -> None:
    _force_env()
    _wrap_stream("stdout")
    _wrap_stream("stderr")
    _wrap_logging_streams()
    _patch_print()
    _patch_locale()


apply()
