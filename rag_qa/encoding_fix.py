"""Make Streamlit Cloud's ASCII locale safe for HuggingFace/Rich progress output.

Cloud logs often fail on the literal 'Downloading…' (ellipsis at index 11).
"""

from __future__ import annotations

import os
import sys


class _SafeWriter:
    """Proxy that never raises UnicodeEncodeError on write."""

    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def write(self, s: object) -> int:
        if not isinstance(s, str):
            return self._wrapped.write(s)  # type: ignore[no-any-return]
        try:
            return self._wrapped.write(s)  # type: ignore[no-any-return]
        except UnicodeEncodeError:
            safe = s.encode("ascii", "replace").decode("ascii")
            return self._wrapped.write(safe)  # type: ignore[no-any-return]

    def writelines(self, lines: object) -> None:
        for line in lines:  # type: ignore[attr-defined]
            self.write(line)

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)


def _force_env() -> None:
    # Force (not setdefault): Streamlit images sometimes pre-set empty/odd values.
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


def _disable_hub_progress() -> None:
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def apply() -> None:
    _force_env()
    _wrap_stream("stdout")
    _wrap_stream("stderr")
    _disable_hub_progress()


apply()
