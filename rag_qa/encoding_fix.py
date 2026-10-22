"""Force UTF-8 I/O before any library prints progress (Streamlit Cloud)."""

from __future__ import annotations

import os
import sys


def apply() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LANG", "C.UTF-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    os.environ.setdefault("LC_CTYPE", "C.UTF-8")
    # Silence hub/tqdm progress lines that often contain Unicode ellipsis.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("FASTEMBED_SKIP_DOWNLOAD_PROGRESS", "1")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass
        try:
            # Keep ASCII locale if forced, but never raise on Unicode prints.
            stream.reconfigure(errors="replace")
            continue
        except Exception:
            pass
        try:
            import io

            buf = getattr(stream, "buffer", None)
            if buf is not None:
                setattr(
                    sys,
                    stream_name,
                    io.TextIOWrapper(
                        buf,
                        encoding="utf-8",
                        errors="replace",
                        line_buffering=True,
                    ),
                )
        except Exception:
            pass


apply()
