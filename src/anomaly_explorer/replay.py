"""Small, deterministic helpers for historical-series replay in the UI."""

import pandas as pd


def initial_cursor(total: int, minimum_context: int, tail: int = 240) -> int | None:
    """Keep enough fitting history and leave at least one point to replay."""
    if total <= minimum_context:
        return None
    return min(total - 1, max(minimum_context, total - tail))


def next_cursor(cursor: int, total: int, start: int, step: int, playing: bool) -> int:
    """Advance only when playing; loop to the first replay frame at the end."""
    if not playing:
        return cursor
    return start if cursor >= total else min(total, cursor + step)


def chronological_prefix(frame: pd.DataFrame, cursor: int) -> pd.DataFrame:
    """Expose only points that have arrived at the current replay position."""
    return frame.sort_values("timestamp", kind="stable").iloc[:cursor].copy()
