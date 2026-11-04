"""Shared pagination helpers for gateway routers."""

from __future__ import annotations


def trim_run_message_page(rows: list[dict], *, limit: int, after_seq: int | None) -> tuple[list[dict], bool]:
    """Trim a ``limit + 1`` run-message page while preserving page boundaries."""
    has_more = len(rows) > limit
    if not has_more:
        return rows, False

    if after_seq is not None:
        return rows[:limit], True

    # `rows[-limit:]` only matches `rows[:limit]` at the boundary for
    # limit >= 1: with limit == 0 the slice index -0 is 0, so it returns
    # every row while still reporting has_more. Spell the tail as a
    # positive offset so both branches agree on degenerate limits.
    return rows[len(rows) - limit :], True
