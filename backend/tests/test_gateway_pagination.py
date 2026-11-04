"""Unit tests for the shared run-message page trimming helper."""

from __future__ import annotations

from app.gateway.pagination import trim_run_message_page


def _rows(count: int) -> list[dict]:
    return [{"seq": index} for index in range(1, count + 1)]


def test_page_without_extra_row_is_returned_whole() -> None:
    data, has_more = trim_run_message_page(_rows(2), limit=2, after_seq=None)

    assert data == _rows(2)
    assert has_more is False


def test_forward_page_keeps_the_first_limit_rows() -> None:
    data, has_more = trim_run_message_page(_rows(4), limit=2, after_seq=1)

    assert data == _rows(2)
    assert has_more is True


def test_backward_page_keeps_the_last_limit_rows() -> None:
    data, has_more = trim_run_message_page(_rows(4), limit=2, after_seq=None)

    assert data == [{"seq": 3}, {"seq": 4}]
    assert has_more is True


def test_zero_limit_trims_to_an_empty_page_in_both_directions() -> None:
    """`rows[-0:]` is `rows[0:]`, so the tail branch used to leak the whole page."""
    forward = trim_run_message_page(_rows(3), limit=0, after_seq=5)
    backward = trim_run_message_page(_rows(3), limit=0, after_seq=None)

    assert forward == ([], True)
    assert backward == ([], True)


def test_negative_limit_does_not_return_rows() -> None:
    assert trim_run_message_page(_rows(3), limit=-1, after_seq=None) == ([], True)
