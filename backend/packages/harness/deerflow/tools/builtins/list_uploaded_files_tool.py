"""Tool for discovering historical uploaded files in the current thread.

Unlike ``<current_uploads>`` which lists only this run's newly uploaded files,
this tool lets the agent discover files uploaded in previous turns on demand.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

from langchain.tools import tool
from langgraph.config import get_config

from deerflow.agents.middlewares.input_sanitization_middleware import neutralize_untrusted_tags
from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.tools.types import Runtime
from deerflow.uploads.manager import is_upload_staging_file
from deerflow.utils.file_outline import extract_outline_for_file

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 20
_MAX_MAX_RESULTS = 100
_CURSOR_VERSION = 1
_MAX_CURSOR_CHARS = 2048
_INVALID_CURSOR_MESSAGE = "Invalid or stale cursor. Restart without a cursor to request the first page."


def _extension_label(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    return neutralize_untrusted_tags(suffix) or "(no extension)"


def _format_omitted_summary(omitted: list[str]) -> str:
    counts = Counter(_extension_label(Path(f)) for f in omitted)
    parts = [f"{count} {ext}" for ext, count in sorted(counts.items())]
    return neutralize_untrusted_tags(", ".join(parts))


def _resolve_thread_id(runtime: Runtime) -> str | None:
    """Resolve the current thread id from runtime context or RunnableConfig."""
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id:
        return thread_id

    runtime_config = getattr(runtime, "config", None) or {}
    thread_id = runtime_config.get("configurable", {}).get("thread_id")
    if thread_id:
        return thread_id

    try:
        return get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        return None


def _resolve_user_id(runtime: Runtime) -> str:
    """Resolve the current user id."""
    from deerflow.runtime.user_context import resolve_runtime_user_id

    return resolve_runtime_user_id(runtime) or get_effective_user_id()


def _normalize_query(query: str | None) -> str | None:
    """Return a stripped query, or None when filtering should be skipped."""
    if not isinstance(query, str):
        return None
    stripped = query.strip()
    return stripped or None


def _normalize_extensions(extensions: list[str] | None) -> frozenset[str] | None:
    """Normalize extension tokens to lowercase dotted suffixes.

    Non-strings, blanks, and a non-list input are dropped. A leading ``*`` is
    stripped so model-supplied glob tokens like ``*.pdf`` still match
    ``Path.suffix``. An empty result means "no extension filter", matching
    the unfiltered historical behavior.
    """
    if not isinstance(extensions, list):
        return None
    normalized: set[str] = set()
    for item in extensions:
        if not isinstance(item, str):
            continue
        token = item.strip().lower().lstrip("*")
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        normalized.add(token)
    return frozenset(normalized) or None


def _matches_filters(
    filename: str,
    suffix: str,
    query: str | None,
    extensions: frozenset[str] | None,
) -> bool:
    if extensions is not None and suffix.lower() not in extensions:
        return False
    if query is not None and query.casefold() not in filename.casefold():
        return False
    return True


def _sort_key(item: tuple[float, Path, int]) -> tuple[float, str]:
    """Deterministic page order: mtime descending, filename ascending."""
    mtime, path, _size = item
    return (-mtime, path.name)


def _binding_hash(
    user_id: str,
    thread_id: str,
    query: str | None,
    extensions: frozenset[str] | None,
    current_run_filenames: set[str],
) -> str:
    """Hash trusted context that a continuation must stay bound to.

    Values are hashed (not embedded) so a decoded cursor never contains raw
    user or thread identifiers.
    """
    payload = {
        "u": user_id,
        "t": thread_id,
        "q": query,
        "e": sorted(extensions) if extensions else None,
        "x": sorted(current_run_filenames),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _listing_fingerprint(candidates: list[tuple[float, Path, int]]) -> str:
    """Fingerprint of the filtered listing so add/delete/rename invalidates."""
    h = hashlib.sha256()
    for mtime, path, size in sorted(candidates, key=_sort_key):
        h.update(f"{path.name}\x00{mtime:.6f}\x00{size}\n".encode("utf-8", errors="surrogateescape"))
    return h.hexdigest()


def _encode_cursor(binding: str, fingerprint: str, last_key: tuple[float, str]) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "b": binding,
        "f": fingerprint,
        "k": [last_key[0], last_key[1]],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, str, tuple[float, str]] | None:
    """Decode a continuation token, or return None when it is unusable."""
    if not isinstance(cursor, str):
        return None
    stripped = cursor.strip()
    if not stripped or len(stripped) > _MAX_CURSOR_CHARS:
        return None
    try:
        padded = stripped + "=" * (-len(stripped) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        return None
    binding = payload.get("b")
    fingerprint = payload.get("f")
    key = payload.get("k")
    if not isinstance(binding, str) or not isinstance(fingerprint, str):
        return None
    if not isinstance(key, list) or len(key) != 2:
        return None
    mtime, filename = key
    if not isinstance(mtime, (int, float)) or isinstance(mtime, bool) or not isinstance(filename, str):
        return None
    return binding, fingerprint, (float(mtime), filename)


def _invalid_cursor_result(message: str = _INVALID_CURSOR_MESSAGE) -> dict:
    return {
        "files": [],
        "total_count": 0,
        "cursor_invalid": True,
        "message": message,
    }


def _list_uploaded_files_impl(
    include_outline: bool | list[str] = False,
    max_results: int = _DEFAULT_MAX_RESULTS,
    runtime: Runtime | None = None,
    *,
    query: str | None = None,
    extensions: list[str] | None = None,
    cursor: str | None = None,
    _paths: Any | None = None,
) -> dict:
    """Core implementation — testable without the @tool wrapper."""
    if runtime is None:
        return {"files": [], "message": "No runtime context available."}

    thread_id = _resolve_thread_id(runtime)
    if thread_id is None:
        return {"files": [], "message": "Thread not found."}

    decoded_cursor: tuple[str, str, tuple[float, str]] | None = None
    if cursor is not None and (not isinstance(cursor, str) or cursor.strip()):
        decoded_cursor = _decode_cursor(cursor) if isinstance(cursor, str) else None
        if decoded_cursor is None:
            return _invalid_cursor_result()

    user_id = _resolve_user_id(runtime)
    paths = _paths or get_paths()
    uploads_dir = paths.sandbox_uploads_dir(thread_id, user_id=user_id)

    if not uploads_dir.exists():
        if decoded_cursor is not None:
            return _invalid_cursor_result("Cursor is no longer valid because the upload listing changed. Restart without a cursor to request the first page.")
        return {"files": [], "message": "No uploads directory for this thread."}

    # Resolve the set of filenames uploaded in the current run so we can exclude them.
    current_run_filenames: set[str] = set()
    try:
        state = runtime.state
        uploaded = state.get("uploaded_files") if isinstance(state, dict) else getattr(state, "uploaded_files", None)
        if isinstance(uploaded, list):
            for entry in uploaded:
                if isinstance(entry, dict) and entry.get("filename"):
                    current_run_filenames.add(entry["filename"])
    except Exception:
        logger.warning(
            "Failed to read uploaded_files from runtime.state; current-run files may appear in list_uploaded_files results",
            exc_info=True,
        )

    # Normalize max_results
    max_results = max(1, min(max_results, _MAX_MAX_RESULTS))

    # Normalize include_outline
    if isinstance(include_outline, bool):
        outline_for_all: bool = include_outline
        outline_filenames: set[str] = set()
    else:
        outline_for_all = False
        outline_filenames = set(include_outline)

    query_filter = _normalize_query(query)
    extension_filter = _normalize_extensions(extensions)

    # Collect historical files (mtime desc, filename asc).
    # Skip .md files that are conversion artifacts (have a same-stem non-.md sibling).
    candidates: list[tuple[float, Path, int]] = []
    try:
        # Collect file entries once to build the name set and iterate.
        entries = [e for e in os.scandir(uploads_dir) if e.is_file() and not e.is_symlink() and not is_upload_staging_file(e.name)]
        all_names: set[str] = {e.name for e in entries}

        for entry in entries:
            if entry.name in current_run_filenames:
                continue
            # Skip .md files that are conversion artifacts of another file.
            # Known limitation: if a user manually uploads both report.pdf and
            # report.md, the .md is hidden as a "conversion artifact".  This is
            # acceptable for the MVP — triggering this requires uploading files
            # whose stems collide with converted documents, which is rare.
            if entry.name.endswith(".md"):
                stem = entry.name[:-3]  # remove ".md"
                non_md_siblings = {n for n in all_names if n != entry.name and Path(n).stem == stem}
                if non_md_siblings:
                    continue
            stat = entry.stat()
            candidates.append((stat.st_mtime, Path(entry.path), stat.st_size))
    except OSError:
        if decoded_cursor is not None:
            return _invalid_cursor_result("Cursor is no longer valid because the upload listing changed. Restart without a cursor to request the first page.")
        return {"files": [], "message": f"Failed to read uploads directory: {uploads_dir}"}

    if not candidates:
        if decoded_cursor is not None:
            return _invalid_cursor_result("Cursor is no longer valid because the upload listing changed. Restart without a cursor to request the first page.")
        return {"files": [], "message": "No historical uploaded files in this thread."}

    if query_filter is not None or extension_filter is not None:
        candidates = [item for item in candidates if _matches_filters(item[1].name, item[1].suffix, query_filter, extension_filter)]
        if not candidates:
            if decoded_cursor is not None:
                return _invalid_cursor_result("Cursor is no longer valid because the upload listing changed. Restart without a cursor to request the first page.")
            return {
                "files": [],
                "total_count": 0,
                "message": "No uploaded files matched the given filters.",
            }

    candidates.sort(key=_sort_key)

    binding = _binding_hash(user_id, thread_id, query_filter, extension_filter, current_run_filenames)
    fingerprint = _listing_fingerprint(candidates)
    total_count = len(candidates)
    page_start = 0

    if decoded_cursor is not None:
        cursor_binding, cursor_fingerprint, last_key = decoded_cursor
        if cursor_binding != binding or cursor_fingerprint != fingerprint:
            return _invalid_cursor_result()
        cursor_key = (-last_key[0], last_key[1])
        page_start = 0
        for index, item in enumerate(candidates):
            if _sort_key(item) > cursor_key:
                page_start = index
                break
        else:
            page_start = len(candidates)

    page = candidates[page_start : page_start + max_results]
    remaining_after_page = candidates[page_start + max_results :]
    has_more = bool(remaining_after_page)
    omitted_paths = [p.name for _, p, _ in remaining_after_page]

    files: list[dict] = []
    for _, file_path, st_size in page:
        filename = file_path.name
        file_info: dict = {
            "filename": neutralize_untrusted_tags(filename),
            "size": st_size,
            "path": neutralize_untrusted_tags(f"/mnt/user-data/uploads/{filename}"),
            "extension": neutralize_untrusted_tags(file_path.suffix),
        }

        should_include_outline = outline_for_all or filename in outline_filenames
        if should_include_outline:
            outline, preview = extract_outline_for_file(file_path)
            if outline:
                file_info["outline"] = [{**entry, "title": neutralize_untrusted_tags(entry["title"])} if "title" in entry else entry for entry in outline]
            if preview:
                file_info["outline_preview"] = [neutralize_untrusted_tags(p) for p in preview]

        files.append(file_info)

    result: dict = {
        "files": files,
        "total_count": total_count,
    }

    if has_more:
        result["truncated"] = True
        result["omitted_summary"] = _format_omitted_summary(omitted_paths)
        last_mtime, last_path, _ = page[-1]
        result["next_cursor"] = _encode_cursor(binding, fingerprint, (last_mtime, last_path.name))
    elif decoded_cursor is not None and not page:
        result["message"] = "No more historical uploaded files match this cursor. Restart without a cursor to request the first page."
        return result

    if files:
        result["message"] = f"Found {total_count} historical file(s)."
    elif decoded_cursor is None:
        result["message"] = "No historical uploaded files in this thread."

    return result


@tool
def list_uploaded_files(
    runtime: Runtime,
    include_outline: Annotated[
        bool | list[str],
        "Control which files get their document outline (headings/preview) returned. "
        "False (default): no outline for any file — just filename, size, and path. "
        "True: include outline/preview for every .md-convertible file. "
        'list of filenames: include outline/preview only for those specific files (e.g. ["report.md", "data.csv"]).',
    ] = False,
    max_results: Annotated[
        int,
        "Maximum number of files to return (default 20, max 100).",
    ] = _DEFAULT_MAX_RESULTS,
    query: Annotated[
        str | None,
        "Optional case-insensitive substring to match against the filename only (not the virtual path). Omit or leave blank to skip name filtering.",
    ] = None,
    extensions: Annotated[
        list[str] | None,
        'Optional file extensions to keep, e.g. ["pdf", ".PNG"]. With or without a leading dot; matching is case-insensitive. Combined with query using AND. Omit to skip type filtering.',
    ] = None,
    cursor: Annotated[
        str | None,
        "Opaque continuation token from a previous response's next_cursor. "
        "Repeat the same query/extensions (and page size if you changed it) with this token to fetch the next page. "
        "Omit for the first page. If the tool reports an invalid or stale cursor, restart without a cursor.",
    ] = None,
) -> dict:
    """Discover historical uploaded files available in this thread.

    Returns files that were uploaded in PREVIOUS turns — files uploaded in the
    current run are excluded (they are already listed in <current_uploads>).

    Use this tool when:
    - The user refers to previously uploaded files without naming them (e.g. "analyze those PDFs I uploaded before")
    - You need to check what files are available in this thread
    - You are starting work on a thread and want an overview of available data

    Skip this tool when:
    - The user names a specific file — use read_file or grep directly with the path
    - The file was uploaded in the current run — it's already in <current_uploads>

    Optional filters (`query`, `extensions`) run before the max_results cap, so
    older matching files are not displaced by newer unrelated uploads.

    When more matches remain after a page, the response includes `next_cursor`.
    Repeat the same filters with that cursor until no further page is available.
    """
    return _list_uploaded_files_impl(
        include_outline=include_outline,
        max_results=max_results,
        runtime=runtime,
        query=query,
        extensions=extensions,
        cursor=cursor,
    )
