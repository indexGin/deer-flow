"""Deterministic package resource graph checks."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from deerflow.skills.package_paths import is_eval_fixture_path
from deerflow.skills.review.models import make_finding, normalize_relative_path

_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_PATH_TOKEN_RE = re.compile(r"(?<![\w./-])(?:references|scripts|templates|assets|evals)/[A-Za-z0-9._~/%+-]+")
_RESOURCE_DIRS = {"references", "scripts", "templates", "assets", "evals"}


def build_resource_graph(snapshot: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files = {str(entry["path"]): entry for entry in snapshot.get("files", [])}
    nodes = [{"path": path, "kind": files[path].get("kind", "unknown")} for path in sorted(files)]
    edges: set[tuple[str, str]] = set()
    missing: set[tuple[str, str]] = set()
    escaping: set[tuple[str, str]] = set()

    for path, entry in files.items():
        if is_eval_fixture_path(path):
            continue
        if entry.get("kind") != "text":
            continue
        content = str(entry.get("content") or "")
        for raw_ref in _extract_references(content):
            resolved = _resolve_reference(path, raw_ref)
            if resolved is None:
                continue
            if resolved == "__ESCAPES__":
                escaping.add((path, raw_ref))
            elif resolved in files:
                edges.add((path, resolved))
            else:
                missing.add((path, resolved))

    referenced = {target for _, target in edges}
    resource_paths = {path for path in files if PurePosixPath(path).parts and PurePosixPath(path).parts[0] in _RESOURCE_DIRS}
    orphans = sorted(resource_paths - referenced - {"evals/evals.json", "evals/trigger_eval_set.json"})
    orphans = [path for path in orphans if not is_eval_fixture_path(path)]

    findings: list[dict[str, Any]] = []
    for source, target in sorted(missing):
        findings.append(
            make_finding(
                "resource.missing",
                severity="warning",
                path=source,
                message=f"Referenced resource does not exist: {target}",
                remediation="Add the referenced file, correct the path, or remove the stale reference.",
                evidence=target,
            )
        )
    for source, raw_ref in sorted(escaping):
        findings.append(
            make_finding(
                "resource.escaping-link",
                severity="warning",
                path=source,
                message=f"Reference escapes the package boundary: {raw_ref}",
                remediation="Keep skill references package-relative and inside the skill directory.",
                evidence=raw_ref,
            )
        )
    for orphan in orphans:
        findings.append(
            make_finding(
                "resource.unreferenced",
                severity="warning",
                path=orphan,
                message="Resource is not reachable from SKILL.md or another referenced resource.",
                remediation="Reference the file with read-when guidance or remove it from the package.",
            )
        )

    graph = {
        "nodes": nodes,
        "edges": [{"source": source, "target": target} for source, target in sorted(edges)],
        "orphans": orphans,
    }
    return graph, findings


def _iter_markdown_link_targets(content: str):
    # A label cannot cross "]", so for each "](" the opening bracket is the
    # first "[" after the previous "]". Scanning one candidate per closer keeps
    # the cost linear instead of retrying the label scan at every "[" (#5714).
    pos = 0
    while True:
        closer = content.find("](", pos)
        if closer == -1:
            return
        previous = content.rfind("]", pos, closer)
        start = content.find("[", pos if previous == -1 else previous + 1, closer)
        match = _MARKDOWN_LINK_RE.match(content, start) if start != -1 else None
        if match is None:
            pos = closer + 1
            continue
        yield match.group(1)
        pos = match.end()


def _extract_references(content: str) -> set[str]:
    refs: set[str] = set()
    for target in _iter_markdown_link_targets(content):
        refs.add(target.split("#", 1)[0])
    for match in _CODE_SPAN_RE.finditer(content):
        token = match.group(1).strip()
        if "/" in token:
            refs.add(token)
    for match in _PATH_TOKEN_RE.finditer(content):
        refs.add(match.group(0))
    return refs


def _resolve_reference(source_path: str, raw_ref: str) -> str | None:
    ref = raw_ref.strip().strip("\"'")
    if not ref or ref.startswith("#") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", ref):
        return None
    try:
        if ref.startswith("/"):
            return "__ESCAPES__"
        base = PurePosixPath(source_path).parent
        if "://" in ref:
            return None
        candidate = (base / ref).as_posix()
        return normalize_relative_path(candidate)
    except ValueError:
        return "__ESCAPES__"
