"""Boundary scaffolding folded into the post-compaction tail: the todo snapshot and the pruned-skill
reload notice, plus the strip that removes a stale copy at the next boundary.

Split out of ``agent/conversation_compression.py`` (code-health line cap); that module re-exports every
name here, so existing imports keep working.
"""

from __future__ import annotations

import contextlib
from typing import Any


def _message_text(message: Any) -> str:
    from agent.conversation_compression import _message_text as text
    return text(message)


def _is_real_user_message(message: Any) -> bool:
    from agent.conversation_compression import _is_real_user_message as real
    return real(message)


def _scaffold_block_start(text: str) -> int:
    """Offset of the boundary scaffold block in ``text`` (todo snapshot, or a reload notice folded without
    one), or -1. The todo header wins when both are present: the notice then rides after it."""
    from tools.todo_tool import TODO_INJECTION_HEADER
    idx = text.find(TODO_INJECTION_HEADER)
    if idx != -1:
        return idx
    return text.find(_PRUNED_SKILL_RELOAD_NOTICE_HEADER)


def _strip_stale_todo_snapshot(content: Any) -> Any:
    """Remove a previously merged todo-snapshot block from message content.
    Snapshots are appended to the trailing user turn, so a surviving header is stale; stripping before
    re-injection prevents accumulation across boundaries."""
    if isinstance(content, str):
        idx = _scaffold_block_start(content)
        if idx == -1:
            return content
        return content[:idx].rstrip()
    if isinstance(content, list):
        cleaned = []
        for part in content:
            text = str(part.get("text") or "") if isinstance(part, dict) and part.get("type") == "text" else ""
            idx = _scaffold_block_start(text) if text else -1
            if idx == -1:
                cleaned.append(part)
            elif stripped := text[:idx].rstrip():
                cleaned.append({**part, "text": stripped})
        return cleaned
    return content


def _todo_snapshot_is_only_content(content: Any, stripped: Any) -> bool:
    """Return whether stripping the snapshot leaves no structured content.
    Text snapshots trail a string; structured ones occupy their own text part, so only an empty remainder
    proves the row was scaffolding alone. Text extraction is deliberately not used: image, audio and other
    non-text parts must survive."""
    if isinstance(content, str) and isinstance(stripped, str):
        return not stripped.strip()
    if isinstance(content, list) and isinstance(stripped, list):
        return not stripped
    return False


def _replace_message_content(message: dict, content: Any) -> None:
    """Rewrite message content without allowing an old API sidecar to replay."""
    from agent.turn_context import drop_stale_api_content
    message["content"] = content
    drop_stale_api_content(message)


# Compaction re-injects the todo list verbatim but prunes skills to markers, so
# couple them: tell the model to reload pruned skills BEFORE acting on tasks.
# Lives after TODO_INJECTION_HEADER so it strips with the snapshot next time.
_PRUNED_SKILL_RELOAD_NOTICE_HEADER = "[Skills pruned during compression — reload before acting on these tasks]"


def _pruned_skill_reload_notice(compressed: list, *, with_todos: bool = True) -> str:
    """Reload notice for skills whose bodies were pruned, or ``""``.
    Scans ``[SKILL_PRUNED: ...]`` markers in the post-compression transcript; first-seen order, deduplicated,
    capped at ``_MAX_PRUNED_SKILL_MARKERS``. ``with_todos=False`` words it for a boundary with no task list."""
    from agent.context_compressor import _MAX_PRUNED_SKILL_MARKERS, _extract_pruned_skill_names
    names: list = []
    for message in compressed:
        if not isinstance(message, dict):
            continue
        for name in _extract_pruned_skill_names(_message_text(message)):
            if name not in names:
                names.append(name)
    del names[_MAX_PRUNED_SKILL_MARKERS:]
    if not names:
        return ""
    calls = "; ".join(f"skill_view(name='{name}')" for name in names)
    if not with_todos:
        return (
            f"{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\n"
            "The skill instructions loaded before this boundary were pruned to markers. Before the next step "
            f"that depends on them, reload them first: {calls}."
        )
    return (
        f"{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\n"
        "The task list above crossed the compression boundary verbatim, but "
        "the skill instructions that governed it were pruned. Before "
        f"executing any preserved task that depends on these skills, reload "
        f"them first: {calls}. After reloading, re-check that each pending "
        "task is still justified — findings recorded before the boundary may have invalidated it."
    )


def _drop_stale_reload_notices(agent: Any, compressed: list) -> None:
    """Remove reload notices an earlier boundary folded in without a task list (all but the tail row, which
    the fold below refreshes in place), so repeated compactions keep one notice."""
    for idx in range(len(compressed) - 2, -1, -1):
        message = compressed[idx]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if _PRUNED_SKILL_RELOAD_NOTICE_HEADER not in _message_text(message) or "CONTEXT COMPACTION" in _message_text(
            message
        )[:280]:
            continue
        stripped = _strip_stale_todo_snapshot(content)
        if stripped == content:
            continue
        if message.get("_todo_snapshot_synthetic") and _todo_snapshot_is_only_content(content, stripped):
            compressed.pop(idx)
            if idx < len(compressed):
                agent._repair_message_sequence(compressed)
        else:
            _replace_message_content(message, stripped)


def _fold_todo_snapshot(agent: Any, compressed: list) -> None:
    """Strip stale todo snapshots from ``compressed`` and fold the live one in (in place)."""
    todo_snapshot = agent._todo_store.format_for_injection()
    # Non-empty store (even all done) is authoritative: drop the old snapshot. A
    # truly empty store may be un-rehydrated post-compaction: keep the snapshot.
    _todo_has_items = getattr(agent._todo_store, "has_items", None)
    # Store may implement only format_for_injection(); unknown authority must
    # preserve the pending snapshot rather than risk deleting it.
    _todo_store_is_authoritative = False
    with contextlib.suppress(Exception):
        _todo_store_is_authoritative = bool(_todo_has_items()) if callable(_todo_has_items) else False
    if _todo_store_is_authoritative:
        for _todo_idx in range(len(compressed) - 1, -1, -1):
            _todo_message = compressed[_todo_idx]
            if not isinstance(_todo_message, dict) or _todo_message.get("role") != "user":
                continue
            _todo_content = _todo_message.get("content")
            _todo_stripped = _strip_stale_todo_snapshot(_todo_content)
            if _todo_stripped == _todo_content:
                continue
            if _todo_message.get("_todo_snapshot_synthetic") and _todo_snapshot_is_only_content(
                _todo_content, _todo_stripped
            ):
                compressed.pop(_todo_idx)
                if _todo_idx < len(compressed):
                    # A standalone snapshot can drift from the tail; deleting it may expose two
                    # assistant rows, so use the normal replay repair to keep metadata consistent.
                    agent._repair_message_sequence(compressed)
            else:
                _replace_message_content(_todo_message, _todo_stripped)
                # No longer todo-only scaffolding; other synthetic flags stay authoritative and
                # _is_real_user_message() recomputes provenance from content + flags.
                _todo_message.pop("_todo_snapshot_synthetic", None)
            break
    if todo_snapshot:
        # If this boundary pruned skill bodies, the policy behind the todos is gone:
        # add a reload notice after TODO_INJECTION_HEADER so both strip together.
        # Retention parity (#84718): the snapshot below re-injects the imperative verbatim. If this same
        # boundary pruned skill bodies to [SKILL_PRUNED: ...] markers, the policy that governed those tasks
        # is gone — couple a reload instruction to the snapshot so the imperative never crosses the boundary
        # alone.
        _reload_notice = _pruned_skill_reload_notice(compressed)
        if _reload_notice:
            todo_snapshot = f"{todo_snapshot}\n\n{_reload_notice}"
    else:
        # No task list, yet the boundary still pruned skill bodies: the markers then sit only inside the
        # summary's "## Pruned Skills" block, which the next turn reads as history, and the session goes
        # on without the skill. Fold the notice alone into the tail, through the same strip/merge path.
        # A kept (un-rehydrated) snapshot already in the transcript is left alone.
        from tools.todo_tool import TODO_INJECTION_HEADER
        if not any(
            isinstance(m, dict) and m.get("role") == "user" and TODO_INJECTION_HEADER in _message_text(m)
            for m in compressed
        ):
            todo_snapshot = _pruned_skill_reload_notice(compressed, with_todos=False)
            if todo_snapshot:
                _drop_stale_reload_notices(agent, compressed)
    if todo_snapshot:
        # Fold the snapshot into a trailing REAL user msg (no synthetic user/user pair);
        # strip old snapshots first. Scaffolding tails must not absorb it (provenance).
        # Any snapshot merged at an earlier boundary is stripped first so repeated compactions refresh
        # rather than accumulate todo state (#26981). Scaffolding tails (continuation marker, summary
        # handoff, a bare stale snapshot row) must never absorb the snapshot: merging would upgrade them to
        # "real user" evidence and break zero-user provenance (#69292), so those keep the flagged standalone
        # append and the real-user preservation pass continues to see todo scaffolding, not human intent.
        from agent.context_compressor import _append_text_to_content
        merged = False
        _tail = compressed[-1] if compressed and isinstance(compressed[-1], dict) else None
        if _tail is not None and _tail.get("role") == "user":
            _stripped = _strip_stale_todo_snapshot(_tail.get("content"))
            _probe = {key: value for key, value in _tail.items() if key != "content"}
            _probe["content"] = _stripped
            if _is_real_user_message(_probe):
                _snapshot_text = f"\n\n{todo_snapshot}" if isinstance(_stripped, str) and _stripped else todo_snapshot
                _replace_message_content(_tail, _append_text_to_content(_stripped, _snapshot_text))
                merged = True
            elif (
                _stripped != _tail.get("content") and not _message_text({"role": "user", "content": _stripped}).strip()
            ):
                # The tail was nothing but an earlier snapshot row —
                # refresh it in place instead of stacking a duplicate.
                _replace_message_content(_tail, todo_snapshot)
                _tail["_todo_snapshot_synthetic"] = True
                merged = True
        if not merged:
            compressed.append({"role": "user", "content": todo_snapshot, "_todo_snapshot_synthetic": True})
