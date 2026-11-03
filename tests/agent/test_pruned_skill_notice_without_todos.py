"""A compaction with no task list still tells the model to reload the skills it pruned.

The reload notice (#84718) rode only on the todo snapshot. A long session that never used the
todo tool kept its ``[SKILL_PRUNED: ...]`` markers only inside the summary's "## Pruned Skills"
block, which the next turn reads as history; measured on one long-running gateway profile, 31
of 33 skills pruned at boundaries without a task list were not reloaded within the next ten
tool calls.
"""

from __future__ import annotations

from agent.context_compressor import _skill_pruned_marker
from agent.conversation_compression import (
    _PRUNED_SKILL_RELOAD_NOTICE_HEADER,
    _is_real_user_message,
    _pruned_skill_reload_notice,
    _strip_stale_todo_snapshot,
)
from hermes_state import SessionDB
from tests.agent.test_skill_todo_retention_parity import _build_agent_with_db, _msgs
from tools.todo_tool import TODO_INJECTION_HEADER


def _summary(*skills: str) -> str:
    return "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n" + "\n".join(_skill_pruned_marker(s) for s in skills)


def _compact(tmp_path, returned, *, todos=None, session="PARENT_NOTICE_NO_TODO"):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session, source="cli")
    agent = _build_agent_with_db(db, session)
    agent.context_compressor.compress.return_value = returned
    agent._todo_store._items = list(todos or [])
    original = _msgs()
    compressed, _ = agent._compress_context(original, "sys", approx_tokens=120_000)
    db.close()
    return compressed


def _rows_with(compressed, needle):
    return [m for m in compressed if isinstance(m, dict) and needle in str(m.get("content") or "")]


class TestNoticeWithoutTodos:
    def test_notice_folds_into_the_tail_user_turn(self, tmp_path):
        original_tail = _msgs()[-1]["content"]
        compressed = _compact(
            tmp_path,
            [
                {"role": "user", "content": _summary("vg-code-reel", "khope-videogen")},
                {"role": "assistant", "content": original_tail},
                {"role": "user", "content": "make the background a sky gradient"},
            ],
        )
        tail = str(compressed[-1]["content"])
        assert tail.startswith("make the background a sky gradient")
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER in tail
        assert "skill_view(name='vg-code-reel')" in tail
        assert "skill_view(name='khope-videogen')" in tail
        assert TODO_INJECTION_HEADER not in tail
        # Worded for a boundary with no task list.
        assert "task list above" not in tail

    def test_standalone_row_is_scaffolding_not_user_intent(self, tmp_path):
        compressed = _compact(
            tmp_path,
            [
                {"role": "user", "content": _summary("frontend-design")},
                {"role": "assistant", "content": "acknowledged"},
            ],
            session="PARENT_NOTICE_NO_TODO_SYNTH",
        )
        rows = _rows_with(compressed, _PRUNED_SKILL_RELOAD_NOTICE_HEADER)
        notice_rows = [r for r in rows if "CONTEXT COMPACTION" not in str(r["content"])[:280]]
        assert len(notice_rows) == 1
        assert notice_rows[0].get("_todo_snapshot_synthetic") is True
        assert not _is_real_user_message(notice_rows[0])

    def test_no_markers_no_notice(self, tmp_path):
        compressed = _compact(
            tmp_path,
            [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary without pruned skills"},
                {"role": "assistant", "content": _msgs()[-1]["content"]},
                {"role": "user", "content": "carry on"},
            ],
            session="PARENT_NOTICE_NO_TODO_CLEAN",
        )
        assert not _rows_with(compressed, _PRUNED_SKILL_RELOAD_NOTICE_HEADER)
        assert any(m.get("content") == "carry on" for m in compressed if isinstance(m, dict))

    def test_repeated_boundaries_keep_one_fresh_notice(self, tmp_path):
        stale = "older human text\n\n" + _pruned_skill_reload_notice(
            [{"role": "user", "content": _summary("stale-skill")}], with_todos=False
        )
        compressed = _compact(
            tmp_path,
            [
                {"role": "user", "content": _summary("fresh-skill")},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": stale},
                {"role": "assistant", "content": _msgs()[-1]["content"]},
                {"role": "user", "content": "next request"},
            ],
            session="PARENT_NOTICE_NO_TODO_RESTRIP",
        )
        notice_rows = [
            r for r in _rows_with(compressed, _PRUNED_SKILL_RELOAD_NOTICE_HEADER)
            if "CONTEXT COMPACTION" not in str(r["content"])[:280]
        ]
        assert len(notice_rows) == 1
        text = str(notice_rows[0]["content"])
        assert text.startswith("next request")
        assert "skill_view(name='fresh-skill')" in text
        assert "older human text" in str(compressed)
        assert "skill_view(name='stale-skill')" not in "".join(str(r["content"]) for r in notice_rows)

    def test_todo_boundary_keeps_the_coupled_wording(self, tmp_path):
        compressed = _compact(
            tmp_path,
            [
                {"role": "user", "content": _summary("plan")},
                {"role": "assistant", "content": _msgs()[-1]["content"]},
                {"role": "user", "content": "tail"},
            ],
            todos=[{"id": "t1", "content": "ship it", "status": "pending"}],
            session="PARENT_NOTICE_WITH_TODO",
        )
        tail = str(compressed[-1]["content"])
        assert tail.count(_PRUNED_SKILL_RELOAD_NOTICE_HEADER) == 1
        assert tail.index(TODO_INJECTION_HEADER) < tail.index(_PRUNED_SKILL_RELOAD_NOTICE_HEADER)
        assert "task list above" in tail


class TestStripReadsANoticeAlone:
    def test_strip_removes_a_notice_folded_without_a_snapshot(self):
        content = "real user words\n\n" + _PRUNED_SKILL_RELOAD_NOTICE_HEADER + "\nreload skill_view(name='x').'"
        assert _strip_stale_todo_snapshot(content) == "real user words"

    def test_strip_list_content_keeps_non_text_parts(self):
        content = [
            {"type": "text", "text": "words\n\n" + _PRUNED_SKILL_RELOAD_NOTICE_HEADER + "\nreload."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]
        stripped = _strip_stale_todo_snapshot(content)
        assert stripped == [
            {"type": "text", "text": "words"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]

    def test_todo_header_still_wins_when_both_present(self):
        content = (
            "words\n\n" + TODO_INJECTION_HEADER + "\n- [ ] t1. a (pending)\n\n"
            + _PRUNED_SKILL_RELOAD_NOTICE_HEADER + "\nreload."
        )
        assert _strip_stale_todo_snapshot(content) == "words"
