from __future__ import annotations

import unittest

from streaming_lora_service.app.models import SessionOptions
from streaming_lora_service.app.runtime_session import RuntimeSession, RuntimeSessionError


class RuntimeSessionTests(unittest.TestCase):
    def make_session(self) -> RuntimeSession:
        return RuntimeSession(
            session_id="sess_test",
            options=SessionOptions(
                model="qwen3-tts-flash-realtime",
                voice="yachiyo_formal",
            ),
        )

    def test_append_and_commit_updates_buffers(self) -> None:
        session = self.make_session()

        session.append_text("你好，")
        session.append_text("欢迎使用。")
        committed = session.commit()

        self.assertEqual(committed, "你好，欢迎使用。")
        self.assertEqual(session.state.committed_text, "你好，欢迎使用。")
        self.assertEqual(session.state.pending_text_tail, "")
        self.assertEqual(session.state.committed_chunks, ["你好，欢迎使用。"])

    def test_clear_pending_text_restores_last_committed_state(self) -> None:
        session = self.make_session()
        session.append_text("第一句。")
        session.commit()
        session.append_text("第二句。")

        session.clear_pending_text()

        self.assertEqual(session.state.raw_text_buffer, "第一句。")
        self.assertEqual(session.state.pending_text_tail, "")

    def test_commit_requires_pending_text(self) -> None:
        session = self.make_session()

        with self.assertRaisesRegex(RuntimeSessionError, "No pending text"):
            session.commit()

    def test_finish_commits_tail_and_closes_session(self) -> None:
        session = self.make_session()
        session.append_text("最后一句。")

        final_chunk = session.finish()

        self.assertEqual(final_chunk, "最后一句。")
        self.assertTrue(session.state.finished)
        with self.assertRaisesRegex(RuntimeSessionError, "after session.finish"):
            session.append_text("不能再写")


if __name__ == "__main__":
    unittest.main()
