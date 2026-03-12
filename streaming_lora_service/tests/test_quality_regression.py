from __future__ import annotations

import unittest

from streaming_lora_service.quality_regression import (
    PathMetrics,
    ValidationCase,
    ValidationCaseResult,
    audio_duration_seconds,
    build_case_warnings,
    build_summary,
)


class QualityRegressionTests(unittest.TestCase):
    def test_audio_duration_seconds_for_pcm16_mono(self) -> None:
        self.assertAlmostEqual(audio_duration_seconds(b"\x00\x00" * 24000, sample_rate=24000), 1.0)

    def test_build_case_warnings_flags_long_streaming_and_non_eos(self) -> None:
        result = ValidationCaseResult(
            case=ValidationCase(id="case_1", text="你好"),
            offline_non_streaming=PathMetrics(
                path="offline_non_streaming",
                sample_rate=24000,
                channels=1,
                total_audio_bytes=96000,
                duration_s=2.0,
            ),
            http_non_streaming=PathMetrics(
                path="http_non_streaming",
                sample_rate=24000,
                channels=1,
                total_audio_bytes=96000,
                duration_s=2.0,
                codec_steps=100,
            ),
            http_streaming_runtime=PathMetrics(
                path="http_streaming_runtime",
                sample_rate=24000,
                channels=1,
                total_audio_bytes=240000,
                duration_s=5.0,
                generated_steps=260,
                finish_reason="length",
            ),
            websocket_realtime=PathMetrics(
                path="websocket_realtime",
                sample_rate=24000,
                channels=1,
                total_audio_bytes=260000,
                duration_s=5.42,
                generated_steps=280,
                finish_reason="length",
            ),
            warnings=[],
        )

        warnings = build_case_warnings(result, max_stream_to_offline_ratio=1.5)

        self.assertGreaterEqual(len(warnings), 3)
        self.assertTrue(any("finish_reason" in item for item in warnings))
        self.assertTrue(any("duration ratio" in item for item in warnings))
        self.assertTrue(any("generated_steps ratio" in item for item in warnings))
        self.assertTrue(any("different audio sizes" in item for item in warnings))

    def test_build_summary_counts_warning_cases(self) -> None:
        result_ok = ValidationCaseResult(
            case=ValidationCase(id="ok", text="hello"),
            offline_non_streaming=PathMetrics("offline_non_streaming", 24000, 1, 100, 0.1),
            http_non_streaming=PathMetrics("http_non_streaming", 24000, 1, 100, 0.1),
            http_streaming_runtime=PathMetrics("http_streaming_runtime", 24000, 1, 100, 0.1),
            websocket_realtime=PathMetrics("websocket_realtime", 24000, 1, 100, 0.1),
            warnings=[],
        )
        result_warn = ValidationCaseResult(
            case=ValidationCase(id="warn", text="hello"),
            offline_non_streaming=PathMetrics("offline_non_streaming", 24000, 1, 100, 0.1),
            http_non_streaming=PathMetrics("http_non_streaming", 24000, 1, 100, 0.1),
            http_streaming_runtime=PathMetrics("http_streaming_runtime", 24000, 1, 100, 0.1),
            websocket_realtime=PathMetrics("websocket_realtime", 24000, 1, 100, 0.1),
            warnings=["boom"],
        )

        summary = build_summary([result_ok, result_warn])

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["warning_count"], 1)
        self.assertEqual(summary["warning_case_count"], 1)
        self.assertEqual(summary["case_ids_with_warnings"], ["warn"])


if __name__ == "__main__":
    unittest.main()