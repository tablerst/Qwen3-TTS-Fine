from __future__ import annotations

import unittest

from streaming_lora_service.concurrent_benchmark import (
    LoadTestRequestMetrics,
    select_case,
    summarize_load_results,
)
from streaming_lora_service.ws_benchmark import BenchmarkCase


class ConcurrentBenchmarkHelpersTests(unittest.TestCase):
    def test_select_case_round_robin(self) -> None:
        cases = [BenchmarkCase(id="a", text="1"), BenchmarkCase(id="b", text="2")]
        self.assertEqual(select_case(cases, 1).id, "a")
        self.assertEqual(select_case(cases, 2).id, "b")
        self.assertEqual(select_case(cases, 3).id, "a")

    def test_summarize_load_results_counts_success_and_failure(self) -> None:
        results = [
            LoadTestRequestMetrics(
                request_index=1,
                case_id="a",
                transport="http-streaming",
                model="m",
                voice="v",
                started_offset_ms=0.0,
                ended_offset_ms=100.0,
                ttft_ms=30.0,
                total_elapsed_ms=100.0,
                audio_duration_s=1.0,
                rtf=0.1,
                audio_chunks=2,
                total_audio_bytes=48000,
                success=True,
            ),
            LoadTestRequestMetrics(
                request_index=2,
                case_id="a",
                transport="http-streaming",
                model="m",
                voice="v",
                started_offset_ms=0.0,
                ended_offset_ms=20.0,
                ttft_ms=None,
                total_elapsed_ms=20.0,
                audio_duration_s=0.0,
                rtf=None,
                audio_chunks=0,
                total_audio_bytes=0,
                success=False,
                error="boom",
            ),
        ]
        summary = summarize_load_results(results)
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["ttft_ms"]["p50"], 30.0)


if __name__ == "__main__":
    unittest.main()