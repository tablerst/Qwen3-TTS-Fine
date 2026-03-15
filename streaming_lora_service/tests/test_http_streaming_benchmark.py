from __future__ import annotations

import unittest

from streaming_lora_service.http_streaming_benchmark import (
    HttpBenchmarkRunMetrics,
    extract_stream_payload_fields,
    summarize_http_case_runs,
)


class HttpStreamingBenchmarkHelpersTests(unittest.TestCase):
    def test_extract_stream_payload_fields_for_audio_chunk(self) -> None:
        payload = {
            "request_id": "req_1",
            "status_code": 200,
            "output": {
                "finish_reason": "null",
                "audio": {
                    "data": "Zm9v",
                    "url": "",
                    "id": "",
                },
            },
            "usage": {"characters": 12},
        }
        fields = extract_stream_payload_fields(payload)
        self.assertEqual(fields["request_id"], "req_1")
        self.assertEqual(fields["audio_data"], "Zm9v")
        self.assertEqual(fields["finish_reason"], "null")
        self.assertEqual(fields["usage_characters"], 12)

    def test_extract_stream_payload_fields_for_final_chunk(self) -> None:
        payload = {
            "request_id": "req_2",
            "output": {
                "finish_reason": "stop",
                "audio": {
                    "data": "",
                    "url": "http://127.0.0.1:9010/v1/audio/a1",
                    "id": "a1",
                },
            },
            "usage": {"characters": 8},
        }
        fields = extract_stream_payload_fields(payload)
        self.assertEqual(fields["audio_url"], "http://127.0.0.1:9010/v1/audio/a1")
        self.assertEqual(fields["audio_id"], "a1")
        self.assertEqual(fields["finish_reason"], "stop")

    def test_summarize_http_case_runs(self) -> None:
        runs = [
            HttpBenchmarkRunMetrics(
                case_id="case1",
                iteration=1,
                model="m",
                voice="v",
                endpoint="/v1/tts",
                sample_rate=24000,
                response_headers_ms=5.0,
                ttft_ms=100.0,
                final_chunk_ms=500.0,
                total_elapsed_ms=510.0,
                total_audio_bytes=48000,
                audio_duration_s=1.0,
                rtf=0.51,
                audio_chunks=3,
                request_id="r1",
                final_finish_reason="stop",
                audio_id="a1",
                audio_url="u1",
                usage_characters=10,
            )
        ]
        summary = summarize_http_case_runs(runs)
        self.assertEqual(summary["run_count"], 1)
        self.assertEqual(summary["ttft_ms"]["p50"], 100.0)
        self.assertEqual(summary["rtf"]["mean"], 0.51)


if __name__ == "__main__":
    unittest.main()