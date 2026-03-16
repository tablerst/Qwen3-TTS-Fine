from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from streaming_lora_service.app.audio_utils import float_audio_to_pcm16le_bytes
from streaming_lora_service.app.server import BundleSpeechService
from streaming_lora_service.app.voice_registry import VoiceRegistry
from streaming_lora_service.quality_regression import (
    PathMetrics,
    ValidationCase,
    ValidationCaseResult,
    audio_duration_seconds,
    build_chunk_boundaries,
    build_case_warnings,
    build_codec_diagnostics,
    build_summary,
    build_waveform_diagnostics,
    compare_codec_sequences,
    compare_waveforms,
    collect_streaming_sampler_full_decode,
)
from streaming_lora_service.tests.test_streaming_generator import FakeQwen3TTS


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

    def test_collect_streaming_sampler_full_decode_returns_full_decode_audio(self) -> None:
        qwen3tts = FakeQwen3TTS()
        service = BundleSpeechService(
            SimpleNamespace(
                loaded_bundle=SimpleNamespace(qwen3tts=qwen3tts),
                voice_registry=VoiceRegistry.single_voice(
                    voice_alias="yachiyo_formal",
                    speaker_name="inference_speaker",
                    model_alias="qwen3-tts-flash-realtime",
                ),
                public_model_alias="qwen3-tts-flash-realtime",
                default_voice_alias="yachiyo_formal",
                chunk_steps=2,
                left_context_steps=1,
                first_chunk_steps=None,
                crossfade_samples=0,
                samples_per_step=4,
            )
        )

        collected = collect_streaming_sampler_full_decode(
            service,
            ValidationCase(id="diag", text="你好，诊断。", language_type="Chinese"),
        )

        expected_audio = np.asarray([0.11] * 4 + [0.12] * 4 + [0.13] * 4, dtype=np.float32)
        self.assertEqual(collected.audio_bytes, float_audio_to_pcm16le_bytes(expected_audio))
        self.assertEqual(collected.metrics.path, "streaming_sampler_full_decode")
        self.assertEqual(collected.metrics.sample_rate, 24000)
        self.assertEqual(collected.metrics.generated_steps, 3)
        self.assertEqual(collected.metrics.codec_steps, 3)
        self.assertEqual(collected.metrics.finish_reason, "eos")
        self.assertEqual(collected.metrics.delta_chunks, 2)
        self.assertIsNotNone(collected.metrics.runtime_metrics)
        assert collected.metrics.runtime_metrics is not None
        self.assertEqual(collected.metrics.runtime_metrics["generated_steps"], 3)
        self.assertIn("total_forward_ms", collected.metrics.runtime_metrics)
        self.assertIn("avg_forward_ms", collected.metrics.runtime_metrics)
        self.assertEqual(collected.codec_tokens, ((11, 31, 51), (12, 32, 52), (13, 33, 53)))
        self.assertEqual(collected.chunk_boundaries_samples, (8,))

    def test_compare_codec_sequences_reports_first_divergence(self) -> None:
        comparison = compare_codec_sequences(
            "http_non_streaming",
            ((1, 10, 11), (2, 12, 13), (3, 14, 15)),
            "streaming_sampler_full_decode",
            ((1, 10, 11), (2, 99, 13), (4, 16, 17)),
        )

        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertEqual(comparison.shared_prefix_steps, 1)
        self.assertEqual(comparison.first_divergence_step, 1)
        self.assertEqual(comparison.reference_step_tokens, [2, 12, 13])
        self.assertEqual(comparison.candidate_step_tokens, [2, 99, 13])
        self.assertFalse(comparison.identical)

    def test_build_codec_diagnostics_collects_available_pairs(self) -> None:
        diagnostics = build_codec_diagnostics(
            {
                "http_non_streaming": SimpleNamespace(codec_tokens=((1, 2, 3), (4, 5, 6))),
                "streaming_sampler_full_decode": SimpleNamespace(codec_tokens=((1, 2, 3), (7, 8, 9))),
                "http_streaming_runtime": SimpleNamespace(codec_tokens=((1, 2, 3), (7, 8, 9))),
                "websocket_realtime": SimpleNamespace(codec_tokens=((1, 2, 3), (7, 8, 9))),
            }
        )

        self.assertIn("http_non_vs_streaming_sampler_full_decode", diagnostics)
        self.assertIn("streaming_sampler_full_decode_vs_http_streaming_runtime", diagnostics)
        self.assertIn("http_streaming_runtime_vs_websocket_realtime", diagnostics)
        self.assertEqual(diagnostics["http_non_vs_streaming_sampler_full_decode"].first_divergence_step, 1)
        self.assertTrue(diagnostics["streaming_sampler_full_decode_vs_http_streaming_runtime"].identical)

    def test_build_chunk_boundaries_uses_chunk_sizes(self) -> None:
        boundaries = build_chunk_boundaries([b"\x00\x00" * 4, b"\x00\x00" * 2, b"\x00\x00" * 3])

        self.assertEqual(boundaries, (4, 6))

    def test_compare_waveforms_reports_global_and_boundary_metrics(self) -> None:
        reference = float_audio_to_pcm16le_bytes(np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        candidate = float_audio_to_pcm16le_bytes(np.asarray([0.0, 0.1, 0.25, 0.35, 0.4], dtype=np.float32))

        comparison = compare_waveforms(
            "reference",
            reference,
            "candidate",
            candidate,
            boundary_samples=(2,),
            boundary_window_samples=1,
        )

        self.assertEqual(comparison.compared_samples, 5)
        self.assertGreater(comparison.mean_abs_diff, 0.0)
        self.assertGreater(comparison.rms_diff, 0.0)
        self.assertLess(comparison.pearson_corr, 1.0)
        self.assertEqual(len(comparison.boundary_windows), 1)
        self.assertEqual(comparison.boundary_windows[0].boundary_sample, 2)

    def test_build_waveform_diagnostics_uses_candidate_chunk_boundaries(self) -> None:
        base = float_audio_to_pcm16le_bytes(np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32))
        shifted = float_audio_to_pcm16le_bytes(np.asarray([0.0, 0.1, 0.25, 0.3], dtype=np.float32))

        diagnostics = build_waveform_diagnostics(
            {
                "http_non_streaming": SimpleNamespace(
                    audio_bytes=base,
                    metrics=PathMetrics("http_non_streaming", 24000, 1, len(base), 0.0),
                    chunk_boundaries_samples=(),
                ),
                "streaming_sampler_full_decode": SimpleNamespace(
                    audio_bytes=base,
                    metrics=PathMetrics("streaming_sampler_full_decode", 24000, 1, len(base), 0.0),
                    chunk_boundaries_samples=(2,),
                ),
                "http_streaming_runtime": SimpleNamespace(
                    audio_bytes=shifted,
                    metrics=PathMetrics("http_streaming_runtime", 24000, 1, len(shifted), 0.0),
                    chunk_boundaries_samples=(2,),
                ),
                "websocket_realtime": SimpleNamespace(
                    audio_bytes=shifted,
                    metrics=PathMetrics("websocket_realtime", 24000, 1, len(shifted), 0.0),
                    chunk_boundaries_samples=(2,),
                ),
            },
            boundary_window_samples=1,
        )

        self.assertIn("streaming_sampler_full_decode_vs_http_streaming_runtime", diagnostics)
        self.assertEqual(
            diagnostics["streaming_sampler_full_decode_vs_http_streaming_runtime"].boundary_windows[0].boundary_sample,
            2,
        )


if __name__ == "__main__":
    unittest.main()