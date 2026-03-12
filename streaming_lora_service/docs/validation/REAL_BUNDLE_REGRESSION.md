# Real Bundle Regression Tool

本文说明如何使用仓库内置的三路质量回归对照工具，快速判断 `streaming_lora_service` 的真实 bundle 是否出现：

- 短文本异常长音频
- `finish_reason` 异常（例如被长度上限截断）
- HTTP 真流式与 WebSocket Realtime 输出不一致

## 1. 工具入口

两种启动方式等价：

```text
qwen-tts-streaming-validate --bundle_dir <bundle_dir> --voice_registry_file <voice_registry.json> --output_dir <output_dir>
```

```text
python -m streaming_lora_service.quality_regression --bundle_dir <bundle_dir> --voice_registry_file <voice_registry.json> --output_dir <output_dir>
```

## 2. 推荐用法

```text
qwen-tts-streaming-validate --bundle_dir outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best_refcand8 --voice_registry_file streaming_lora_service/configs/voice_registry.example.json --output_dir streaming_lora_service/docs/validation/20260312_compare --local_files_only
```

默认会使用两条内置 case：

- `zh_formal`
- `ja_formal`

如果你想自定义 case，可传：

```text
--cases_json <path_to_cases.json>
```

JSON 格式示例：

```json
[
  {
    "id": "zh_short",
    "text": "你好，欢迎使用实时语音服务。",
    "language_type": "Chinese",
    "instructions": "正式，平静，清晰。",
    "voice": "yachiyo_formal"
  }
]
```

## 3. 输出内容

工具会在 `output_dir` 下写出：

- `metrics.json`
- `<case_id>/offline_non_streaming.wav`
- `<case_id>/http_non_streaming.wav`
- `<case_id>/http_streaming_runtime.wav`
- `<case_id>/websocket_realtime.wav`

其中 `metrics.json` 会记录：

- `duration_s`
- `total_audio_bytes`
- `elapsed_ms`
- 流式路径的 `ttfb_ms`
- `delta_chunks`
- `generated_steps`
- `emitted_chunks`
- `first_emitted_step`
- `finish_reason`
- `warnings`

## 4. 如何判断结果可疑

工具当前会自动标记以下典型异常：

1. `http_streaming_runtime` 或 `websocket_realtime` 的时长相对离线超过阈值
2. 流式路径 `finish_reason != eos`
3. `http_streaming_runtime` 与 `websocket_realtime` 的输出字节数不同

默认阈值：

```text
--max_stream_to_offline_ratio 1.5
```

默认还会使用：

```text
--seed 1234
```

这样同一条 case 在离线 / HTTP / WebSocket 三条路径上的比较会更稳定，尽量减少采样随机性带来的噪声。

如果你现在正在排查“短文本被拉到几十秒”，建议先不要把阈值放太宽；工具就是来负责吹哨的，不是来安慰数据的。

## 5. 当前建议排查顺序

1. 先看 `offline_non_streaming` 是否正常
2. 再看 `http_non_streaming` 是否与离线接近
3. 然后比较 `http_streaming_runtime` 与 `websocket_realtime`
4. 若流式路径显著更长，再重点看：
   - `finish_reason`
   - `generated_steps`
   - `delta_chunks`

## 6. 2026-03-12 当前观察到的结论

先前问题定位基于：

- `docs/validation/20260312_compare_seeded/metrics.json`

- `http_streaming_runtime` 与 `websocket_realtime` 在固定种子下已经对齐
- 但它们都比离线 / HTTP 非流式长约 `3.5x`
- 且 `generated_steps` 相对 HTTP 非流式 `codec_steps` 高约 `3.37x ~ 3.59x`
- 同时 `finish_reason` 仍然是 `eos`

因此当前更像是：

- **流式 step-level 生成链路本身生成了过多 codec step**

而不是：

- WebSocket 封装层把音频块弄坏
- Base64 `response.audio.delta` 拼接错误
- 单纯播放器把 24k PCM16 解错

### 6.1 修复后基线

根因修复后，新基线产物为：

- `docs/validation/20260312_compare_fix1/metrics.json`

当前结果显示：

- `http_streaming_runtime` 与 `websocket_realtime` 仍然对齐
- 且二者已回落到与 `http_non_streaming` 基本一致
- summary 为：
  - `warning_count = 0`
  - `warning_case_count = 0`

因此目前推荐把 `20260312_compare_fix1` 作为后续继续改 streaming 路径时的回归基线。

## 7. 当前已知限制

- 该工具用于**质量对照与回归**，不是公开 API 的一部分
- 当前仍不能替代人工试听
- 当前 runtime session 的 cross-append/commit continuation 问题仍未由该工具自动修复，只是更容易被定位出来