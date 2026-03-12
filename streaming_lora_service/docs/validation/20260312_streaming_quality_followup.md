# Streaming Quality Follow-up (2026-03-12)

本文记录 2026-03-12 针对 `streaming_lora_service` 的一轮质量补强结论，目的是把“已经落地的防呆措施”和“仍待修复的真实音质问题”分开。

## 本次已落地改进

### 1. 收紧 WebSocket 音频契约

- `session.update` 当前只接受：
  - `response_format = "pcm"`
  - `sample_rate = 24000`
- 若客户端传入不支持的值，服务端现在会直接返回 `error`
- 这样可以避免客户端看到 `session.updated` 就误以为自己成功切到 WAV / 16kHz，而实际下行仍然是 24kHz PCM16

### 2. 增加流式结束原因诊断

- `StreamingCustomVoiceGenerator` 现已记录 `finish_reason`
- 当前取值：
  - `eos`
  - `length`
- 该指标已同步进入 runtime session 的 generation metrics，便于后续定位“是正常命中 EOS，还是被长度上限截断”

### 3. 补齐第一批关键回归测试

已新增/增强的覆盖点：

- WebSocket Realtime 拒绝不支持的 `response_format`
- WebSocket Realtime 拒绝不支持的 `sample_rate`
- HTTP `stream=true` 中间 PCM 片段与最终 `audio.url` WAV 的内容一致性
- 流式生成 `finish_reason = eos`
- 流式生成 `finish_reason = length`

### 4. 新增三路质量回归对照工具

当前仓库已新增：

- `python -m streaming_lora_service.quality_regression`
- 控制台别名：`qwen-tts-streaming-validate`

用途：

- 对同一条 case 同时跑离线、HTTP 非流式、HTTP 真流式 runtime、WebSocket Realtime
- 输出每条路径的字节数、时长、耗时、TTFB、delta chunk 数、`generated_steps`、`finish_reason`
- 自动标记明显可疑情况，例如：
  - 流式路径时长显著高于离线路径
  - `finish_reason != eos`
  - HTTP 真流式与 WebSocket Realtime 输出大小不一致

建议命令：

```text
qwen-tts-streaming-validate --bundle_dir outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best_refcand8 --voice_registry_file streaming_lora_service/configs/voice_registry.example.json --output_dir streaming_lora_service/docs/validation/20260312_compare
```

## 当前仍未解决的问题

### 1. 真实 bundle 短文本仍出现异常长音频证据

现有验证产物 `docs/validation/20260311_real_bundle/metrics.json` 与对应 WAV 仍显示：

- `sample_01_zh_formal.wav` 约 `49.2s`
- `sample_02_ja_formal.wav` 约 `40.96s`

对于这两条短文本来说，这已经明显超出正常预期。

### 2. 当前改动没有宣称“音质问题已修复”

本次落地的内容主要是：

- 防止客户端把 WebSocket 音频契约用错；
- 让诊断信号更清晰；
- 让回归测试先守住明显边界；

它**并不意味着**真实 bundle 的“重复词 / 过生成 / 严重拉长”已经解决。

## 新增实证：固定种子后的三路对照结果

本次已用新工具实际跑出一份固定种子报告：

- `docs/validation/20260312_compare_seeded/metrics.json`

关键观察：

1. `offline_non_streaming` 与 `http_non_streaming` 时长接近
  - 中文约 `8.0s` vs `7.76s`
  - 日文约 `6.24s` vs `6.56s`
2. `http_streaming_runtime` 与 `websocket_realtime` 在固定种子下完全对齐
  - 中文均为 `27.84s`
  - 日文均为 `22.08s`
  - `generated_steps` 也一致
3. 两条流式路径都显著长于离线/HTTP 非流式
  - 中文时长比约 `3.48x`
  - 日文时长比约 `3.54x`
4. 两条流式路径虽然最终 `finish_reason = eos`，但 `generated_steps` 仍远高于 HTTP 非流式 `codec_steps`
  - 中文约 `3.59x`
  - 日文约 `3.37x`

这说明一个非常关键的结论：

> 当前异常已经基本定位到 **step-level 流式生成链路本身**，而不是 WebSocket 包装层、Base64 delta 协议层，或单纯的播放器解码错误。

更具体地说：

- 问题发生在“生成出了过多 codec step”这一步；
- 不是增量解码后才把正常音频重复裁出来；
- 也不是 WS 事件包装把音频块拼坏了。

### 3. 跨 `append/commit` 的真正 continuation 仍是后续项

当前 runtime session 的状态复用仍偏向：

- 同一段文本生成中的恢复

而不是：

- 真实跨 `append/commit` 的连续续流

这与 `DELIVERY_CHECKLIST.md` 中的未完成项保持一致。

## 建议下一步

1. 用同一 bundle、同一文本做离线 / HTTP 非流式 / WebSocket 拼接音频三路对比
2. 继续对照 `StreamingCustomVoiceGenerator` 与 `model.generate(...)` 的 prompt/prefill/step 推进差异
3. 重点检查为什么流式路径在 `finish_reason = eos` 的前提下仍会多生成约 `3.3x ~ 3.6x` codec step
4. 把“短文本最长时长阈值”做成可选的真实 bundle 回归测试
5. 在继续推进 cross-append/commit continuation 前，先把当前真实流式 stop 行为定位清楚