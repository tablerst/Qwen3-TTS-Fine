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

## 已定位并修复的根因

根因已经在代码层面定位到：

- 文件：`streaming_lora_service/app/streaming_generator.py`
- 问题点：`_run_single_step()` 调用 `talker.forward(...)` 时，使用的是**扩展前**的 `attention_mask`

这会导致：

- 当前新采样 token 已经作为 `input_ids` 参与该步生成；
- 但 attention mask 仍停留在“上一时刻”的长度；
- 结果是 step-level 流式生成轨迹逐步偏离 `talker.generate(...)`

在真实模型上做前 20 个 step 的 fixed-seed 对照时，只要把 mask 改为“先扩一位再 forward”，手写流式 sampler 的主 codec token 序列就能重新对齐 `talker.generate(...)` 的前缀。

## 修复后验证结果

修复后的新产物：

- `docs/validation/20260312_compare_fix1/metrics.json`

关键结果：

1. `http_streaming_runtime` 与 `websocket_realtime` 继续保持一致
2. 它们不再比非流式长 `3.5x`
3. 反而已经与 `http_non_streaming` 对齐：
  - 中文：`7.76s / generated_steps=97`
  - 日文：`6.56s / generated_steps=82`
4. 报告 summary：
  - `warning_count = 0`
  - `warning_case_count = 0`

这意味着：

> 本次“短文本被拉长、重复词、严重过生成”的主根因已经得到修复。

## 关于 `offline_non_streaming` 听感仍略有差别的说明

这不是你耳朵在“找 bug”，而是当前代码路径里确实存在一个**有意保留的模式差异**：

- `offline_non_streaming`
  - 走的是 `qwen3tts.generate_custom_voice(..., non_streaming_mode=True)`
- 服务当前的 `http_non_streaming` / `http_streaming_runtime` / `websocket_realtime`
  - 统一走 `generate_custom_voice_step_aware(...)`
  - 当前服务基线更接近 `non_streaming_mode=False` 的 streaming-compatible prompt 语义

已在固定种子下验证：

- `offline wrapper == step-aware + non_streaming_mode=True`
- 但它**不等于**当前服务对齐后的 `http_non_streaming`

这能解释你在中文场景里听到的那一点点差异。

因此后续要先做一个产品决定：

1. **优先服务端三条路径完全一致**
   - 维持当前状态
   - HTTP 非流式 / HTTP 流式 / WebSocket Realtime 完全对齐
2. **优先服务 HTTP 非流式 与离线 SDK 一致**
   - 则需要把 HTTP 非流式改成 `non_streaming_mode=True`
   - 但这样它会重新与 streaming / realtime 出现模式差异

当前更推荐方案 1，因为它更适合对外服务的一致性。

## 当前服务候选主线选择

结合最新服务验证结果，当前更推荐把：

- `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

作为 **当前服务候选主线 bundle**。

依据：

- `docs/validation/20260312_candidate8_v2_service_validation/metrics.json`
- `warning_count = 0`
- 中日文默认回归 case 均通过
- 流式 / Realtime 路径稳定

剩余工作更偏向：

- 继续扩大真实 bundle 回归样本
- 补人工试听结论
- 后续再推进 cross-append/commit continuation

### 3. 跨 `append/commit` 的真正 continuation 仍是后续项

当前 runtime session 的状态复用仍偏向：

- 同一段文本生成中的恢复

而不是：

- 真实跨 `append/commit` 的连续续流

这与 `DELIVERY_CHECKLIST.md` 中的未完成项保持一致。

## 新增诊断方案：`streaming_sampler_full_decode`

为了把“流式采样轨迹偏移”和“增量解码拼接损耗”拆开，当前方案新增一个显式诊断路径：

- `offline_non_streaming`
  - 官方 `generate_custom_voice(..., non_streaming_mode=True)`
  - 用于观察离线整段模式的最佳参考口径
- `http_non_streaming`
  - 服务侧 `generate_custom_voice_step_aware(...)`
  - 用于观察服务 full-generate 但非流式下发的结果
  - 当前默认走 `non_streaming_mode=False`，与在线流式共享同一套 streaming-compatible prompt 语义
- `streaming_sampler_full_decode`
  - 与 `http_streaming_runtime` / `websocket_realtime` 共享同一套 step-level sampler
  - 但不采用增量音频拼接，而是在采样结束后把收集到的全部 codec step **一次性 full decode**
- `http_streaming_runtime`
  - 真实在线流式路径：step-level sampler + incremental decoder
- `websocket_realtime`
  - 真实在线 Realtime 路径：与 HTTP 流式共享 sampler 与增量解码链路

这条新增路径的用途非常直接：

1. 如果 `http_non_streaming` 与 `streaming_sampler_full_decode` 已经分叉，说明问题主要在 **step-level sampler 与 full generate 不一致**。
2. 如果 `streaming_sampler_full_decode` 与 `http_streaming_runtime` 分叉，说明问题主要在 **incremental decode / overlap 裁切 / 分块拼接**。
3. 如果 `http_streaming_runtime` 与 `websocket_realtime` 一致，而两者都偏离 `streaming_sampler_full_decode`，则基本可以把问题锁定在 **服务内部增量解码链路**，而不是 WS 包装层。

需要特别强调的是：

- `http_non_streaming`
- `streaming_sampler_full_decode`

这两条路径当前都基于 `non_streaming_mode=False` 的 prompt 语义。

因此如果这两条路径仍然分叉，优先应该怀疑：

- 官方 `talker.generate(...)` 的采样循环
- 我们手写 `StreamingCustomVoiceGenerator` 的 step loop / logits sampling 复刻

而不是先怀疑 prompt layout 本身。

因此，后续主观听感明显差异不应只再看“三路时长是否接近”，而应优先看：

- `http_non_streaming` vs `streaming_sampler_full_decode`
- `streaming_sampler_full_decode` vs `http_streaming_runtime`

这两组比较能分别回答“采样是否分叉”和“解码是否损声”。

## 新增 codec 级分叉诊断

当前 `metrics.json` 还会额外输出 `codec_diagnostics`，至少包含以下三组：

- `http_non_vs_streaming_sampler_full_decode`
- `streaming_sampler_full_decode_vs_http_streaming_runtime`
- `http_streaming_runtime_vs_websocket_realtime`

每组包含：

- `reference_codec_steps`
- `candidate_codec_steps`
- `shared_prefix_steps`
- `first_divergence_step`
- `reference_step_tokens`
- `candidate_step_tokens`
- `identical`

解释方式：

1. 如果 `shared_prefix_steps` 很短，说明两条路径在 very early stage 就开始采样分叉。
2. 如果 `shared_prefix_steps` 接近总 steps 且 `identical=false`，说明只是尾段收束差异。
3. 如果 `streaming_sampler_full_decode_vs_http_streaming_runtime.identical=true`，则可以把 codec 采样层面的责任从增量解码层剥离出去。

这比单看音频相关性更硬：

- 音频相关性告诉我们“听起来差了多少”；
- codec 分叉诊断告诉我们“从第几步开始走岔了”。

## 新增实证：采样敏感性对照

当前还补跑了一轮采样敏感性实验，产物位于：

- `docs/validation/20260312_candidate8_v2_sampling_sensitivity/metrics.json`

使用同一 bundle、同一 speaker、同一 text / instructions / seed，只改变：

- `do_sample`
- `repetition_penalty`

关键结果：

### 中文 `zh_formal`

- baseline（采样 + `repetition_penalty=1.05`）
  - `first_divergence_step = 7`
  - `106` vs `99` codec steps
- 去掉 repetition penalty（仍采样）
  - `first_divergence_step = 7`
  - `95` vs `99` codec steps
- greedy（`do_sample=false`）
  - `identical = true`
  - `94` vs `94` codec steps
- greedy + `repetition_penalty=1.0`
  - `identical = true`
  - `104` vs `104` codec steps

### 日文 `ja_formal`

- baseline（采样 + `repetition_penalty=1.05`）
  - `first_divergence_step = 67`
- 去掉 repetition penalty（仍采样）
  - `first_divergence_step = 65`
- greedy（`do_sample=false`）
  - `identical = true`
- greedy + `repetition_penalty=1.0`
  - `identical = true`

这组结果非常重要，因为它说明：

1. **只要关闭采样（greedy），official `talker.generate(...)` 与手写 step loop 就能重新对齐。**
2. 中文 baseline 第 7 步分叉并不是由 `repetition_penalty` 单独触发，因为把 penalty 改成 `1.0` 后仍然在第 7 步分叉。
3. 因此当前 candidate8 v2 的主矛盾更像是：
   - **采样模式下** official `GenerationMixin` 采样循环
   - 与手写 `_sample_next_codec_token(...)` 之间
   - 仍存在行为差异。

换句话说：

> 这轮实验把嫌疑进一步收敛到了 **sampling loop fidelity**，而不再只是泛泛地说“在线和离线生成方式不同”。

## 2026-03-13 追加修复：采样前统一转 `float32`

进一步对照官方 `transformers.GenerationMixin._sample()` 后发现：

- 官方在采样前会执行 `outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, ...)`
- 原先 `StreamingCustomVoiceGenerator` 直接在模型原始 dtype（典型是 `bfloat16`）上执行 softmax / multinomial

当前已将流式 step sampler 改为：

- prefill 后的 `next_logits` 统一转 `float32`
- 每个 step 后缓存的 `next_logits` 统一转 `float32`
- `_sample_next_codec_token(...)` 内部在采样前也再次显式使用 `float32`

并已补充单测：

- `test_sampling_logits_are_upcast_to_float32`

### 修复效果（candidate8 v2）

新产物：

- `docs/validation/20260313_candidate8_v2_path_diagnostic_float32/metrics.json`
- `docs/validation/20260313_candidate8_v2_sampling_sensitivity_float32/zh_formal_baseline.json`
- `docs/validation/20260313_candidate8_v2_sampling_sensitivity_float32/ja_formal_baseline.json`

#### 中文 `zh_formal`

修复前：

- `first_divergence_step = 7`
- `106` vs `99` codec steps
- `http_non_streaming` vs `streaming_sampler_full_decode` 波形相关性约 `0.011`

修复后：

- `first_divergence_step = 83`
- `106` vs `105` codec steps
- `http_non_streaming = 8.48s`
- `streaming_sampler_full_decode = 8.40s`
- `http_non_streaming` vs `streaming_sampler_full_decode` 波形相关性提升到约 `0.801`

这说明 `float32` 采样精度差异确实是中文主观差异中的一个重要来源，而且修复收益非常显著。

#### 日文 `ja_formal`

修复前后基本一致：

- `first_divergence_step = 67`
- `83` vs `83` codec steps

说明日文 baseline 的剩余差异并不主要由 logits 精度造成，后续仍需继续检查 sampling loop 的其余细节。

### 当前结论更新

截至目前，可以把剩余问题拆成两层：

1. **已修复的一层**：采样前未对齐官方 `float32` logits 处理，中文收益明显。
2. **仍待继续排查的一层**：在 `float32` 已对齐后，日文 baseline 与中文尾段仍存在 sampling-loop fidelity gap，说明后续还要继续比对 official `generate(...)` 的 processor / warper / sampling 细节。

## 2026-03-13 追加修复：重复 token 被重复施加 repetition penalty

继续比对 HuggingFace 的 `RepetitionPenaltyLogitsProcessor` 后发现：

- 官方实现对“历史里出现过的 token”是**每个 token 至多施加一次 penalty**；
- 原先 `StreamingCustomVoiceGenerator._apply_repetition_penalty()` 会按 `sampled_tokens` 顺序逐个处理；
- 当同一个 token 在历史中出现多次时，会被**重复惩罚多次**，这会在 sampling 模式的中后段持续放大偏移。

当前已修复为：

- 只对 `sampled_tokens` 的去重集合施加 penalty；
- 并已补充单测：
  - `test_repetition_penalty_is_applied_once_per_unique_token`

## 修复后的关键结果

### baseline sampling sensitivity 已完全对齐

新产物：

- `docs/validation/20260313_candidate8_v2_sampling_sensitivity_float32_repfix/zh_formal_baseline.json`
- `docs/validation/20260313_candidate8_v2_sampling_sensitivity_float32_repfix/ja_formal_baseline.json`

结果：

- 中文 `zh_formal`
  - `106` vs `106` codec steps
  - `shared_prefix_steps = 106`
  - `identical = true`
- 日文 `ja_formal`
  - `83` vs `83` codec steps
  - `shared_prefix_steps = 83`
  - `identical = true`

这说明在默认 sampling 参数下：

> `http_non_streaming` 与 `streaming_sampler_full_decode` 的 codec 轨迹已经重新完全对齐。

### 五路服务诊断也已完全收敛

新产物：

- `docs/validation/20260313_candidate8_v2_path_diagnostic_samplingfix/metrics.json`

关键结果：

- 中文：
  - `http_non_streaming = 8.48s / 106 steps`
  - `streaming_sampler_full_decode = 8.48s / 106 steps`
  - `http_streaming_runtime = 8.48s / 106 steps`
  - `websocket_realtime = 8.48s / 106 steps`
- 日文：
  - `http_non_streaming = 6.64s / 83 steps`
  - `streaming_sampler_full_decode = 6.64s / 83 steps`
  - `http_streaming_runtime = 6.64s / 83 steps`
  - `websocket_realtime = 6.64s / 83 steps`

所有 codec 对比均为：

- `identical = true`

即：

- `http_non_streaming`
- `streaming_sampler_full_decode`
- `http_streaming_runtime`
- `websocket_realtime`

在 candidate8 v2 默认回归 case 上已经完成对齐。

## 当前最终判断（截至 2026-03-13）

candidate8 v2 本轮主要修复了两个 sampling fidelity 问题：

1. **采样前未统一转 `float32`**
2. **重复 token 被重复施加 repetition penalty**

两者修复后，服务路径已经与服务非流式路径重新对齐。

当前仍保留的差异主要只剩：

- `offline_non_streaming`
  - 走 `generate_custom_voice(..., non_streaming_mode=True)`
- 服务路径
  - 走 `non_streaming_mode=False` 的服务语义

因此，当前已不再优先怀疑服务 streaming fidelity 本身；剩余 offline/online 差异主要属于**模式定义差异**，而不是服务在线路径继续“偷偷跑偏”。

## 建议下一步

1. 用同一 bundle、同一文本做离线 / HTTP 非流式 / `streaming_sampler_full_decode` / HTTP 流式 / WebSocket 五路对比
2. 优先检查 `http_non_streaming` 与 `streaming_sampler_full_decode` 的 codec step 数、时长与主观听感是否一致
3. 再检查 `streaming_sampler_full_decode` 与 `http_streaming_runtime` 的音频差异，确认增量解码损耗占比
4. 把“短文本最长时长阈值”做成可选的真实 bundle 回归测试
5. 扩大真实 bundle case 覆盖到更多语言/更长文本
6. 在继续推进 cross-append/commit continuation 前，保持当前 seeded 五路对照作为回归基线