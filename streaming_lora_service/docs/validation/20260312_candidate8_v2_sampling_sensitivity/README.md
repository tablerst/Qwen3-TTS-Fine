# Candidate8 v2 Sampling Sensitivity

验证时间：2026-03-12

目标 bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

脚本：`streaming_lora_service/sampling_sensitivity.py`

产物：`metrics.json`

## 实验设置

同一 bundle、同一 speaker、同一 text、同一 instructions、同一 seed（`1234`），只改变采样相关参数：

- baseline：默认采样参数
- `rep1_0`：`repetition_penalty=1.0`
- `greedy`：`do_sample=false`
- `greedy_rep1_0`：`do_sample=false, repetition_penalty=1.0`

对每个 variant，同步比较：

- `http_non_streaming`
- `streaming_sampler_full_decode`

并记录：

- `codec_steps`
- `duration_s`
- `first_divergence_step`
- `shared_prefix_steps`

## 关键结论

### 中文 `zh_formal`

- baseline：第 `7` 步开始分叉，`106` vs `99` steps
- `rep1_0`：仍在第 `7` 步开始分叉，`95` vs `99` steps
- `greedy`：完全一致，`94` vs `94` steps
- `greedy_rep1_0`：完全一致，`104` vs `104` steps

解读：中文的早期分叉不是单独由 `repetition_penalty` 触发，而是**采样模式本身**在 official `talker.generate(...)` 与手写 step loop 之间仍存在差异。

### 日文 `ja_formal`

- baseline：第 `67` 步开始分叉
- `rep1_0`：第 `65` 步开始分叉
- `greedy`：完全一致
- `greedy_rep1_0`：完全一致

解读：日文同样呈现“采样时分叉、greedy 时对齐”的模式，只是分叉出现得更晚。

## 当前结论

这份实验把问题进一步收敛为：

- **采样模式下**：official `GenerationMixin` 采样循环 与 手写 `_sample_next_codec_token(...)` 之间存在 fidelity gap
- **greedy 模式下**：两条路径可以对齐

因此，后续如果要继续修 candidate8 v2 的在线听感问题，优先方向应是：

1. 对齐 official `generate(...)` 的 sampling loop 行为
2. 而不是继续优先怀疑：
   - LoRA 未加载
   - Speaker 映射错误
   - WebSocket 包装层
   - 增量解码本身
