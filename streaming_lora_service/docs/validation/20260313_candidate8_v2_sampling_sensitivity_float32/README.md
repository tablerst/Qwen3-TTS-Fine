# Candidate8 v2 Sampling Sensitivity after float32 Fix

验证时间：2026-03-13

目标 bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

脚本：`streaming_lora_service/sampling_sensitivity.py`

本轮只跑 baseline sampling variant，并在运行前修复了在线 step sampler 的一个关键差异：

- 采样前统一将 `next_token_logits` 转成 `float32`

该行为与官方 `transformers.GenerationMixin._sample()` 对齐。

## 产物

- `zh_formal_baseline.json`
- `ja_formal_baseline.json`

## 中文结果

- 修复后：
  - `http_non_streaming_codec_steps = 106`
  - `streaming_sampler_codec_steps = 105`
  - `shared_prefix_steps = 83`
  - `first_divergence_step = 83`

对比修复前（2026-03-12 baseline）：

- 修复前：第 `7` 步开始分叉，`106` vs `99` steps
- 修复后：第 `83` 步才开始分叉，`106` vs `105` steps

说明：`float32` 采样精度对齐对中文 baseline 收益非常明显，已经把大部分早期漂移消掉。

## 日文结果

- 修复后：
  - `http_non_streaming_codec_steps = 83`
  - `streaming_sampler_codec_steps = 83`
  - `shared_prefix_steps = 67`
  - `first_divergence_step = 67`

与修复前基本一致。

说明：日文 baseline 的剩余差异并不主要来自 logits 精度，而更可能来自 sampling loop 其他细节。

## 当前结论

当前 candidate8 v2 的 sampling fidelity 问题已经拆成两层：

1. 已修复：采样前 `bfloat16` logits 直接 softmax / multinomial 带来的中文早期分叉。
2. 未完全解决：在 `float32` 对齐后仍存在的 sampling loop 差异，尤其体现在日文 baseline 与中文尾段。

## 建议下一步

优先继续比对：

- official `generate(...)` 的 logits processor / logits warper 顺序
- 我们手写 `_sample_next_codec_token(...)` 的 top-k / top-p / multinomial 细节
- 是否还存在额外的 sampling-related processor 未被复刻
