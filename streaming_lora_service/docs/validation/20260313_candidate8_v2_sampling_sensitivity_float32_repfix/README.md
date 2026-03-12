# Candidate8 v2 Sampling Sensitivity after float32 + repetition-penalty Fixes

验证时间：2026-03-13

目标 bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

产物：

- `zh_formal_baseline.json`
- `ja_formal_baseline.json`

## 本轮修复背景

在 `float32` 采样对齐之后，中文 baseline 已经显著改善，但日文仍存在 baseline sampling 分叉。

继续排查后发现：

- 官方 `RepetitionPenaltyLogitsProcessor` 对重复 token 是**每个 token 最多惩罚一次**；
- 原先 `StreamingCustomVoiceGenerator` 会对 `sampled_tokens` 中重复出现的 token 反复施加 penalty。

修复后重新跑 baseline sampling，结果如下。

## 中文 `zh_formal`

- `http_non_streaming_codec_steps = 106`
- `streaming_sampler_codec_steps = 106`
- `shared_prefix_steps = 106`
- `first_divergence_step = null`
- `identical = true`

## 日文 `ja_formal`

- `http_non_streaming_codec_steps = 83`
- `streaming_sampler_codec_steps = 83`
- `shared_prefix_steps = 83`
- `first_divergence_step = null`
- `identical = true`

## 当前结论

在默认 baseline sampling 参数下，candidate8 v2 现在已经实现：

- official `talker.generate(...)`
- hand-written `StreamingCustomVoiceGenerator`

在 codec 序列上的完全对齐。

因此本轮已经把此前定位到的 sampling fidelity 问题收口为两个具体修复：

1. 采样前 `float32` 对齐
2. `repetition_penalty` 对重复 token 只施加一次

修复后，baseline sampling 不再发生 codec 分叉。
