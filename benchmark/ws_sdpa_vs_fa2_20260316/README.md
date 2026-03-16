# ws_sdpa_vs_fa2_20260316

- 日期：2026-03-16
- 目标：对比 `streaming_lora_service.ws_benchmark` 在 `sdpa` 与 `flash_attention_2` 两种 attention backend 下的端到端 WebSocket 服务表现
- bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- voice registry：`streaming_lora_service/configs/voice_registry.candidate8_v2.json`
- voice：`yachiyo_candidate8_v2`
- model alias：`qwen3-tts-flash-realtime`
- case：`zh_formal`、`ja_formal`
- warmup：`1`
- measured iterations：`2`
- 采样率：`24000`
- 响应格式：`pcm`
- 运行环境：Linux x86_64 (WSL2), Python `3.12.13`, torch `2.9.1+cu128`, torchaudio `2.9.1+cu128`
- FlashAttention wheel：`flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`

## Summary

本轮 **live WebSocket benchmark** 中，`sdpa` 明显快于 `flash_attention_2`。

### overall

| 指标 | sdpa | flash_attention_2 | 差值 |
|---|---:|---:|---:|
| TTFT p50 | `283.2112 ms` | `326.3585 ms` | `+43.1473 ms` (`+15.24%`) |
| TTFT p95 | `287.6646 ms` | `333.3233 ms` | `+45.6587 ms` (`+15.87%`) |
| total_elapsed p50 | `21724.6961 ms` | `26004.6151 ms` | `+4279.9190 ms` (`+19.70%`) |
| total_elapsed mean | `21954.0911 ms` | `25804.0891 ms` | `+3849.9980 ms` (`+17.54%`) |
| RTF mean | `2.4489` | `2.8869` | `+0.4380` (`+17.89%`) |
| audio_duration mean | `8.96 s` | `8.94 s` | `-0.02 s` (`-0.22%`) |

结论：在当前服务链路里，`flash_attention_2` 没有带来端到端收益，反而使首包和总耗时都变慢了约 `15% ~ 20%`。

## 分 case 结果

### zh_formal

| 指标 | sdpa | flash_attention_2 | 差值 |
|---|---:|---:|---:|
| TTFT p50 | `286.4875 ms` | `321.1398 ms` | `+12.10%` |
| TTFT p95 | `288.0009 ms` | `326.2712 ms` | `+13.29%` |
| total_elapsed p50 | `23007.0704 ms` | `25518.5471 ms` | `+10.92%` |
| total_elapsed mean | `23007.0704 ms` | `25518.5471 ms` | `+10.92%` |
| RTF mean | `2.4897` | `2.8226` | `+13.37%` |
| audio_duration mean | `9.24 s` | `9.04 s` | `-2.16%` |

### ja_formal

| 指标 | sdpa | flash_attention_2 | 差值 |
|---|---:|---:|---:|
| TTFT p50 | `276.7715 ms` | `330.1714 ms` | `+19.29%` |
| TTFT p95 | `281.1318 ms` | `334.0376 ms` | `+18.82%` |
| total_elapsed p50 | `20901.1119 ms` | `26089.6311 ms` | `+24.82%` |
| total_elapsed mean | `20901.1119 ms` | `26089.6311 ms` | `+24.82%` |
| RTF mean | `2.4080` | `2.9512` | `+22.56%` |
| audio_duration mean | `8.68 s` | `8.84 s` | `+1.84%` |

## 关键观察

### 1. `flash_attention_2` 确实生效在主模型上

`flash_attention_2` 服务启动日志中有明确快照：

- `requested_attn_implementation = "flash_attention_2"`
- `model_config_attn_implementation = "flash_attention_2"`

因此，这次结果**不是**“FA2 没吃进去”。

### 2. 服务链路不是全链路 FA2

同一份 attention backend snapshot 还显示：

- `speech_tokenizer_attn_implementation = "sdpa"`

说明当前 WebSocket benchmark 测到的是**整条服务链路**：

- WebSocket 协议交互
- session/update/commit
- 主模型生成
- speech tokenizer / decode
- streaming chunking

而不是单独测主模型 attention kernel 的 micro-benchmark。

因此，这里的结论应理解为：

> 在当前端到端实时服务场景里，`sdpa` 比 `flash_attention_2` 更适合作为默认 backend。

### 3. 小 batch / 实时场景下，FA2 不一定占优

当前场景是实时 TTS 服务，不是大 batch 长序列训练。首包和总耗时很可能更受以下因素影响：

- 小步长 streaming runtime 的调度开销
- decode / tokenizer 占比
- chunk 输出路径
- WebSocket 往返与服务编排

所以即使主模型已经切到 `flash_attention_2`，整体 TTFT/RTF 仍可能不如 `sdpa`。

## 运行情况与异常

### 1. `sdpa` 测试

- 服务端口：`9011`
- benchmark 成功：`4 / 4`

### 2. `flash_attention_2` 首次尝试端口冲突

首次尝试使用端口：`9012`

结果：启动日志显示端口已被占用：

- `ERROR: [Errno 98] ... address ('127.0.0.1', 9012): address already in use`

因此该次 benchmark 全部失败；对应日志仅作为异常记录保留，不纳入结果。

### 3. `flash_attention_2` 重新在 `19012` 端口完成正式测试

- 服务端口：`19012`
- benchmark 成功：`4 / 4`
- 本文中的 FA2 结果全部来自 `19012` 这一轮正式运行

## 当前建议

如果目标是当前这套 `candidate8 v2` WebSocket 服务的**真实联机表现**，建议：

- 默认保持 `attn_implementation = "sdpa"`
- 暂不将 `flash_attention_2` 作为服务默认 backend

如果后续还要继续排查，可以优先做：

1. 增加 iteration（如 `5` 或 `10`）重跑，确认趋势是否稳定
2. 做更细的 profiling，把主模型生成、tokenizer decode、streaming chunking 分开计时
3. 进一步确认是否存在 `torch_dtype` / `dtype` 传参差异对 FA2 路径的影响

## 产物

### benchmark 结果

- `ws_benchmark_sdpa.json`
- `ws_benchmark_fa2.json`

### 服务日志

- `server_sdpa.log`
- `server_fa2.log`（端口 `9012` 冲突失败记录）
- `server_fa2_19012.log`（正式 FA2 成功运行日志）

## 当前结论

本轮记录可以作为一个明确的仓库内基线：

> 对 `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best` 这条实时服务主线，在当前 Linux + torch 2.9.1 + FlashAttention 2 预编译 wheel 环境下，`ws_benchmark` 端到端表现中 `sdpa` 优于 `flash_attention_2`。
