# 基准说明

这个目录用于记录 `vllm-omni` 路线的最小基准结果。

## 首版目标

PoC 阶段只要求回答四个问题：

1. 服务是否能稳定启动？
2. 首包 / 首音频时间大概是多少？
3. 音频是否真的按 HTTP / WS 流式返回？
4. 与当前 `streaming_lora_service` 的体验相比，大致是更快还是更慢？

## 建议记录的指标

### HTTP

- 连接建立时间
- 响应头返回时间
- 首 chunk 时间
- 总字节数
- 总耗时

### WebSocket

- 连接时间
- `session.config` 到首个音频帧时间
- 句子数
- 音频 chunk 数
- 总音频字节数
- 总耗时

## 与现有仓库对齐的口径

可以参考：

- `streaming_lora_service/http_streaming_benchmark.py`
- `streaming_lora_service/ws_benchmark.py`

建议继续使用这些熟悉的口径：

- TTFT / TTFA（首 token / 首音频时间）
- RTF（实时因子）
- p50 / p95 / p99

## 当前建议

先做：

- 单请求串行基线；
- 固定文本长度；
- 固定 task_type；
- 固定 response_format=pcm；
- 固定单卡环境。

不要在 PoC 首轮就追求：

- 高并发；
- 多 speaker；
- 多 LoRA；
- 复杂路由。

## 输出建议

建议把原始探针结果 JSON 落到：

- `outputs/vllm_omni/`

再把结论摘要同步到这个目录的 markdown 文件，方便和已有 `benchmark/`、`outputs/` 结果横向比对。
