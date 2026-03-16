# vLLM-Omni Qwen3-TTS 落地目录

这个目录用于在当前仓库里落地一条 **`vllm-omni` 在线服务路线**，目标是：

- 复用官方 `Qwen3-TTS` 在线服务能力；
- 首版聚焦 **单实例 + 单 LoRA**；
- 对外直接使用官方 `OpenAI/Speech API` 与官方 WebSocket 协议；
- 把当前仓库已有的 LoRA bundle、speaker patch、benchmark 经验接进来；
- 在验证完兼容性之前，不重写一套新的推理服务。

> 当前定位：**PoC / 兼容性验证骨架**。
> 这个目录已经把文档、脚本、探针和配置位点搭出来，但 **`Qwen3-TTS + LoRA` 能否被 `vllm-omni` 原生在线挂载** 仍然需要验证。

## 当前结论

### 已确认

- 官方文档已经支持 `Qwen3-TTS` 在线服务。
- 支持 `/v1/audio/speech`。
- 支持 `stream=true` + `response_format="pcm"` 的 HTTP PCM 流式输出。
- 支持 `/v1/audio/speech/stream` WebSocket 增量文本输入。
- 流式依赖 stage config 中 `async_chunk: true`。

### 尚未确认

- `Qwen3-TTS` 在线服务场景下，`vllm-omni` 是否支持直接加载当前仓库导出的 **PEFT LoRA + speaker patch bundle**。
- 如果不支持，首版 fallback 是：
  1. 先把 LoRA 整理/导出成 `vllm-omni` 能直接消费的本地模型目录；或
  2. 直接合并回基模后再由 `vllm-omni` 启动。

### 2026-03-16 本机实测结果

在当前机器上（WSL2 + Mirror 网络模式 + 本机启用 TUN / 代理）已经确认：

- `vllm-omni` 服务可以成功启动；
- API server 启动日志已经落盘；
- 通过 **非 loopback 地址** 可以访问服务：
  - `198.18.0.1:8091`
  - `192.168.31.180:8091`
- `127.0.0.1:8091` 在当前环境下**不可作为可靠探测地址**；
- `GET /v1/audio/voices` 可成功返回 `200 OK`；
- `POST /v1/audio/speech`（Base + 本地参考音频 + `x_vector_only_mode=true`）可成功返回音频；
- `POST /v1/audio/speech` 的 **HTTP PCM 流式** 已成功验证；
- `WS /v1/audio/speech/stream` 在 **句级返回** 模式下可用；
- `WS stream_audio=true` 当前失败，报错指向上游服务实现缺少 `_generate_pcm_chunks`，应视为上游缺口/bug，而不是本仓库脚本问题。

## 目录结构

```text
vllm_omni_service/
├── README.md
├── benchmarks/
│   └── README.md
├── configs/
│   └── README.md
├── docs/
│   └── DECISIONS.md
└── scripts/
    ├── http_stream_probe.py
    ├── prepare_artifact.py
    ├── serve.sh
    ├── sync_stage_config.py
    └── ws_stream_probe.py
```

## 建议工作流

### 1. 安装 `vllm-omni`

当前仓库虚拟环境里尚未检测到 `vllm` / `vllm_omni`。安装完成后，再继续下面两步：

- 同步官方 `qwen3_tts.yaml` 到本地 `configs/`；
- 对 LoRA bundle 做兼容性实验。

### 2. 同步官方 stage config

使用 `scripts/sync_stage_config.py`：

- 自动定位已安装的 `vllm_omni` 包；
- 复制官方 `qwen3_tts.yaml` 到本地；
- 生成 `configs/qwen3_tts.stage.yaml` 作为后续实验基线。

### 3. 规范化当前仓库的 LoRA bundle

使用 `scripts/prepare_artifact.py`：

- 读取现有 `manifest.json`、`adapter/`、`config_patch.json`、`speaker_embedding.safetensors`；
- 解析出 base model、speaker 名称、speaker id；
- 输出一个规范化实验目录和 `service_artifact.json`；
- 明确告诉后续服务启动需要什么、当前还缺什么。

> 这个脚本 **不会假装已经完成 LoRA merge**。
> 它的职责是把现有 bundle 整理成“服务验证输入”。

### 4. 启动服务

使用 `scripts/serve.sh`：

- 默认支持官方 HF 模型名（`CustomVoice` / `VoiceDesign` / `Base`）；
- 后续也支持把 `--model` 指到本地模型目录；
- 优先使用本地同步后的 stage config；
- 如果本地 config 不存在，则 fallback 到 `vllm_omni` 安装目录里的官方 config。
- 默认会把服务日志落到 `vllm_omni_service/logs/`，分别保存：
  - `*.stdout.log`
  - `*.stderr.log`
  - `*.meta.log`

示例：

- 前台启动并自动落日志：
  - `scripts/serve.sh --task-type Base`
- 指定日志目录：
  - `scripts/serve.sh --task-type Base --log-dir /path/to/logs`
- 只看命令、不真正启动：
  - `scripts/serve.sh --task-type Base --dry-run`

### 5. 验证流式

- `scripts/http_stream_probe.py`：验证 REST PCM 流式；
- `scripts/ws_stream_probe.py`：验证 WebSocket 增量文本输入和音频输出；
- 基准和结果记录建议统一放到 `outputs/vllm_omni/` 或 `vllm_omni_service/benchmarks/`。

当前实测建议：

- HTTP：优先验证 `198.18.0.1:8091`；
- WebSocket：先验证默认句级模式，再单独验证 `stream_audio=true`；
- Base 任务尽量优先用**本地参考音频**，由探针脚本自动编码为 base64 data URL，避免远程 `ref_audio` 抓取出现 `403`。

## 推荐首版边界

### 包含

- 单实例固定 1 个 LoRA；
- 直接复用官方 API；
- LoRA 兼容性验证；
- HTTP / WS 流式探针；
- 最小可用 PoC 文档。

### 不包含

- 请求级动态切 LoRA；
- 同一实例多 LoRA 路由；
- 自定义 FastAPI 包装层；
- 生产级鉴权、容器化、监控告警；
- 立即把此目录接入根 `pyproject.toml` 的发布/CLI 体系。

## 已知限制

1. `stream=true` 只能和 `response_format="pcm"` 一起用。
2. WebSocket 的“全流式”是：**文本增量输入 + 音频按句/按 chunk 增量输出**，不是无限细粒度地逐 token 出音。
3. 当前仓库里 LoRA bundle 的事实来源仍然是：
   - `lora_finetuning/export_custom_voice.py`
   - `lora_finetuning/common.py`
   - `streaming_lora_service/app/bundle_loader.py`
4. 根目录 `README.md` 里关于 vLLM 的说明已落后于官方最新在线服务文档，需要后续统一更新。
5. 当前机器上的 `HTTP_PROXY` / `HTTPS_PROXY` 会影响探针结果；`vllm_omni_service` 下的探针脚本现在默认**不继承环境代理**。
6. 本机实测中，远程 `ref_audio`（如 OSS URL）可能被服务端抓取时返回 `403`；优先改用本地文件或 base64 data URL。

## WSL Mirror / TUN / 代理注意事项

你当前环境补充了两条非常关键的现实条件：

- WSL 使用 **Mirror 网络模式**；
- 本机存在 **TUN / 代理模式**。

这会让“端口似乎在监听”和“客户端真的能连到本地服务”之间出现偏差，尤其是当环境里还设置了：

- `HTTP_PROXY`
- `HTTPS_PROXY`
- `http_proxy`
- `https_proxy`

建议遵循下面的调试习惯：

1. **优先让服务绑定 `0.0.0.0`**，不要只绑定 `127.0.0.1`。
2. 对本地探针显式绕过代理：
  - `curl --noproxy '*' http://127.0.0.1:8091/...`
  - 在当前环境里，更推荐直接探 `http://198.18.0.1:8091/...` 或 `http://192.168.31.180:8091/...`
3. 如果前台看起来启动了，但请求拿到 `502`，先怀疑是**代理链路返回**，不要第一时间把锅甩给模型本身。
4. 优先看 `logs/*.stderr.log` 和 `logs/*.meta.log`，再判断是：
  - 模型阶段未 ready；
  - 端口绑定问题；
  - 代理干扰；
  - WSL / 镜像网络导致的访问路径差异。

## 下一步建议

优先顺序如下：

1. 安装并确认 `vllm-omni`；
2. 同步 stage config；
3. 用一个现成 bundle 生成 `service_artifact.json`；
4. 验证“官方模型在线服务”先能正常启动；
5. 再验证“本地 LoRA 产物”能否挂到 `vllm-omni` 上；
6. 如果失败，明确切换到 merged-model 路线。

## 当前验证摘要

- HTTP 非流式 Base 请求（本地 ref_audio, `x_vector_only_mode=true`）
  - 成功
  - 产出 wav 大小：`222,764` 字节
  - 首包时间约：`80.3s`
  - 音频时长约：`4.64s`
  - `RTF≈17.31`

- HTTP PCM 流式 Base 请求（本地 ref_audio, `x_vector_only_mode=true`）
  - 成功
  - `chunk_count=14`
  - `total_bytes=188,160`
  - `first_chunk_ms≈4266.66`
  - `elapsed_ms≈21582.1`
  - 音频时长约：`3.92s`
  - `RTF≈5.51`

- WebSocket Base 请求（句级模式，`stream_audio=false`）
  - 成功
  - `sentences=2`
  - `audio_chunks=2`
  - `audio_bytes=230,400`
  - `first_audio_ms≈20814.1`
  - 音频时长约：`4.8s`
  - `RTF≈6.74`

- WebSocket Base 请求（chunk 模式，`stream_audio=true`）
  - 失败
  - 服务端异常：`'OmniOpenAIServingSpeech' object has no attribute '_generate_pcm_chunks'`
  - 初步判断：上游 `vllm-omni` 当前版本的 WS chunk 音频流式实现缺口
