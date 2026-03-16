# Architecture

## 1. 总体架构

建议采用下面的分层：

```text
Client / SDK
  ↕ Qwen-compatible payloads & events
Public API Layer
  ├─ Realtime WebSocket Adapter
  └─ HTTP TTS Adapter (shared contract)
       ↓
Service Core
  ├─ Voice Registry
  ├─ Bundle Loader
  ├─ Session Manager
  ├─ Prompt Builder
  ├─ Step Generator
  ├─ Incremental Decoder
  └─ Metrics / Logging
       ↓
Qwen3-TTS Base + LoRA Adapter + Speaker Patch
```

这张图的关键含义是：

- **公开协议层**负责“长得像官方”；
- **服务核心层**负责“内部跑得通、跑得稳”；
- **LoRA 装配**属于服务核心，而不是公开 API 心智。

## 2. 组件职责

### 2.1 Public API Layer

职责：

- 接收官方风格字段与事件；
- 校验输入是否符合公开契约；
- 把公开语义翻译为内部 runtime 调用；
- 把内部增量音频翻译为 `response.audio.delta` 等官方风格输出。

建议拆为：

- `Realtime WebSocket Adapter`
- `HTTP TTS Adapter`

其中 Realtime 为当前目录的核心优先级。

### 2.2 Voice Registry

负责把对外公开的：

- `model`
- `voice`

映射到服务内部的：

- 默认 bundle
- speaker profile
- 模型能力标签

关键原则：

- 对外 `voice` 不直接等于内部文件名；
- registry 是公开别名层；
- LoRA bundle 细节不向普通客户端暴露。

### 2.3 Bundle Loader

负责从 LoRA bundle 恢复一个可推理的 `custom_voice` 模型实例。

输入：

- `bundle_dir`

输出：

- 已注入 LoRA adapter 的 `Qwen3TTSModel`
- bundle 元信息（speaker_name、speaker_id、base_model_path）

复用现有函数：

- `load_lora_adapter(...)`
- `apply_config_patch(...)`
- `apply_speaker_patch(...)`

### 2.4 Session Manager

负责：

- 创建/销毁 session；
- 限制最大并发 session 数；
- 为每个 session 分配状态容器；
- 响应 `finish` 与内部中断逻辑；
- 维护 `server_commit` / `commit` 的行为差异。

### 2.5 Prompt Builder

负责把公开请求转换成模型 prefill 所需的输入结构：

- `input_ids`
- `instruct_ids`
- `speaker`
- `language`
- 初始 `talker_input_embeds`
- `trailing_text_hidden`

关键原则：

- 不重复发明 prompt 规则；
- 尽量复用 `Qwen3TTSForConditionalGeneration.generate(...)` 里已有逻辑；
- 公开 `model` / `voice` 的解释在 registry 层完成，而不是散落在 prompt builder 中。

### 2.6 Step Generator

负责逐步推进生成，而不是一次性 `generate()` 完整序列。

状态项建议包括：

- `past_key_values`
- `past_hidden`
- `generation_step`
- `trailing_text_hidden`
- `tts_pad_embed`
- `codec_steps`
- `finished`

每次 step 的目标产物：

- 新增一帧完整 codec group；
- 更新后的 generation state；
- 是否遇到 eos。

### 2.7 Incremental Decoder

负责把“已生成 codec 序列”转成“尚未发给客户端的新增音频”。

设计要求：

- 维护 codec ring buffer；
- 保留左上下文；
- 避免每次从头全量 decode；
- 输出原始音频字节；
- 由公开协议层编码成 Base64 `delta`。

### 2.8 Metrics / Logging

职责：

- 记录 TTFB；
- 记录总 step 数与 chunk 数；
- 区分 `server_commit` / `commit` 模式；
- 为后续压测与质量比较提供基础日志。

## 3. 关键数据流

### 3.1 服务启动期

```text
bundle_dir
  -> manifest.json
  -> base_model_path
  -> adapter/
  -> config_patch.json
  -> speaker_embedding.safetensors
  -> Bundle Loader
  -> Qwen3TTSModel (常驻)
  -> Voice Registry
  -> public model / voice aliases ready
```

关键点：

- LoRA 默认挂载发生在**服务启动期**；
- 普通第三方调用不需要知道这些细节。

### 3.2 会话建立

```text
session.update
  -> Public API Layer
  -> resolve public model / voice
  -> create Session State
  -> return session.created / session.updated
```

### 3.3 文本增量输入

```text
input_text_buffer.append
  -> raw_text_buffer 追加
  -> if server_commit: commit policy 计算稳定前缀
  -> Prompt Builder / Step Generator 推进若干步
  -> codec buffer 增长
  -> Incremental Decoder 产出新增音频字节
  -> Public API Layer 编码为 response.audio.delta
  -> 下发给客户端
```

### 3.4 commit

```text
input_text_buffer.commit
  -> 将当前缓冲文本提交给 runtime
  -> 返回 input_text_buffer.committed
  -> 触发 response.created
  -> 逐批下发 response.audio.delta
```

### 3.5 finish

```text
session.finish
  -> 尽量生成并发送剩余音频
  -> response.audio.done
  -> response.done
  -> session.finished
```

## 4. 模型状态设计建议

### 4.1 `StreamingSession` / `RuntimeSession`

建议定义类似对象：

- `session_id`
- `public_model_alias`
- `public_voice_alias`
- `speaker_name`
- `language_type`
- `instructions`
- `mode`
- `raw_text_buffer`
- `committed_text`
- `pending_text_tail`
- `past_key_values`
- `past_hidden`
- `generation_step`
- `trailing_text_hidden`
- `tts_pad_embed`
- `generated_codes`
- `decoded_until_step`
- `finished`

补充建议：

- `RuntimeSession` 应允许把“状态绑定频率”作为独立策略控制，而不是默认死绑到每个 generation step；
- 对当前服务实现，更推荐默认使用 **chunk 级同步**，仅在调试或验证跨步恢复时切回 `step`；
- 这样可以把 session 语义保留在公开协议之外，同时减少流式热路径里的 Python 复制税。

### 4.2 为什么同时需要公开别名与内部 speaker 信息

因为：

- 客户端认知的是 `voice="yachiyo_formal"`；
- 模型推理真正需要的是内部 `speaker_name` / profile；
- 二者之间需要 registry 进行稳定映射。

## 5. LoRA 相关架构要求

### 5.1 模型层只加载一次 LoRA

不要每个 session 重复：

- `load_lora_adapter(...)`
- `apply_speaker_patch(...)`

这两个动作应在服务启动时完成。

### 5.2 speaker patch 属于模型静态配置

当前 bundle 的 `speaker_embedding.safetensors` 与 `config_patch.json` 本质上是：

- 把 base 模型改造成当前服务默认的 custom voice runtime。

因此 V1 不应在公开协议中要求客户端理解这层配置。

### 5.3 voice routing 应优先在 registry 层做，而不是会话里临时拼装

这样可以确保：

- 对外 voice 命名稳定；
- 后续更换 bundle 时文档与客户端改动更小；
- speaker patch / profile 变更不会直接污染公开 API。

## 6. 并发模型建议

V1 建议：

- **单模型实例**
- **单 GPU 串行 step** 或受控轮询 step
- **多 session 接入，但实际生成排队**

原因：

- 增量生成过程对 cache / GPU memory 很敏感；
- Windows + 单卡环境更适合先做稳态串行；
- 第一阶段目标是把公开契约和流式质量先做稳，不是先卷并发数字。

## 7. 编码/解码粒度建议

### codec step 到音频时长

基于 12Hz tokenizer 当前配置：

- `decode_upsample_rate = 1920`
- `sample_rate = 24000`

则单个 step 时长约为：

$$
1920 / 24000 = 0.08s
$$

即约 80ms。

### 推荐默认发包粒度

建议默认每 4~6 steps 解码并发送一次：

- 4 steps ≈ 320ms
- 6 steps ≈ 480ms

这个粒度在“延迟”和“解码成本”之间更平衡。

## 8. 推荐监控指标

每个 session 建议记录：

- TTFB（首包时间）
- 总生成 step 数
- 总音频 delta 数
- 每 delta 平均字节数
- commit 次数
- `server_commit` / `commit` 模式占比
- eos 结束 / 手动 finish 结束
- `state_sync_calls`
- `total_state_sync_ms`

## 8.1 当前已落地的实现层提速策略

截至 2026-03-17，`streaming_lora_service` 已落地第一轮“非 attention 热路径”优化：

1. `RuntimeSession` 状态绑定默认从 step 级降到 chunk 级；
2. `attention_mask` 改为预分配 buffer + 按长度切片，避免每 step `torch.cat(...)` 增长式复制。

这两项都属于：

- 不改公开协议；
- 不改模型权重；
- 不改采样逻辑；
- 以减少实现层 CPU / 内存税为目标。

## 9. 未来扩展点

### 9.1 HTTP 兼容层

与当前目录共享：

- `voice_registry`
- `bundle_loader`
- 默认 LoRA 装配

这样可以避免 HTTP 和 Realtime 使用两套完全不同的 `voice` / `model` 语义。

### 9.2 Voice Design

相对容易扩展，只需增加 instruction prompt 管理与 profile 路由。

### 9.3 Voice Clone

复杂度明显更高，因为需要会话中维护：

- `ref_audio`
- `ref_text`
- `ref_code`
- `ref_spk_embedding`

不建议在 V1 直接实现。

### 9.4 Native/Internal Protocol

如果未来仍需要更低开销的二进制 PCM 通道：

- 可单独定义 internal/native 协议；
- 但不应取代当前 Qwen-compatible 公开协议。
