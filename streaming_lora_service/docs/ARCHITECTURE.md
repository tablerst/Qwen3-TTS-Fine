# Architecture

## 1. 总体架构

建议采用下面的分层：

```text
Client
  ↕ WebSocket
Streaming Server
  ├─ Session Manager
  ├─ Bundle Loader
  ├─ Prompt Builder
  ├─ Step Generator
  ├─ Incremental Decoder
  └─ Metrics / Logging
       ↓
Qwen3-TTS Base + LoRA Adapter + Speaker Patch
```

## 2. 组件职责

### 2.1 Bundle Loader

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

### 2.2 Session Manager

负责：

- 创建/销毁会话
- 限制最大并发会话数
- 为每个会话分配状态容器
- 响应 `cancel` / `stop`

### 2.3 Prompt Builder

负责把外部请求转换成模型 prefill 所需的输入结构：

- `input_ids`
- `instruct_ids`
- `speaker`
- `language`
- 初始 `talker_input_embeds`
- `trailing_text_hidden`

关键原则：

- 不重复发明 prompt 规则；
- 尽量复用 `Qwen3TTSForConditionalGeneration.generate(...)` 里已有的 prompt 组装逻辑。

### 2.4 Step Generator

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

- 新增一帧完整 codec group
- 更新后的 generation state
- 是否遇到 eos

### 2.5 Incremental Decoder

负责把“已生成 codec 序列”转成“尚未发给客户端的新增音频”。

设计要求：

- 维护 codec ring buffer
- 保留左上下文
- 避免每次从头全量 decode
- 最终输出 PCM16 little-endian

### 2.6 WebSocket Transport

职责：

- 接收控制信令 JSON
- 下发二进制音频 chunk
- 保持 session 生命周期
- 给出错误与完成状态

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
```

### 3.2 会话建立

```text
session.start
  -> 创建 Session State
  -> 记录 speaker / language / defaults
  -> 返回 session.ready
```

### 3.3 文本增量输入

```text
text.delta
  -> raw_text_buffer 追加
  -> tokenizer / commit policy 计算稳定前缀
  -> 新的 trailing text hidden
  -> step generator 推进若干步
  -> codec buffer 增长
  -> incremental decoder 产出 PCM chunk
  -> audio.chunk 下发
```

### 3.4 flush

```text
flush
  -> 强制提交未稳定文本尾部
  -> 继续 step 直到阶段性收敛或 eos
  -> 发送剩余音频块
```

### 3.5 stop / cancel

```text
stop
  -> 尽量输出已生成且尚未发送的音频
  -> session.completed

cancel
  -> 立即终止会话
  -> 丢弃未发音频
  -> session.cancelled
```

## 4. 模型状态设计建议

### 4.1 StreamingSession

建议定义类似对象：

- `session_id`
- `speaker_name`
- `language`
- `instruct`
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
- `cancelled`

### 4.2 为什么需要 `committed_text` 与 `pending_text_tail`

因为输入是增量文本，而 tokenizer 不是天然字符级稳定的。

推荐模式：

- `committed_text`：已确认喂入模型的稳定前缀
- `pending_text_tail`：尚未稳定提交的尾部

这样可减小重新 tokenize 导致的边界抖动。

## 5. LoRA 相关架构要求

### 5.1 模型层只加载一次 LoRA

不要每个会话重复：

- `load_lora_adapter(...)`
- `apply_speaker_patch(...)`

这两个动作应在服务启动时完成。

### 5.2 speaker patch 属于模型静态配置的一部分

当前 bundle 的 `speaker_embedding.safetensors` 和 `config_patch.json` 本质上是：

- 把 base 模型暂时改造成 custom voice 模型

因此流式服务不应在运行期频繁切换 speaker patch，除非未来引入多 bundle registry。

## 6. 并发模型建议

V1 建议：

- **单模型实例**
- **单 GPU 串行 step** 或受控轮询 step
- **多会话接入，但实际生成排队**

原因：

- 增量生成过程对 cache / GPU memory 很敏感；
- Windows + 单卡环境更适合先做稳态串行。

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
- 总音频 chunk 数
- 每 chunk 平均样本数
- flush 次数
- cancel 次数
- eos 结束 / 手动 stop 结束

## 9. 未来扩展点

### 9.1 Voice Design

相对容易扩展，只需增加 instruction prompt 管理。

### 9.2 Voice Clone

复杂度明显更高，因为需要会话中维护：

- `ref_audio`
- `ref_text`
- `ref_code`
- `ref_spk_embedding`

不建议在 V1 直接实现。

### 9.3 多 bundle / 多 speaker registry

后续可考虑：

- 启动时加载多个 bundle
- 通过 `voice_id` 路由到不同 session profile

这属于 V2+ 能力。
