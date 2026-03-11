# Qwen Compatibility Matrix

## 1. 说明

本文用于明确：

- 哪些能力会与官方 Qwen3-TTS / Qwen-TTS-Realtime **强兼容**；
- 哪些能力是**部分兼容**；
- 哪些能力在 V1 **明确不支持**。

目标不是宣称“完全等同官方云服务”，而是让第三方调用方明确知道：

> 这是一套**官方风格兼容接口**，底层为本地 LoRA + 自有 runtime 实现。

## 2. 兼容矩阵

| 能力项 | 官方 Qwen3-TTS / Realtime | 本项目 V1 | 说明 |
| --- | --- | --- | --- |
| `model` 字段 | 支持 | **部分兼容** | 支持公开别名兼容，底层模型为本地部署版本 |
| `voice` 字段 | 支持 | **部分兼容** | 由本地 `voice_registry` 决定，不等于官方全量音色集合 |
| `language_type` | 支持 | **强兼容** | 公开语义保持一致 |
| `instructions` | 支持（指定模型） | **部分兼容** | 是否生效取决于当前公开 `model` 能力 |
| `optimize_instructions` | 支持（指定模型） | **部分兼容** | 可保留字段与语义，内部可按能力决定是否真正实现 |
| Realtime `session.update` | 支持 | **强兼容** | 作为公开主初始化事件 |
| Realtime `input_text_buffer.append` | 支持 | **强兼容** | 作为公开主增量输入事件 |
| Realtime `input_text_buffer.commit` | 支持 | **强兼容** | 作为公开主提交事件 |
| Realtime `input_text_buffer.clear` | 支持 | **强兼容** | 作为公开主清空事件 |
| Realtime `session.finish` | 支持 | **强兼容** | 作为公开主结束事件 |
| `server_commit` 模式 | 支持 | **强兼容** | 语义保持一致 |
| `commit` 模式 | 支持 | **强兼容** | 语义保持一致 |
| `response.created` | 支持 | **强兼容** | 公开事件名与语义对齐 |
| `response.audio.delta` | 支持 | **强兼容** | 公开主输出事件 |
| `response.audio.done` | 支持 | **强兼容** | 公开事件名与语义对齐 |
| `response.done` | 支持 | **强兼容** | 公开事件名与语义对齐 |
| `session.finished` | 支持 | **强兼容** | 公开事件名与语义对齐 |
| `error` 事件 | 支持 | **强兼容** | 公开错误模型保持官方风格 |
| Base64 音频 delta | 支持 | **强兼容** | 公开协议输出形态与官方保持一致 |
| 二进制 PCM 帧 | 非主公开形式 | **不作为公开主协议** | 如保留，仅作为 internal/native 扩展 |
| `bundle_dir` 客户端传入 | 不涉及 | **不支持** | 属于服务部署参数，不对第三方公开 |
| LoRA adapter 细节暴露 | 不涉及 | **不支持** | 属于服务内部实现 |
| 系统音色全量覆盖 | 支持 | **不支持** | V1 仅支持本地 registry 中已配置 voice |
| 官方云端专属音色注册 | 支持 | **不支持** | 本地服务使用 bundle / registry 机制替代 |
| `voice_clone` 真流式 | 官方特定模型支持 | **暂不支持** | V1 先聚焦 `custom_voice` |
| `voice_design` 真流式 | 官方特定模型支持 | **暂不支持** | V1 先聚焦 `custom_voice` |
| 完整 DashScope endpoint 路径兼容 | 官方支持 | **暂不承诺** | V1 优先保证 payload / event compatibility |

## 3. 关键解释

### 3.1 为什么 `voice` 只是部分兼容

因为官方 `voice` 背后通常对应：

- 系统音色集合；
- 云端专属音色资产；
- 官方维护的模型能力矩阵。

本项目的 `voice` 背后对应的是：

- 本地 LoRA bundle；
- speaker patch；
- 服务内部 registry。

所以我们兼容的是**字段形态与调用心智**，不是官方全量资源池。

### 3.2 为什么 `model` 只是部分兼容

因为对第三方而言，看到：

- `qwen3-tts-flash-realtime`
- `qwen3-tts-instruct-flash-realtime`

就能快速理解服务能力边界。

但底层实现仍是：

- 本地 Qwen3-TTS 模型
- 默认挂载 LoRA bundle
- 本地部署时的能力裁剪

### 3.3 为什么 Base64 `response.audio.delta` 要强兼容

因为这是第三方最容易感知、也最能降低接入成本的部分：

- 前端易消费；
- SDK 易封装；
- 文档与官方示例更容易对齐。

## 4. 对外表述建议

对外宣传时建议使用下面的表述：

- **“兼容 Qwen3-TTS / Qwen-TTS-Realtime 风格接口”**
- **“字段与事件命名尽量对齐官方调用习惯”**
- **“服务端默认挂载本地 LoRA bundle，具体可用 `voice` 以本服务文档为准”**

不建议使用下面这种过于绝对的表述：

- “完全等同官方云服务”
- “100% 官方原生兼容，无任何差异”

## 5. 当前结论

V1 最重要的不是“把每个官方细节都复刻出来”，而是：

1. 把最有价值的**公开调用体验**兼容到位；
2. 把 LoRA 默认挂载与 bundle 细节藏在服务端内部；
3. 让第三方一眼看懂、几乎不用学习新协议。
