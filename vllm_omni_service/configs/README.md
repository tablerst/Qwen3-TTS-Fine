# 配置说明

这个目录用于保存本地实验使用的 `qwen3_tts` stage config。

## 为什么当前先不直接提交完整 YAML

原因很简单：

1. 当前仓库虚拟环境里还没有安装 `vllm_omni`；
2. 上游 stage config 处于持续迭代中；
3. 如果现在手工抄一份不完整 YAML，很容易变成“看起来像能跑，实际上坑更多”的配置陷阱。

所以这里采用的策略是：

- **优先从已安装的 `vllm_omni` 包里同步官方 `qwen3_tts.yaml`**；
- 然后把同步后的文件保存为：
  - `configs/qwen3_tts.stage.yaml`
- 后续再基于这个本地副本做最小修改。

## 如何同步

使用：

- `../scripts/sync_stage_config.py`

它会：

- 自动查找 `vllm_omni` 包路径；
- 定位 `model_executor/stage_configs/qwen3_tts.yaml`；
- 复制到当前目录。

## 上游配置里我们最关心的键

即便还没把完整 YAML 落盘，下面这些键已经可以视为首版必须关注的重点：

### 流式相关

- `async_chunk: true`
- connector `extra.codec_streaming: true`
- `codec_chunk_frames: 25`
- `codec_left_context_frames: 25`

### stage 划分

- stage 0：talker / AR
- stage 1：code2wav / generation

### 资源相关

- stage 0 `gpu_memory_utilization` 约为 `0.3`
- stage 1 `gpu_memory_utilization` 约为 `0.2`

> 上述数值来自上游公开文档和源码上下文，最终仍以同步到本地的官方 YAML 为准。

## 建议做法

1. 先同步官方 YAML；
2. 在副本上最小修改；
3. 每次修改后保留 diff 说明，避免未来回不去；
4. 对于 PoC 阶段，优先保证“能稳定流式”，再考虑大幅调 chunk/window/memory 参数。
