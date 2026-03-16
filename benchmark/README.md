# Benchmark Data Layout

根目录 `benchmark/` 用于保存每次 benchmark 的原始产物，避免和 `streaming_lora_service/docs/validation/` 中已经整理过的验证结论混放。

## 命名规则

- 单次运行目录命名：`<summary>_vN`
- `<summary>`：本轮 benchmark 的主题，建议使用小写下划线，例如：
  - `candidate8_v2_ws`
  - `candidate8_v2_http`
  - `candidate8_v2_ws_load`
- `vN`：版本号，从 `v1` 开始递增；同主题重跑时创建新目录，不覆盖旧数据

## 推荐目录内容

- `*_metrics.json`：本轮 benchmark 的指标汇总
- `audio/`：保存每轮收到的音频样本，便于试听排查（其中 `.wav` 默认不纳入版本控制）
- `server.log`：服务启动与运行日志
- `notes.md`：可选，记录本轮参数、环境和结论

## 当前已创建目录

- `candidate8_v2_ws_v1/`：首次 WebSocket benchmark
