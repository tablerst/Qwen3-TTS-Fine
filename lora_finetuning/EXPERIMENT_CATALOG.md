# Qwen3-TTS LoRA 实验 Catalog

> 最后更新：2026-03-11  
> 作用：把当前工作区中**已经实际产生的训练 / bundle / A/B 结果**整理成可追溯复盘台账，避免后续只凭记忆做判断。

---

## 1. 记录范围与说明

本 catalog 只记录两类内容：

1. **工作区中已经存在真实产物的实验**
2. **当前会话中已经明确得到主观试听结论的实验**

说明：

- 自动指标主要来自各实验目录下的 `metrics.json`
- 数据构成主要来自对应数据目录下的 `*_dataset_summary.json`
- 主观结论来自本轮会话的实际试听反馈
- 若某项是**派生 bundle / speaker patch A/B**，会明确标记为“非训练实验”
- 若某项**尚未执行**，不会写成“已完成”

---

## 2. 当前总览

| ID | 实验标签 | 类型 | 核心变化 | 当前结论 |
| --- | --- | --- | --- | --- |
| E0 | `formal_single_speaker_rerun_20260310_stable` | 训练 | 保守版 `talker_only + q/v/o` | 稳，但音色推进偏弱，可作为保守基线 |
| E1 | `formal_single_speaker_timbre_transfer_20260310` | 训练 | 扩大 LoRA 到 `q/k/v/o + gate/up/down` | 日语更像，ZH 开始带出一点声线影子 |
| E1A | `timbre_transfer_refcand8_bundle_ab` | 非训练实验 | 保持 adapter 不变，仅替换 `speaker_embedding` | 某些场景更像，说明 `ref_audio` 确实重要 |
| E2 | `candidate8_multilingual_calibration_20260310_v1` | 训练 | 从 base 重新训练，加入较高比例 synthetic ZH/EN | 听感整体更不像，方向错误 |
| E3 | `candidate8_multilingual_warmstart_20260310_v1` | 训练 | 基于 E1 warm-start，小比例多语校准 | 当前方向正确：JA 基本保住，ZH/EN 稍有增强 |
| E4 | `candidate8_multilingual_warmstart_20260310_v2` | 训练 | 基于 E3 best-checkpoint 继续 warm-start，并下调 LR 到 `5e-6` | 当前多语主线最佳候选：JA 稳，ZH/EN 仍可继续优化 |
| E5 | `candidate8_multilingual_warmstart_20260311_v3` | 训练 | 基于 E4 warm-start，并把 ZH 从 `3` 条加到 `5` 条 | 自动指标更好，但主观听感不如 v2，暂不作为主线 |

---

## 3. 详细实验记录

### E0. `formal_single_speaker_rerun_20260310_stable`

**定位**

- 保守稳定基线
- 用于和后续“更强音色迁移版”比较

**关键文件**

- 配置：`lora_finetuning/configs/formal_single_speaker_local_1p7b.yaml`
- 输出：`outputs/lora_formal_single_speaker_1p7b_rerun_20260310_stable`
- bundle：`outputs/lora_formal_single_speaker_1p7b_rerun_20260310_stable_bundle_best`

**关键配置**

- `target_scope: talker_only`
- `target_modules: q_proj, v_proj, o_proj`
- `r: 8`
- `alpha: 16`
- `dropout: 0.05`
- `learning_rate: 5e-5`
- `num_epochs: 2`
- `validation_split_ratio: 0.1`

**关键指标**

- `trainable_params`: `2,523,136`
- `trainable_ratio`: `0.1307%`
- best validation loss: `9.244953155517578`
- best checkpoint: `outputs/lora_formal_single_speaker_1p7b_rerun_20260310_stable/best-checkpoint`

**主观结论**

- 优点：稳定、保守、不容易炸
- 缺点：更像“轻微偏移”，不够像“把 speaker identity 打进去”
- 用途：适合作为保守基线，不适合作为最终跨语言音色迁移候选

---

### E1. `formal_single_speaker_timbre_transfer_20260310`

**定位**

- 第一版明确以“音色迁移优先”为目标的训练
- 相比 E0，扩大了 `talker` 主干的 LoRA 范围

**关键文件**

- 配置：`lora_finetuning/configs/formal_single_speaker_timbre_transfer_1p7b_20260310.yaml`
- 输出：`outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310`
- bundle：`outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best`
- metrics：`outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310/metrics.json`

**关键配置**

- `target_scope: talker_only`
- `target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `r: 16`
- `alpha: 32`
- `dropout: 0.05`
- `learning_rate: 3e-5`
- `num_epochs: 2`
- `sub_talker_loss_weight: 0.2`

**关键指标**

- `trainable_params`: `17,432,576`
- `trainable_ratio`: `0.8958%`
- validation loss:
  - epoch 0: `8.70765495300293`
  - epoch 1: `8.10518741607666`
- best checkpoint: `outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310/best-checkpoint`

**主观结论**

- 日语音色明显比 E0 更像
- 中文开始带出一点“这个人的声线影子”
- 这是当前整条路线真正转向正确方向的起点

**备注**

- E1 是后续 warm-start 多语校准的母体实验
- 如果只看“像不像目标人”，E1 比 E0 更接近目标

---

### E1A. `timbre_transfer_refcand8_bundle_ab`

**定位**

- 非训练实验
- 用于验证 `ref_audio` / `speaker_embedding` 对最终 bundle 的影响

**关键文件**

- 基础 bundle：`outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best`
- A/B 版本：`outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best_refcand8`
- 新参考音频：`resources/reference/yachiyo_ref_candidate8_20260310.wav`

**实验方式**

- 保持 E1 的 `adapter/` 不变
- 仅替换 `speaker_embedding.safetensors`
- 重新生成 JA / ZH / EN 试听

**主观结论**

- 某些场景下更像目标音色
- 说明 `ref_audio` 的选择确实会影响最终“像不像”
- 但只换 `speaker patch` 不能单独解决 ZH/EN 迁移偏弱的问题

**当前价值**

- `...bundle_best_refcand8` 是后续所有比较里很重要的“单语主线版本”
- 它证明了：
  1. E1 的 adapter 主体路线是对的
  2. 更好的 `ref_audio` 能进一步抬高相似度上限

---

### E2. `candidate8_multilingual_calibration_20260310_v1`

**定位**

- 第一版“多语校准”实验
- 目标是把 candidate8 的 JA + synthetic ZH/EN 一起用于训练
- 训练方式：**从 base 重新开始**，不是 warm-start

**关键文件**

- 配置：`lora_finetuning/configs/candidate8_multilingual_calibration_1p7b_20260310_v1.yaml`
- 数据摘要：`resources/candidate8_20260310/candidate8_multilingual_calibration_20260310_v1/candidate8_multilingual_calibration_20260310_v1_dataset_summary.json`
- 输出：`outputs/lora_candidate8_multilingual_calibration_1p7b_20260310_v1`
- bundle：`outputs/lora_candidate8_multilingual_calibration_1p7b_20260310_v1_bundle_best`

**数据构成**

- train:
  - JA: `31`
  - ZH: `6`
  - EN: `8`
  - total: `45`
- validation:
  - JA: `4`
  - ZH: `2`
  - EN: `2`
  - total: `8`
- 共享参考音频：`resources/reference/yachiyo_ref_candidate8_20260310.wav`

**关键配置**

- `target_scope: talker_only`
- `target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `r: 16`
- `alpha: 32`
- `learning_rate: 2e-5`
- `num_epochs: 4`
- `validation_source: external_jsonl`

**关键指标**

- validation loss:
  - epoch 0: `11.142566680908203`
  - epoch 1: `9.6207857131958`
  - epoch 2: `9.092447280883789`
  - epoch 3: `8.970254898071289`
- best checkpoint: `outputs/lora_candidate8_multilingual_calibration_1p7b_20260310_v1/best-checkpoint`

**主观结论**

- 听感整体**更不像**目标人声
- 虽然 loss 在下降，但模型更像是在拟合 synthetic 多语老师输出
- 这是一个**负例实验**，非常值得保留

**复盘结论**

- 失败原因不是训练崩掉，而是训练目标被带偏：
  1. 从 base 重开，丢掉了 E1 已经学到的目标 timbre
  2. ZH/EN synthetic 数据占比过高
  3. synthetic 数据更像“校准信号”，不适合在这一比例下重定义 speaker identity

**结论标签**

- `status: archived_negative_example`

---

### E3. `candidate8_multilingual_warmstart_20260310_v1`

**定位**

- 当前主线最有价值的多语校准实验
- 核心思想：
  - 不从 base 重开
  - 基于 E1 的最佳 adapter warm-start
  - 用很小比例的 ZH/EN synthetic 做小步校准

**关键文件**

- 配置：`lora_finetuning/configs/candidate8_multilingual_warmstart_1p7b_20260310_v1.yaml`
- 数据摘要：`resources/candidate8_20260310/candidate8_multilingual_warmstart_20260310_v1/candidate8_multilingual_warmstart_20260310_v1_dataset_summary.json`
- 输出：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v1`
- bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v1_bundle_best`
- warm-start 来源：`outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310/best-checkpoint/adapter`

**数据构成**

- train counts:
  - Japanese: `33`
  - Chinese: `3`
  - English: `3`
- train durations:
  - Japanese: `86.93s`
  - Chinese: `23.45s`
  - English: `20.90s`
- validation counts:
  - Japanese: `2`
  - Chinese: `1`
  - English: `1`

**关键配置**

- `init_adapter_dir`: `outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310/best-checkpoint/adapter`
- `target_scope: talker_only`
- `target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `r: 16`
- `alpha: 32`
- `learning_rate: 7.5e-6`
- `num_epochs: 2`
- `early_stopping_patience: 1`

**关键指标**

- validation loss:
  - epoch 0: `8.506144523620605`
  - epoch 1: `8.480901718139648`
- best checkpoint: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v1/best-checkpoint`

**主观结论**

- `JA 基本保住`
- `ZH/EN 比单语版多一点声线影子`
- `同时不再像上一版那样整体训歪`

**复盘结论**

- 这证明 warm-start 多语小步校准是**正确方向**
- 它不是“飞跃式提升”，但它做到了：
  1. 不明显破坏已像的 JA 主体
  2. 给 ZH/EN 增加一点 timbre anchor
  3. 避免 E2 那种从 base 重开的整体跑偏

**结论标签**

- `status: current_best_multilingual_direction`

---

### E4. `candidate8_multilingual_warmstart_20260310_v2`

**定位**

- 在 E3 已证明“warm-start 小步多语校准”方向正确后，做一次真正独立命名的续训实验
- 目标不是引入新变量，而是：
  - 明确从 `E3 best-checkpoint` 继续训练
  - 仅下调学习率
  - 观察是否能在不伤 JA 主体的前提下，再给 ZH/EN 多一点 timbre anchor

**关键文件**

- 配置：`lora_finetuning/configs/candidate8_multilingual_warmstart_1p7b_20260310_v2.yaml`
- 输出：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2`
- bundle：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`
- 试听：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best/samples/`
- warm-start 来源：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v1/best-checkpoint/adapter`

**数据构成**

- 与 E3 相同，未改数据配比：
  - train counts:
    - Japanese: `33`
    - Chinese: `3`
    - English: `3`
  - train durations:
    - Japanese: `86.93s`
    - Chinese: `23.45s`
    - English: `20.90s`
  - validation counts:
    - Japanese: `2`
    - Chinese: `1`
    - English: `1`

**关键配置**

- `init_adapter_dir`: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v1/best-checkpoint/adapter`
- `target_scope: talker_only`
- `target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `r: 16`
- `alpha: 32`
- `learning_rate: 5e-6`
- `num_epochs: 2`
- `early_stopping_patience: 1`

**关键指标**

- validation loss:
  - epoch 0: `8.398953437805176`
  - epoch 1: `8.394292831420898`
- best checkpoint: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2/best-checkpoint`

**主观结论**

- 这轮至少解决了“只讨论了 v2，但没有独立产物路径”的追溯问题
- 从自动指标看，`v2` 比 `v1` 继续小幅变好：
  - `v1 best val loss = 8.480901718139648`
  - `v2 best val loss = 8.394292831420898`
- 当前听感判断可概括为：
  - `JA 继续稳住`
  - `ZH 仍有提升空间`
  - 因而更像是“把正确方向再往前推了一小步”，而不是终局版
- 截至 `v3` 试听后的复盘，`v2` 仍是当前**主观上更好的多语版本**

**复盘结论**

- 这说明继续从已像版本的 `best-checkpoint` 往下 warm-start，是比重新回到 base 更安全的路线
- 只降学习率、保持数据不变，也能带来小幅稳定收益
- 但如果下一轮目标是**继续强化中文音色**，光靠再降 LR 可能不够，下一步更值得试的是：
  1. 仍然沿用 warm-start
  2. 小幅增加 ZH 比例
  3. 明确控制增量，避免回到 E2 那种“synthetic 把 speaker identity 带偏”的情况

**结论标签**

- `status: current_best_multilingual_direction`

---

### E5. `candidate8_multilingual_warmstart_20260311_v3`

**定位**

- 在 E4 基础上尝试一版“中文加量”的延续实验
- 目标是验证：
  - 继续沿用 warm-start 主线
  - 保持 EN 不变
  - 将 ZH 训练样本从 `3` 条提高到 `5` 条
  - 是否能继续提升中文音色

**关键文件**

- 配置：`lora_finetuning/configs/candidate8_multilingual_warmstart_1p7b_20260311_v3.yaml`
- 数据摘要：`resources/candidate8_20260310/candidate8_multilingual_warmstart_20260311_v3/candidate8_multilingual_warmstart_20260311_v3_dataset_summary.json`
- 输出：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260311_v3`
- bundleTest：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260311_v3_bundleTest`
- warm-start 来源：`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2/best-checkpoint/adapter`

**数据构成**

- train counts:
  - Japanese: `33`
  - Chinese: `5`
  - English: `3`
- train durations:
  - Japanese: `86.93s`
  - Chinese: `38.784s`
  - English: `20.904s`
- validation counts:
  - Japanese: `2`
  - Chinese: `1`
  - English: `1`
- 相比 E4 新增的中文训练样本：
  - `01_zh_01_greeting_zh`
  - `06_zh_06_question_zh`

**关键配置**

- `init_adapter_dir`: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2/best-checkpoint/adapter`
- `target_scope: talker_only`
- `target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `r: 16`
- `alpha: 32`
- `learning_rate: 4e-6`
- `num_epochs: 2`
- `early_stopping_patience: 1`

**关键指标**

- validation loss:
  - epoch 0: `8.347177505493164`
  - epoch 1: `8.336464881896973`
- 对比 E4：
  - `E4 best val loss = 8.394292831420898`
  - `E5 best val loss = 8.336464881896973`
- best checkpoint: `outputs/lora_candidate8_multilingual_warmstart_1p7b_20260311_v3/best-checkpoint`

**主观结论**

- 尽管自动指标继续改善，但本轮试听反馈是：`v3 效果不如 v2`
- 也就是说，这一版出现了非常重要的现象：
  - `validation loss 更低`
  - 但 `主观音色并没有更好`
- 当前结论应以主观听感为准，因此 `v3` **不取代** `v2`

**复盘结论**

- 这说明在当前这套数据和目标下，**继续增加 ZH synthetic 比例并不保证主观音色继续变好**
- `v3` 不是无效实验，它的重要价值在于提醒：
  1. 自动指标不能替代试听判断
  2. 中文样本加量要非常克制
  3. 下一轮如果继续做中文增强，不应默认从 `v3` 再往下堆，而应优先回到 `v2` 重新设计变量

**结论标签**

- `status: superseded`

---

## 4. 代码里程碑（影响实验可追溯性）

### M1. `sft_12hz_lora.py` 已支持 warm-start 续训

当前仓库已经支持：

- CLI：`--init_adapter_dir`
- YAML：`artifacts.init_adapter_dir`

能力含义：

- 先按当前配置注入 LoRA 结构
- 再从已有 `adapter/` 中加载权重继续训练
- 并校验 `target_modules / r / alpha / dropout / bias` 是否兼容

相关文件：

- `lora_finetuning/sft_12hz_lora.py`
- `lora_finetuning/common.py`
- `tests/test_lora_smoke.py`

验证状态：

- `python -m unittest tests.test_lora_smoke -v` 已通过

这项改动非常关键，因为 E3 的成立依赖于 warm-start 入口。

---

## 5. 当前推荐保留的主线候选

### 单语主线候选

`outputs/lora_formal_single_speaker_1p7b_timbre_transfer_20260310_bundle_best_refcand8`

适用：

- 以“目标音色相似度”为第一优先级
- JA 主体最重要
- ZH/EN 只要求带一点影子即可

### 多语主线候选

`outputs/lora_candidate8_multilingual_warmstart_1p7b_20260310_v2_bundle_best`

适用：

- 希望在不明显破坏 JA 的前提下，给 ZH/EN 再补一点 timbre
- 当前主观试听下仍是最值得保留的多语候选
- 如果继续做后续实验，应优先把它当作新的 warm-start 出发点，而不是默认沿用 `v3`

---

## 6. 明确未执行项

当前**没有已经命名但尚未执行**的多语主线实验。

说明：

- `v2`、`v3` 都已经作为真实独立实验执行并留痕
- 如果继续做下一轮中文增强，请重新命名为新的版本（例如 `v4`），并明确：
  - warm-start 来源
  - 数据配比变化
  - 新输出目录
  - 新 bundle / 试听目录

---

## 7. 后续新增实验时的填写规范

每新增一轮实验，建议最少补以下字段：

1. `experiment_tag`
2. `config path`
3. `train data summary`
4. `warm-start source`（如果有）
5. `output_root`
6. `bundle path`
7. `best validation loss`
8. `subjective listening summary`
9. `status`

推荐状态值：

- `baseline`
- `candidate`
- `current_best_direction`
- `archived_negative_example`
- `superseded`

---

## 8. 一句话结论

截至 2026-03-11，当前最清晰的实验结论是：

> **“扩大 `talker` 主干 LoRA 范围 + 更优 `ref_audio` + 基于已像版本 best-checkpoint 的小步 warm-start 校准” 是有效路线；但在当前条件下，继续增加 ZH synthetic 比例（`3 -> 5`）虽然能进一步降低 validation loss，却未能带来更好的主观听感，因此当前多语最佳仍是 `v2`。”**
