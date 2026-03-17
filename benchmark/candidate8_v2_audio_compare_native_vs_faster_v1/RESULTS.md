# candidate8_v2 audio compare: native vs faster

## Inputs

- native source audio: `benchmark/candidate8_v2_ws_v2/audio_zh_formal/`
- faster source audio: `benchmark/candidate8_v2_ws_faster_v1/audio_zh_formal/`
- pairing rule: same filename / same benchmark iteration (`custom_iter01.wav` ~ `custom_iter03.wav`)

## Summary

From `audio_compare_metrics.json`:

- paired files: `3`
- sample rate match: `true`
- channel count match: `true`
- native mean duration: `9.0933 s`
- faster mean duration: `9.5467 s`
- mean duration delta (`faster - native`): `0.4533 s`
- mean absolute error over truncated aligned samples: `0.041171`
- RMSE mean: `0.058662`
- Pearson correlation mean: `-0.002179`

## Pair-wise notes

- `custom_iter01.wav`: faster 比 native 短约 `0.56 s`
- `custom_iter02.wav`: faster 比 native 长约 `1.44 s`
- `custom_iter03.wav`: faster 比 native 长约 `0.48 s`

## Interpretation

这份对照更适合回答“输出格式是否一致、波形差异量级大概如何”，不适合单独当作主观音质结论，原因有两点：

1. native 与 faster 当前并不是 bit-identical 路径；
2. benchmark 采样本身带随机性，即便文本相同、iteration 名相同，也不代表 token 序列严格对齐。

所以：

- `sample_rate/channel` 一致，说明协议输出格式对齐；
- `duration` 有一定波动，说明 faster 路径当前生成长度分布和 native 不完全等价；
- `pearson_corr` 接近 `0` 并不直接表示质量差，而是提示“这不是同一条波形的逐样本回放”，更多是采样路径差异。

## Recommended next step

如果要继续做更像“质量回归”的对照，建议下一轮：

1. 在 native / faster 两边都固定随机种子；
2. 对同一批文本生成更多样本；
3. 增加主观试听记录，或引入 mel / F0 / 语速等更稳健的音频特征比较。
