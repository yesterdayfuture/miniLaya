# 目标

把一个开源编码器微调成 Laya 风格的类型化决策模型：读一段 state 和若干道题，一次前向给出选项分布，并用官方 test 报告可复核的基准。

## 已完成

这些是本仓库 0.1.0 的交付范围，当前代码和 `runs/bert-typed-decisions` 已经满足。

1. 骨干是 `bert-base-uncased`。英文基准、词表里有 `[CLS]` `[SEP]` `[MASK]`，110M 参数可以在 16GB 的 Apple M1 上全量微调。
2. 数据是 `LocalLLaMA/typed-decisions` 的配置 `all`。只使用官方 train / test，不把四个工作流配置重复计入。
3. 支持且只支持 `choice`、`score`、`noul`。选项顺序稳定，金标分布可归一化。
4. 序列在 `[MASK]` 上打分，选项不会被丢掉。超长 state 从尾部截断，并统计截断条数。
5. 默认训练损失是金标分布的 soft cross-entropy，入口是 `python -m minilaya train`。校准集在训练前按案例、按工作流留出。温度按题型拟合。官方 test 只在最后打一次分。RLCD 是同一 `train()` 上的另一次运行，不替换这条默认路径。
6. 推理对象与 Laya 对齐：choice 含选项、概率和 confidence；score 含期望分、等级概率和 confidence；noul 含 P(true) 和概率，不含 confidence。
7. 基准同时给出本次模型、本次均匀分布、本次先验，以及数据集卡上的已发表数字。均匀分布的 KL 与 Brier 用来确认评分器。

## 验收

一次 `train` 结束时，`runs/bert-typed-decisions` 里应有检查点、`benchmark.json`、`predictions.jsonl` 和 `demo_predict.json`。`eval` 能复现校准后的测试分数。`predict --demo` 能打印测试集第一条的预测和金标。`pytest tests` 通过。

本次已记录的官方 test（2000 条决策）：

| | 准确率 | KL | Brier | ECE |
|---|---:|---:|---:|---:|
| BERT，校准后 | 0.648 | 0.193 | 0.109 | 0.084 |
| BERT，温度 = 1 | 0.648 | 0.204 | 0.115 | 0.054 |
| 均匀分布 | 0.318 | 0.444 | 0.238 | 0.045 |
| 训练集先验 | 0.481 | 0.327 | 0.181 | 0.036 |

校准温度：choice 1.2，score 1.2，noul 1.3。六个 epoch 的校准 soft CE 从 1.050 降到 0.948，每一轮都低于上一轮，保留的是第 6 个 epoch。第 5 轮到第 6 轮只降了 0.005。训练在 MPS 上约 225 分钟。

相对本次先验，准确率高 0.167。相对同一官方 test 上的已发表结果：准确率与 ModernBERT-base 分工作流专家的 0.646 持平，仍低于零样本 Jev 1.13.0 的 0.727，也低于 Laya typed-decisions 的 0.766。分题型上，Laya 公布 choice 0.733、noul 0.857、score 0.723；本次是 0.580、0.747、0.625。Jev 没有公布分题型准确率。完整对照见 [comparison.md](comparison.md)。

同一切分、同一骨干上的 RLCD 运行（`scripts/train_rlcd.py`，6 个 epoch，检查点 `runs/bert-typed-decisions-rlcd`）校准后准确率 0.572，KL 0.259，Brier 0.145，ECE 0.071。温度 choice 1.1、score 1.1、noul 1.2。校准奖励从 -0.854 升到 -0.719，保留的是第 6 个 epoch。校准 soft CE 从 1.125 降到 1.018。分题型准确率：choice 0.523，noul 0.672，score 0.532。分数 MAE 0.462。训练在 MPS 上约 224 分钟。

## 读数字时的界限

- 金标是教师软标签。教师自一致准确率 0.735 是这份标签的上限参考。超过它通常是在拟合教师，而不是更接近客观事实。
- Jev 1.13.0 的准确率 0.727、KL 1.442 是零样本接口的成绩。本模型在同一批金标上训练过，两者不能比谁更高。
- 标量温度不改变 argmax。准确率相同是预期结果。本次温度把 KL 从 0.204 降到 0.193，Brier 从 0.115 降到 0.109，ECE 从 0.054 升到 0.084。
- 均匀分布 ECE 与数据集卡的 0.169 不一致，因为平坦分布的 argmax 取决于并列处理。KL 和 Brier 已与数据集卡对齐，评分器以这两项为准。
- 约 4% 的训练决策和 3% 的测试决策在 512 token 处丢掉了 state 尾部。这部分误差算在当前分数里。

## 不在本版范围内

- 默认 `train` 保持 soft CE。RLCD 写在 `scripts/train_rlcd.py`，检查点与 soft CE 分开。仍不做按选项数分桶的温度，也不换 ModernBERT / mmBERT 检查点。
- 不把四个工作流拆成四个专家。
- 不训练中文或多语决策。英文 `bert-base-uncased` 对非拉丁文字会保持高置信度同时失效；换语言应在进编码器之前换骨干。
- 不提供 HTTP 服务或网页界面。入口是命令行和 `minilaya.load`。
- 当前检查点的官方 test 准确率是 0.648。不把教师自一致 0.735 或 Laya 的 0.766 写成已经达到的目标。
