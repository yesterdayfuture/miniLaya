# Jev、Laya、miniLaya

三份数字都对着 `LocalLLaMA/typed-decisions` 的官方 test：400 个案例、2000 条决策。读的时候先看「怎么得到的」，再看高低。

| | 角色 | 骨干 | 怎么得到这个分数 |
|---|---|---|---|
| Jev 1.13.0 | 通用决策接口 | 未公开 | 零样本。数字来自 [laya-typed-decisions 模型卡](https://huggingface.co/convaiinnovations/laya-typed-decisions)，Laya 标明是第三方公布，没有在同一套提示上重测 |
| Laya typed-decisions | 这四个工作流上的专家 | ModernBERT-large，421M，上下文 1024 | 在 1200 个训练案例上用 RLCD 微调。模型卡自称，数据集讨论区把它标成 self-reported |
| miniLaya | 这四个工作流上的专家 | bert-base-uncased，110M，上下文 512 | 本仓库在 train 上用 soft CE 微调。校准集从 train 按案例留出，test 只打一次分 |

Jev 没见过这些工作流的训练标签。Laya 和 miniLaya 见过。数据集卡把这两种分数分成 generalist 和 specialist，并排时要带上这一列。教师自一致准确率是 0.735。Laya 的 0.766 高于这个上限。

## 总体

| | 准确率 | Brier | ECE | 分数 MAE | KL |
|---|---:|---:|---:|---:|---:|
| miniLaya | 0.648 | 0.109 | 0.084 | 0.360 | 0.193 |
| Jev 1.13.0 | 0.727 | 0.148 | 0.144 | 0.391 | — |
| Laya typed-decisions | 0.766 | 0.062 | 0.213 | 0.242 | — |

准确率：miniLaya 比 Jev 低 0.079，比 Laya 低 0.118。Laya 比 Jev 高 0.039。和数据集卡上 ModernBERT-base 分工作流专家的 0.646 持平，KL 0.193 低于对方的 0.223，Brier 0.109 低于对方的 0.119。教师自一致是 0.735，当前分数还在这条线下面。

这一版是 6 个 epoch 的检查点，校准温度 choice 1.2、score 1.2、noul 1.3。温度不改变 argmax，所以校准后和温度 = 1 的准确率都是 0.648。校准把 KL 从 0.204 降到 0.193，Brier 从 0.115 降到 0.109，ECE 从 0.054 升到 0.084。

Brier 和 ECE 越低越好。Jev 的 Brier 0.148 同时出现在数据集卡一脉的引用和 Laya 的模型卡上；miniLaya 的 0.109 是本仓库评分器算的，该评分器复现了数据集卡上均匀分布的 Brier 0.238。Laya 的 0.062 是他们自己公布的，本仓库没有他们的逐条概率，没有重算。

分数 MAE 只在 `score` 题上计算，是预测期望等级和金标期望等级之差的绝对值平均。miniLaya 的 0.360 来自 `predictions.jsonl` 的 800 条 score。Laya 公布 0.242，Jev 公布 0.391，数据集卡上 ModernBERT-base 专家是 0.444。

KL 只有本仓库和数据集卡的基线有数。Laya 和 Jev 的模型卡没有给出 KL。miniLaya 是 0.193 nat，数据集卡上的均匀分布是 0.444，先验是 0.347，ModernBERT-base 专家是 0.223。

Laya 还公布了 soft accuracy：Jev 0.580，Laya 0.471。模型卡没有写公式。本仓库没有把这一项算进 `benchmark.json`。

## 按题型

官方 test 的题量是 choice 600、noul 600、score 800。Jev 没有公布分题型准确率。

| 题型 | miniLaya 准确率 | miniLaya ECE | Laya 准确率 | Laya ECE |
|---|---:|---:|---:|---:|
| choice | 0.580 | 0.098 | 0.733 | 0.255 |
| noul | 0.747 | 0.059 | 0.857 | 0.192 |
| score | 0.625 | 0.123 | 0.723 | 0.199 |

和 Laya 的准确率差：choice 低 0.153，noul 低 0.110，score 低 0.098。三题型都低，choice 的差距最大。

miniLaya 分题型的分布指标：

| 题型 | KL | Brier | n |
|---|---:|---:|---:|
| choice | 0.251 | 0.127 | 600 |
| noul | 0.133 | 0.102 | 600 |
| score | 0.195 | 0.102 | 800 |

Laya 的分题型表只有准确率和 ECE。两边的 ECE 不必逐项对齐：Laya 文档里 noul 的 confidence 是 `max(p, 1-p)`，本仓库所有题型的 ECE 都用 `max(p)`。

## 按工作流

每个工作流 500 条决策。Jev 没有公布分工作流准确率。

| 工作流 | miniLaya | Laya | 差距 |
|---|---:|---:|---:|
| invoice_processing | 0.672 | 0.804 | 0.132 |
| security_incidents | 0.676 | 0.766 | 0.090 |
| customer_service | 0.664 | 0.764 | 0.100 |
| agent_trace_observability | 0.580 | 0.730 | 0.150 |

差距最大的仍是 agent trace。这一版准确率 0.580，已经高于本仓库算出的全体先验 0.481。

## 推理速度

三边不是同一台机器、同一种计时。下表只把各自公布或实测的口径写在旁边。

| | 设备 | 1 道题 | 5 道题（一个案例） | 口径 |
|---|---|---:|---:|---|
| Jev 1.13.0 | 托管 API | p50 236–276 ms | — | Laya 家族文档里的第三方对比，含网络往返 |
| Laya 英文检查点 | Tesla T4 | 39.5 ms（p95 42.1 ms） | 84.5 ms | 本地热身，`cuda.synchronize`。这是上下文 512 的 `laya`，不是 typed-decisions 检查点 |
| miniLaya | Apple M1，MPS | p50 33.7 ms（p95 34.6 ms） | p50 149.5 ms（p95 161.3 ms） | `scripts/bench_infer.py`，2026-10-12，官方 test 第 0 条，预热 3 次后计 20 次 |

miniLaya 这一次的序列长度：1 道题 182 token；5 道题为 149、182、182、183、196。加载检查点 4.6 s。第一次调用 5 道题 247 ms，里面包含 MPS 编译，没有放进 p50。

热身之后，miniLaya 在 M1 上的单题时间和 Laya 在 T4 上的 39.5 ms 落在同一档。一个案例的 5 道题，T4 上的 Laya 英文检查点是 84.5 ms，本机 miniLaya 是 149.5 ms。Jev 的 236–276 ms 是 API 单题延迟，含网络，不能当成模型前向时间。

typed-decisions 那个 421M 检查点的当前模型卡没有单独的延迟表。它的上下文是 1024，比上面那条 T4 数字所用的 512 上下文更长。

复测：

```bash
python scripts/bench_infer.py --checkpoint runs/bert-typed-decisions --repeat 20
```
