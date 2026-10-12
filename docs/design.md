# 设计

miniLaya 把预训练 BERT 微调成非自回归的类型化决策模型。一道题的全部选项在同一次双向编码里打分，再变成 Laya 形式的 `choice`、`score` 或 `noul`。

参考实现是 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)。Laya 用 ModernBERT 或 mmBERT 编码器、RLCD 损失和按题型与选项数分桶的温度。本仓库换成 `bert-base-uncased`、金标分布上的 soft cross-entropy，以及每种题型一个标量温度。Jev 是闭源托管接口，权重、奖励和结构都未公开，不能作为微调起点。

## 模块

| 模块 | 职责 |
|---|---|
| `schema` | 题型、选项顺序、金标归一化、Laya 答案对象 |
| `sequence` | 把一道题装进 BERT 序列，并记下每个选项的 `[MASK]` 位置 |
| `model` | BERT、题型嵌入、一层 Transformer 头、每个标记一个 logit |
| `data` | 读取 `LocalLLaMA/typed-decisions` 配置 `all`，按案例切出校准集 |
| `batch` | 把变长序列补到长度桶，把选项轴补齐 |
| `train` | 训练、按校准集 soft CE 选 epoch、在官方 test 上打分 |
| `calibrate` | 每种题型在 0.5 到 5.0 上网格搜索温度 |
| `metrics` | 准确率、KL、Brier、ECE，以及数据集卡上的对照数字 |
| `predict` | 加载检查点，返回与训练共用的答案对象 |
| `cli` | `train`、`eval`、`predict` |

## 数据

只加载 Hugging Face 配置 `all`。它是四个工作流的并集：`agent_trace_observability`、`customer_service`、`invoice_processing`、`security_incidents`。每个工作流 train 300 个案例、test 100 个案例，每个案例 5 道题。合计 train 1200 个案例 / 6000 条决策，test 400 / 2000。四个工作流配置与 `all` 不能同时加载，否则行数翻倍。

一条样本是 `{id, workflow, split, state, questions, gold}`。`state` 若已是字符串则原样送进编码器，避免基准文本被重新序列化。`questions` 和 `gold` 是 JSON。金标是三次教师采样（约 4B 参数，温度 0.7）的平均分布，是教师软标签，不是客观真值。教师在该测试集上的自一致准确率是 0.735；明显高于这个数通常是在拟合教师的习惯。

题型只有三种：

| type | criteria | 金标概率的键 | 答案里的点估计 |
|---|---|---|---|
| `choice` | 选项名到说明的字典，键顺序即选项顺序 | 选项名 | argmax 选项 |
| `score` | 等级说明的列表，或等级名到说明的字典 | `"0"`、`"1"`… | 等级下标的期望 |
| `noul` | 可选；缺省用 false/true 的固定英文 | `false`、`true` | P(true) |

`score` 的金标有时用等级序号而不是 rubric 原文做键。对不齐时按数值键的顺序回退，长度必须与等级数相同。概率必须非负有限；和为 0 时退化为均匀分布。

## 序列

布局与 Laya 相同：

```text
[CLS] instructions [SEP] [MASK] option [MASK] option ... [SEP] state [SEP]
```

logit 只从 `[MASK]` 读取。选项一个都不能丢。题干加选项先限制在 `head_max_len`（默认 256）以内：先缩短 instructions，再按选项均分剩余 token。state 填进 `max_len`（默认 512）的剩余位置。标记必须全部留在最终序列里，否则直接报错。文本里若出现 mask 词本身，编码前会换成空格，避免假标记。

本次全量编码后，训练决策的长度中位数是 336，最长顶到 512。state 被截断的数量：训练 211/5400，校准 26/600，测试 66/2000。

## 模型

`DecisionBert`：

1. BERT 最后一层隐状态。
2. 加上题型嵌入。`choice=0`、`score=1`、`noul=2`，同一道题的每个位置加同一个向量。
3. 可选的 norm-first `TransformerEncoder`。默认 1 层，头数为 `hidden // 64`（BERT-base 为 12），前馈宽度为 `4 * hidden`，padding 位置用 `src_key_padding_mask` 挡住。`head_layers=0` 时跳过这一层。
4. 在每个 `[MASK]` 上做 LayerNorm → Linear → GELU → Linear，得到一个 logit。无效标记的 logit 填 `-1e4`，softmax 时被压掉。

`bert-base-uncased` 的编码器约 109.5M 参数，决策头约 7.7M。加载 `AutoModel` 时 MLM 和 NSP 头的权重是多余的，会被丢掉。

默认优化：AdamW，编码器学习率 `2e-5`，决策头 `1e-4`，权重衰减 0.01，偏置和 LayerNorm 不做衰减。命令默认 3 个 epoch；当前检查点用 6 个 epoch 训完。线性 warmup 占全部 step 的 10%，梯度裁剪 1.0。同长度的样本打包成一个 batch，再把序列补到 64 的倍数，避免每个 batch 一种长度。选项数不同的 batch 在评估时先把选项轴补齐再拼接。

设备顺序是 CUDA、MPS、CPU。MPS 上使用 fp32，并打开编码器的 gradient checkpointing。batch 8、序列 512 时，不打开 checkpointing 会在统一内存水位上卡住。

## 损失与温度

训练损失是软标签交叉熵：

```text
- sum_k gold_k * log softmax(logit)_k
```

只在真实选项上求和。金标已经是分布，所以用 soft CE，而不是 RLCD。Laya 的 RLCD 在其 issue #741 里没有带来收益，且有一次纯 soft CE 高于 RLCD。

每个 epoch 结束时在校准集上算未调温度的 soft CE，保留最低的那一版权重。校准集在任何梯度之前划出，按工作流分层，默认 10%，且按案例而不是按题目切，避免同一案例的题目漏进两边。默认得到训练 1080 个案例 / 5400 条决策，校准 120 / 600。

温度在校准集上按题型网格搜索，范围 `[0.5, 5.0]`、46 个点，目标仍是 soft CE。某题型校准样本少于 8 条时温度保持 1。正的标量温度不改变 argmax，因此准确率不变，KL、Brier、ECE 会变。当前检查点的拟合结果：choice 1.2，score 1.2，noul 1.3。

## 推理输出

`DecisionAgent.predict(state, questions)` 把该状态下的全部题目打成一个 batch，除以对应题型的温度，再解码：

```json
{
  "model": "bert-base-uncased",
  "answers": {
    "action": {
      "type": "choice",
      "choice": "human_review",
      "probabilities": {"continue": 0.20, "human_review": 0.38},
      "confidence": 0.38
    },
    "risk": {
      "type": "score",
      "score": 0.79,
      "probabilities": {"0": 0.41, "1": 0.43, "2": 0.12, "3": 0.04},
      "confidence": 0.43
    },
    "needs_review": {
      "type": "noul",
      "noul": 0.51,
      "probabilities": {"false": 0.49, "true": 0.51}
    }
  }
}
```

`confidence` 是 `max(p)`。`score` 是 `sum_i i * p_i`。`noul` 是 P(true)，不带 `confidence`。金标也用同一解码函数变成同样的对象，便于并排查看。

## 指标

对每条决策：

- 准确率：`argmax(预测) == argmax(金标)`。并列时 NumPy 取第一个最大值。
- KL：`sum_k gold_k * log(gold_k / pred_k)`，再对决策取平均。
- Brier：`sum_k (pred_k - gold_k)^2`，再对决策取平均。
- ECE：15 个等宽箱子，置信度是 `max(p)`，正确性是 argmax 是否一致。

均匀分布的准确率按 `mean(1/K)` 计算，也就是机会准确率。对平坦向量取 argmax 会永远落到下标 0，不能当作均匀基线。数据集卡写的机会准确率是 0.308，本次按官方 test 算出 0.318。KL 与 Brier 和数据集卡一致（0.444、0.238）。均匀分布的 ECE 对 argmax 的并列处理敏感，本次是 0.045，数据集卡是 0.169，这项不必强行对齐。

先验是训练集软标签按 `workflow + 题目 id` 取平均，再拿到 test 上打分。校准集也算在这个平均里，因为它来自官方 train。本次先验准确率 0.481，数据集卡 0.470；卡上的先验更接近硬标签频率。

`metrics.PUBLISHED` 里的 Jev、教师自一致、ModernBERT 专家和 Laya 数字来自数据集卡与 Laya 的已发表结果，是对照，不是本次运行测出来的。分题型准确率、工作流和推理速度的并排表在 [comparison.md](comparison.md)。
