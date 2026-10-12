# miniLaya

用 `bert-base-uncased` 在 [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) 上微调，得到一个 Laya 风格的类型化决策模型。一次前向同时给一道题的全部选项打分，推理返回 `choice` / `score` / `noul` 三种答案。

设计见 [docs/design.md](docs/design.md)，目标见 [docs/goals.md](docs/goals.md)，与 Jev、Laya 的分题型和速度对比见 [docs/comparison.md](docs/comparison.md)。

## 环境

Python 3.10 及以上。本仓库在 Python 3.13、Apple M1（MPS）上完成训练。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

命令都在仓库根目录执行，并带上 `PYTHONPATH=.`。

## 训练

```bash
PYTHONPATH=. .venv/bin/python -m minilaya train \
  --model bert-base-uncased \
  --output runs/bert-typed-decisions \
  --epochs 3 \
  --batch-size 8 \
  --max-len 512 \
  --head-max-len 256
```

只读取数据集配置 `all`（train 1200 个案例，test 400 个案例）。不要把四个工作流配置再加进来，否则会重复计数。

训练前按工作流分层，从 train 里留出 10% 案例做温度校准。官方 test 不参与训练，也不参与温度拟合。MPS 上编码器启用 gradient checkpointing。

检查点目录：

| 文件 | 内容 |
|---|---|
| `model.safetensors` | 编码器与决策头，磁盘上为 fp16，加载后为 fp32 |
| `config.json` | 长度、头层数、按题型的温度 |
| `encoder/`、`tokenizer/` | 重建 BERT 与分词器所需的配置 |
| `benchmark.json` | 官方 test 的准确率、KL、Brier、ECE |
| `predictions.jsonl` | 每条测试决策的校准后概率 |
| `demo_predict.json` | 测试集第一条的 Laya 形式预测与金标 |

已有一次完整运行的结果在 `runs/bert-typed-decisions`（该目录被 git 忽略）。

## 评估

```bash
PYTHONPATH=. .venv/bin/python -m minilaya eval \
  --checkpoint runs/bert-typed-decisions
```

指标定义见设计文档。准确率比较预测与金标的 argmax。KL 是 `KL(金标 ‖ 预测)`，单位为 nat。Brier 是选项上平方误差之和。ECE 用 15 个箱子，置信度取 `max(p)`。

## 推理

演示官方 test 的第一条：

```bash
PYTHONPATH=. .venv/bin/python -m minilaya predict \
  --checkpoint runs/bert-typed-decisions \
  --demo
```

自定义输入：`--state` 或 `--state-file` 提供状态，`--questions` 提供题目 JSON。题目是一个对象，键是题目 id，值包含 `type`、`instructions`、`criteria`。

```python
from minilaya import load

agent = load("runs/bert-typed-decisions")
result = agent.predict(state, questions)
```

`result["answers"]` 的形状：

- `choice`：`choice`、`probabilities`、`confidence`
- `score`：期望分 `score`、等级概率 `probabilities`（键为 `"0"`、`"1"`…）、`confidence`
- `noul`：`noul` 即 P(true)，以及 `probabilities`；没有 `confidence`

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

推理时间：

```bash
python scripts/bench_infer.py --checkpoint runs/bert-typed-decisions
```

## 本次 test 结果

2000 条官方测试决策，6 个 epoch 的检查点校准后准确率 0.648，KL 0.193，Brier 0.109，ECE 0.084。分题型准确率：choice 0.580，noul 0.747，score 0.625。

同一次官方 test 上，Jev 1.13.0 公布准确率 0.727，Laya typed-decisions 公布 0.766，数据集卡上的 ModernBERT-base 分工作流专家是 0.646。Laya 的分题型准确率是 choice 0.733、noul 0.857、score 0.723。Jev 没有公布分题型数字。Apple M1 上热身后的单题推理 p50 是 33.7 ms；Laya 英文检查点在 Tesla T4 上的对应公布值是 39.5 ms。设备、检查点和计时口径不同，完整对照在 [docs/comparison.md](docs/comparison.md)。
