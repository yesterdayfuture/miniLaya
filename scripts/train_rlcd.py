"""用 RLCD 在 typed-decisions 上微调 6 个 epoch，并在官方 test 上评估。

与 `python -m minilaya train` 共用数据切分、模型和温度校准。差别只在训练损失：
对数分 + 0.5 × 球面分，score 题再减排名概率分；每题 4 个 logit 噪声样本，
噪声标准差从 1.0 线性降到 0.3，优势是组内奖励减去组内均值。

检查点写到 runs/bert-typed-decisions-rlcd，不覆盖 soft CE 的 runs/bert-typed-decisions。
"""

from __future__ import annotations

import argparse

from minilaya.train import train


def build_parser() -> argparse.ArgumentParser:
    """参数与 `minilaya train` 对齐，默认改成这次 RLCD 实验的设置。

    输出目录和 epoch 数与 soft CE 检查点分开：6 个 epoch，写到另一个目录。
    学习率、长度、校准比例保持和普通微调相同，方便并排看损失本身的差别。
    """
    parser = argparse.ArgumentParser(description="用 RLCD 微调 BERT，并在官方 test 上评估。")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--output", default="runs/bert-typed-decisions-rlcd")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--head-max-len", type=int, default=256)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--calib-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="只取每个切分的前 N 个案例，0 表示全量")
    parser.add_argument("--rlcd-group", type=int, default=4, help="每题的噪声报告数")
    # 起点 1.0、终点 0.3 是公开描述里的噪声退火。组大小 4 对应本仓库对 Laya 更新的描述。
    parser.add_argument("--rlcd-noise-start", type=float, default=1.0)
    parser.add_argument("--rlcd-noise-end", type=float, default=0.3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # train() 靠这个字段分流。不设的话会落到 soft CE。
    args.loss = "rlcd"
    train(args)


if __name__ == "__main__":
    main()
