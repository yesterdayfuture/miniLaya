"""训练、评估和 Laya 形式推理的命令行。

train 用 soft CE 微调，并在官方 test 上写 benchmark.json。RLCD 走 scripts/train_rlcd.py。
eval 只重打已有检查点。
predict 对一条 state 输出 answers；--demo 使用官方 test 的一条案例。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minilaya.predict import DecisionAgent, dump_result, format_answers, resolve_device
from minilaya.train import evaluate_checkpoint, train


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    """train 和 eval 共用的运行参数。默认 batch 4。MPS 上全量训练用过 batch 8，并打开编码器的 gradient checkpointing。"""
    parser.add_argument("--device", default="auto", help="auto、cpu、mps 或 cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="只取每个切分的前 N 个案例，0 表示全量")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minilaya",
        description="用 BERT 在 LocalLLaMA/typed-decisions 上微调，推理时返回 Laya 形式的 choice / score / noul。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="用 soft CE 微调，并在官方 test 上评估")
    train_parser.add_argument("--model", default="bert-base-uncased")
    train_parser.add_argument("--output", default="runs/bert-typed-decisions")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--head-lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup", type=float, default=0.1)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--max-len", type=int, default=512)
    train_parser.add_argument("--head-max-len", type=int, default=256)
    train_parser.add_argument("--head-layers", type=int, default=1)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--calib-frac", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--log-every", type=int, default=50)
    _add_runtime(train_parser)

    eval_parser = sub.add_parser("eval", help="重新评估已保存的检查点")
    eval_parser.add_argument("--checkpoint", default="runs/bert-typed-decisions")
    _add_runtime(eval_parser)

    predict_parser = sub.add_parser("predict", help="对一条 state 输出 Laya 形式的 answers")
    predict_parser.add_argument("--checkpoint", default="runs/bert-typed-decisions")
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--state", default=None, help="直接传入的 state 文本")
    predict_parser.add_argument("--state-file", default=None)
    predict_parser.add_argument("--questions", default=None, help="题目 JSON 文件，对象的键是题目 id")
    predict_parser.add_argument("--demo", action="store_true", help="用官方 test 的一条案例做演示")
    predict_parser.add_argument("--demo-index", type=int, default=0)
    return parser


def _demo(args) -> None:
    from minilaya.data import load_cases

    cases = load_cases("test")
    case = cases[args.demo_index]
    agent = DecisionAgent.load(args.checkpoint, device=args.device)
    predicted = agent.predict(case.state, case.questions)
    gold = agent.predict_gold(case.questions, case.gold)
    print("案例 %s / %s" % (case.workflow, case.id))
    print(format_answers(predicted))
    print("--- 金标 ---")
    print(format_answers(gold))
    print(dump_result(predicted))


def _predict(args) -> None:
    if args.demo:
        _demo(args)
        return
    if not args.state and not args.state_file:
        raise SystemExit("传入 --state、--state-file，或使用 --demo")
    if not args.questions:
        raise SystemExit("传入 --questions 题目 JSON 文件")
    state = Path(args.state_file).read_text(encoding="utf-8") if args.state_file else args.state
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    agent = DecisionAgent.load(args.checkpoint, device=args.device)
    result = agent.predict(state, questions)
    print(format_answers(result))
    print(dump_result(result))


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args)
        return
    if args.command == "eval":
        evaluate_checkpoint(args)
        return
    if args.command == "predict":
        _predict(args)
        return
    raise SystemExit("未知命令")


if __name__ == "__main__":
    main()
