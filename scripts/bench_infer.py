"""测量检查点的推理时间。

加载、第一次调用、预热后的重复调用分开计时。CUDA / MPS 在每次计时前后同步，
避免把还在队列里的核函数算进下一次。默认用官方 test 的一条案例，
分别测 1 道题和该案例的全部题目。

    python scripts/bench_infer.py
    python scripts/bench_infer.py --checkpoint runs/bert-typed-decisions --repeat 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from minilaya.data import load_cases
from minilaya.predict import DecisionAgent
from minilaya.sequence import build_sequence


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def take_questions(questions: dict, count: int) -> dict:
    if count < 1:
        raise SystemExit("--count 必须是正整数")
    chosen = list(questions.items())[:count]
    if len(chosen) < count:
        raise SystemExit("该案例只有 %d 道题，少于 --count %d" % (len(chosen), count))
    return dict(chosen)


def sequence_lengths(agent: DecisionAgent, state, questions: dict) -> list[int]:
    lengths = []
    for question in questions.values():
        ids, _markers, _stats = build_sequence(
            agent.tokenizer,
            state,
            question,
            max_len=int(agent.config["max_len"]),
            head_max_len=int(agent.config["head_max_len"]),
        )
        lengths.append(len(ids))
    return lengths


def time_calls(agent: DecisionAgent, state, questions: dict, repeat: int) -> list[float]:
    samples = []
    for _ in range(repeat):
        sync(agent.device)
        started = time.perf_counter()
        agent.predict(state, questions)
        sync(agent.device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def report(label: str, samples: list[float], n_questions: int) -> None:
    p50 = percentile(samples, 0.50)
    p95 = percentile(samples, 0.95)
    mean = sum(samples) / len(samples)
    print(
        "  %-12s  n=%d  题=%d  p50 %7.1f ms  p95 %7.1f ms  均值 %7.1f ms  每题 %6.1f ms"
        % (label, len(samples), n_questions, p50, p95, mean, p50 / n_questions)
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="测量 minilaya 检查点的推理时间")
    parser.add_argument("--checkpoint", default="runs/bert-typed-decisions")
    parser.add_argument("--device", default="auto", help="auto、cpu、mps 或 cuda")
    parser.add_argument("--demo-index", type=int, default=0, help="官方 test 案例下标")
    parser.add_argument("--count", type=int, default=0, help="只测前 N 道题；0 表示先测 1 道，再测全部")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeat < 1:
        raise SystemExit("--warmup 和 --repeat 必须是正整数")

    cases = load_cases("test")
    case = cases[args.demo_index]
    counts = [args.count] if args.count else [1, len(case.questions)]

    started = time.perf_counter()
    agent = DecisionAgent.load(args.checkpoint, device=args.device)
    sync(agent.device)
    load_ms = (time.perf_counter() - started) * 1000.0
    print("设备 %s  检查点 %s" % (agent.device, args.checkpoint))
    print("案例 %s / %s  共 %d 道题" % (case.workflow, case.id, len(case.questions)))
    print("加载 %.0f ms" % load_ms)

    # 第一次调用含设备上的内核编译，单独列出，不放进预热后的分位数。
    full = take_questions(case.questions, counts[-1])
    sync(agent.device)
    started = time.perf_counter()
    agent.predict(case.state, full)
    sync(agent.device)
    print("第一次调用 %d 道题  %.1f ms" % (len(full), (time.perf_counter() - started) * 1000.0))

    for count in counts:
        questions = take_questions(case.questions, count)
        lengths = sequence_lengths(agent, case.state, questions)
        print(
            "序列长度 %s  最短 %d  最长 %d"
            % (" ".join(str(item) for item in lengths), min(lengths), max(lengths))
        )
        for _ in range(args.warmup):
            agent.predict(case.state, questions)
        sync(agent.device)
        samples = time_calls(agent, case.state, questions, args.repeat)
        report("预热后", samples, count)


if __name__ == "__main__":
    main()
