"""读取 LocalLLaMA/typed-decisions，把每道题变成一条训练样本。

只加载配置 "all"。它已经是四个工作流的并集。再叠加各个工作流配置会把同一案例算两次。
一条数据集行是一个案例，里面有多道题；编码后每道题成为一条 Example。
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from minilaya.schema import (
    QTYPES,
    parse_jsonish,
    target_from_gold,
)
from minilaya.sequence import build_sequence

DATASET_ID = "LocalLLaMA/typed-decisions"
# "all" = agent_trace_observability + customer_service + invoice_processing + security_incidents。
DATASET_CONFIG = "all"


@dataclass
class Case:
    """一个案例：同一段 state 上的一组题目和对应金标。校准切分以案例为单位，避免同案题目分到两侧。"""
    id: str
    workflow: str
    state: Any
    questions: Dict[str, Any]
    gold: Dict[str, Any]


@dataclass
class Example:
    """一道已经编码的题。marker_pos 与 target 等长，顺序都是 option_keys。"""
    case_id: str
    workflow: str
    qid: str
    qtype: str
    input_ids: List[int]
    marker_pos: List[int]
    target: List[float]
    state_dropped: int


def load_cases(split: str) -> List[Case]:
    """读取官方 split。state 保持数据集原值；questions 和 gold 从 JSON 字符串解析出来。"""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    cases: List[Case] = []
    seen = set()
    for row in dataset:
        case_id = str(row["id"])
        if case_id in seen:
            # 配置被重复加载时同一 id 会出现多次，只保留第一条。
            continue
        seen.add(case_id)
        questions = parse_jsonish(row["questions"])
        gold = parse_jsonish(row["gold"])
        if not isinstance(questions, dict) or not isinstance(gold, dict):
            raise ValueError("案例 %s 的 questions 或 gold 不是对象" % case_id)
        missing = [qid for qid in questions if qid not in gold]
        if missing:
            raise ValueError("案例 %s 缺少金标: %s" % (case_id, ", ".join(missing)))
        cases.append(
            Case(
                id=case_id,
                workflow=str(row["workflow"]),
                state=row["state"],
                questions=questions,
                gold=gold,
            )
        )
    if not cases:
        raise ValueError("切分 %s 是空的" % split)
    return cases


def split_calibration(
    cases: Sequence[Case],
    frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Case], List[Case]]:
    """在任何梯度之前，按工作流分层留出整案例。

    每个工作流内部洗牌后取前 frac 做校准，其余做训练。frac > 0 且该组不止一条时至少留 1 条，
    避免小子集把某一工作流的校准集留空。测试集不参与这次切分。
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError("calib_frac 必须在 [0, 1) 内")
    grouped: Dict[str, List[Case]] = defaultdict(list)
    for case in cases:
        grouped[case.workflow].append(case)
    rng = random.Random(seed)
    train: List[Case] = []
    calib: List[Case] = []
    for workflow in sorted(grouped):
        group = list(grouped[workflow])
        rng.shuffle(group)
        holdout = int(len(group) * frac)
        if holdout == 0 and frac > 0 and len(group) > 1:
            holdout = 1
        calib.extend(group[:holdout])
        train.extend(group[holdout:])
    if not train:
        raise ValueError("校准切分后训练集为空")
    return train, calib


def encode_cases(
    cases: Sequence[Case],
    tokenizer,
    max_len: int,
    head_max_len: int,
) -> Tuple[List[Example], Dict[str, int]]:
    """编码一个案例列表。state_truncated 统计的是“state 被截掉至少一个 token”的决策数。"""
    examples: List[Example] = []
    dropped_state = 0
    for case in cases:
        for qid, question in case.questions.items():
            qtype = question["type"]
            if qtype not in QTYPES:
                raise ValueError("案例 %s 题目 %s 的题型 %r 不受支持" % (case.id, qid, qtype))
            input_ids, markers, stats = build_sequence(
                tokenizer,
                case.state,
                question,
                max_len=max_len,
                head_max_len=head_max_len,
            )
            target = target_from_gold(question, case.gold[qid])
            if len(target) != len(markers):
                raise RuntimeError("案例 %s 题目 %s 的选项数和金标长度不一致" % (case.id, qid))
            dropped_state += int(stats["state_tokens_dropped"] > 0)
            examples.append(
                Example(
                    case_id=case.id,
                    workflow=case.workflow,
                    qid=qid,
                    qtype=qtype,
                    input_ids=input_ids,
                    marker_pos=markers,
                    target=target,
                    state_dropped=int(stats["state_tokens_dropped"] > 0),
                )
            )
    stats = {
        "decisions": len(examples),
        "state_truncated": dropped_state,
    }
    return examples, stats


def limit_cases(cases: Sequence[Case], limit: Optional[int]) -> List[Case]:
    if not limit:
        return list(cases)
    if limit < 1:
        raise ValueError("limit 必须是正整数")
    return list(cases)[:limit]
