"""类型化概率决策的准确率、Brier、KL 和 ECE。

准确率：argmax(预测) == argmax(金标)。并列时 NumPy 取第一个最大值。
KL：KL(金标 ‖ 预测)，单位 nat，再对决策取平均。方向不能反，否则和数据集卡对不上。
Brier：选项上 (p - g)^2 之和，再对决策取平均。
ECE：15 个等宽箱子，置信度是 max(p)。温度改变的就是这个置信度。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

# 数据集卡上 typed-decisions test（2000 条决策）的已发表数字。
# 在 train 上拟合过的行不能和零样本 Jev 比高低。放在这里是为了和本次运行并排阅读。
# 均匀分布的 ECE 对 argmax 并列处理敏感，本次复现不一定等于卡上的 0.169；KL 和 Brier 应当吻合。
PUBLISHED = [
    {"model": "均匀分布", "kind": "参考", "accuracy": 0.308, "kl": 0.444, "brier": 0.238, "ece": 0.169},
    {"model": "先验（忽略输入）", "kind": "参考", "accuracy": 0.470, "kl": 0.347, "brier": 0.189, "ece": 0.088},
    {"model": "教师自一致", "kind": "上限", "accuracy": 0.735, "kl": None, "brier": None, "ece": None},
    {"model": "Jev 1.13.0", "kind": "零样本", "accuracy": 0.727, "kl": 1.442, "brier": 0.148, "ece": 0.144},
    {"model": "ModernBERT-base 分工作流专家", "kind": "在 train 上拟合", "accuracy": 0.646, "kl": 0.223, "brier": 0.119, "ece": 0.179},
    {"model": "Laya typed-decisions", "kind": "在 train 上拟合", "accuracy": 0.766, "kl": None, "brier": None, "ece": None},
]


def _normalized(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        return np.full(len(array), 1.0 / len(array))
    return array / total


def decision_scores(probability: Sequence[float], target: Sequence[float]) -> Dict[str, float]:
    """一条决策的四项分数。概率和金标都会先截断负值再归一化。"""
    pred = _normalized(probability)
    gold = _normalized(target)
    correct = float(int(np.argmax(pred) == np.argmax(gold)))
    brier = float(np.sum((pred - gold) ** 2))
    gold_safe = np.clip(gold, 1e-12, None)
    pred_safe = np.clip(pred, 1e-12, None)
    # 金标为 0 的项不贡献 KL，避免 0 * log(0) 变成 NaN。
    kl = float(np.sum(np.where(gold > 0, gold * np.log(gold_safe / pred_safe), 0.0)))
    return {
        "correct": correct,
        "brier": brier,
        "kl": kl,
        "confidence": float(np.max(pred)),
    }


def ece_score(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """期望校准误差。confidence 用 max(p)，也就是温度会移动的那个量。

    第 0 箱含左端点，其余箱子左开右闭，使 0 和 1 都能落进某一箱。
    """
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == 0:
            selected = (confidence >= low) & (confidence <= high)
        else:
            selected = (confidence > low) & (confidence <= high)
        if selected.any():
            error += float(selected.mean()) * abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
    return float(error)


def summarize(rows: Iterable[Mapping[str, Any]], group_keys: Sequence[str] = ("qtype", "workflow")) -> Dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("没有可评估的决策")

    def pack(group: List[Mapping[str, Any]]) -> Dict[str, Any]:
        correct = np.array([row["correct"] for row in group], dtype=np.float64)
        confidence = np.array([row["confidence"] for row in group], dtype=np.float64)
        return {
            "n": len(group),
            "accuracy": float(correct.mean()),
            "brier": float(np.mean([row["brier"] for row in group])),
            "kl": float(np.mean([row["kl"] for row in group])),
            "ece": ece_score(confidence, correct),
        }

    summary: Dict[str, Any] = pack(records)
    for key in group_keys:
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for row in records:
            grouped.setdefault(str(row[key]), []).append(row)
        summary["by_" + key] = {name: pack(group) for name, group in sorted(grouped.items())}
    return summary


def uniform_probability(size: int) -> List[float]:
    return [1.0 / size] * size


def chance_accuracy(targets: Iterable[Sequence[float]]) -> float:
    """平均 1/K。这才是数据集卡上的均匀准确率。

    对一条平坦向量取 argmax 会永远落到下标 0，那个准确率不是机会水平。
    """
    widths = [len(target) for target in targets]
    if not widths:
        return float("nan")
    return float(np.mean([1.0 / width for width in widths]))
