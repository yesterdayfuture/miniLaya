"""训练和推理共用的题目格式。

三种题型与 Laya 对齐：
- choice：在若干互斥选项上给出分布，点估计是概率最大的选项。
- score：在有序等级上给出分布，点估计是等级下标的期望。
- noul：只判断陈述是否成立，点估计是 P(true)，不带 confidence。
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np

# 题型 id 要写进 Embedding，所以固定成 0/1/2，不能随数据出现顺序变化。
QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {index: name for name, index in QTYPES.items()}

State = Union[str, Mapping[str, Any], list]
JsonValue = Union[str, Mapping[str, Any], list]


def parse_jsonish(value: JsonValue) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def serialize_state(state: State) -> str:
    """编码器实际读到的文本。数据集里的 state 已是字符串时原样保留，避免重新 dump 改变基准。"""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(", ", ": "))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def option_keys(question: Mapping[str, Any]) -> List[str]:
    """选项顺序。训练目标、[MASK] 位置和推理概率必须使用同一顺序。

    choice 用 criteria 字典的键顺序。score 的金标按 "0","1",... 编号，
    所以这里返回的字符串也要能对上这些键。noul 固定 false 在前、true 在后，
    这样 P(true) 永远是概率向量的第 2 项。
    """
    qtype = question["type"]
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice 题目需要非空的 criteria 字典")
        return [str(key) for key in criteria.keys()]
    if qtype == "score":
        if isinstance(criteria, dict):
            labels = list(criteria.keys())
        elif isinstance(criteria, list):
            labels = list(range(len(criteria)))
        else:
            raise ValueError("score 题目需要 criteria 列表或字典")
        if not labels:
            raise ValueError("score 题目至少需要一个等级")
        return [str(label) for label in labels]
    if qtype == "noul":
        return ["false", "true"]
    raise ValueError("未知题型 %r，只支持 choice、score、noul" % (qtype,))


def render_options(question: Mapping[str, Any]) -> List[str]:
    """Option strings placed after each [MASK], in option_keys order."""
    qtype = question["type"]
    criteria = question.get("criteria")
    keys = option_keys(question)
    if qtype == "choice":
        rendered = []
        for key in keys:
            value = ""
            for raw, raw_value in criteria.items():
                if str(raw) == key:
                    value = raw_value
                    break
            text = _as_text(value).strip()
            rendered.append(key if not text else "%s: %s" % (key, text))
        return rendered
    if qtype == "score":
        if isinstance(criteria, dict):
            values = []
            for key in criteria.keys():
                values.append(_as_text(criteria[key]).strip() or str(key))
        else:
            values = [_as_text(item).strip() for item in criteria]
        return ["level %d: %s" % (index, text or str(index)) for index, text in enumerate(values)]
    false_text = "no, the statement does not hold"
    true_text = "yes, the statement holds"
    if isinstance(criteria, dict):
        if _as_text(criteria.get("false")).strip():
            false_text = _as_text(criteria.get("false")).strip()
        if _as_text(criteria.get("true")).strip():
            true_text = _as_text(criteria.get("true")).strip()
    return ["false: %s" % false_text, "true: %s" % true_text]


def target_from_gold(question: Mapping[str, Any], gold_question: Mapping[str, Any]) -> List[float]:
    """把一条金标收成与 option_keys 等长、和为 1 的分布。

    金标是教师三次采样的平均，已经是软标签，不是 one-hot。
    缺概率字典时才退回 label 的 one-hot。
    """
    keys = option_keys(question)
    probabilities = gold_question.get("probabilities")
    if isinstance(probabilities, str):
        probabilities = json.loads(probabilities)
    if not isinstance(probabilities, dict):
        probabilities = _one_hot(keys, gold_question.get("label"))
    target = [float(probabilities.get(key, probabilities.get(str(key), 0.0))) for key in keys]
    # score 的 rubric 可能是单词，但金标键是等级序号。对不齐时按数值键排序回退。
    if question["type"] == "score" and sum(target) <= 0 and probabilities:
        ordered = sorted(probabilities, key=lambda item: int(item) if str(item).lstrip("-").isdigit() else 10**9)
        if len(ordered) == len(keys):
            target = [float(probabilities[key]) for key in ordered]
    if any(not math.isfinite(value) or value < 0 for value in target):
        raise ValueError("概率必须是非负有限数，得到 %r" % (target,))
    total = sum(target)
    if total <= 0:
        return [1.0 / len(target)] * len(target)
    return [value / total for value in target]


def _one_hot(keys: Sequence[str], label: Any) -> Dict[str, float]:
    if label is None:
        return {}
    name = str(label).lower() if str(label).lower() in {"true", "false"} else str(label)
    return {key: 1.0 if key == name or key == str(label) else 0.0 for key in keys}


def distribution_answer(question: Mapping[str, Any], probabilities: Sequence[float]) -> Dict[str, Any]:
    """一条决策的 Laya 答案。

    confidence 一律是 max(p)，温度缩放会改变它，但不会改变 argmax。
    noul 故意不写 confidence：Laya 的该题型只暴露 P(true)。
    score 的点估计是期望等级，不是 argmax 等级。
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or len(probs) != len(option_keys(question)):
        raise ValueError("概率长度与选项数不一致")
    total = float(probs.sum())
    if total <= 0:
        probs = np.full(len(probs), 1.0 / len(probs))
    else:
        probs = probs / total
    qtype = question["type"]
    keys = option_keys(question)
    if qtype == "choice":
        chosen = keys[int(np.argmax(probs))]
        return {
            "type": "choice",
            "choice": chosen,
            "probabilities": {key: float(value) for key, value in zip(keys, probs)},
            "confidence": float(np.max(probs)),
        }
    if qtype == "score":
        levels = np.arange(len(probs), dtype=np.float64)
        return {
            "type": "score",
            "score": float(levels @ probs),
            "probabilities": {str(index): float(value) for index, value in enumerate(probs)},
            "confidence": float(np.max(probs)),
        }
    return {
        "type": "noul",
        "noul": float(probs[1]),
        "probabilities": {"false": float(probs[0]), "true": float(probs[1])},
    }


def gold_as_answers(questions: Mapping[str, Any], gold: Mapping[str, Any]) -> Dict[str, Any]:
    answers = {}
    for qid, question in questions.items():
        answers[qid] = distribution_answer(question, target_from_gold(question, gold[qid]))
    return answers
