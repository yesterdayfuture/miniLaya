"""加载微调检查点，返回 Laya 形式的类型化答案。

检查点目录：
- model.safetensors：整网权重，磁盘上是 fp16，加载时转回 fp32
- encoder/：BERT 结构配置，不含预训练词表权重
- tokenizer/：与训练时相同的分词器
- config.json：长度、头层数，以及每种题型的温度
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from minilaya.batch import collate
from minilaya.calibrate import scale_logits
from minilaya.model import DecisionBert
from minilaya.schema import QTYPES, State, distribution_answer, gold_as_answers
from minilaya.sequence import build_sequence


def resolve_device(name: str = "auto") -> torch.device:
    """auto 的优先级是 CUDA、MPS、CPU。调用方传入具体名字时不再探测。"""
    if name not in (None, "auto"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def probabilities_from_logits(logits: np.ndarray, width: int) -> List[float]:
    """对前 width 个 logit 做数值稳定的 softmax。width 是真实选项数，后面的填充位丢掉。"""
    selected = np.asarray(logits[:width], dtype=np.float64)
    selected = selected - np.max(selected)
    exp = np.exp(selected)
    total = float(exp.sum())
    if total <= 0:
        return [1.0 / width] * width
    return (exp / total).tolist()


class DecisionAgent:
    """一个决策模型。predict 的返回值与 Laya 的 answers 对象对齐。"""

    def __init__(self, model: DecisionBert, tokenizer, config: Mapping[str, Any], device: torch.device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = dict(config)
        self.device = device

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "auto") -> "DecisionAgent":
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        root = Path(path)
        with open(root / "config.json", encoding="utf-8") as handle:
            config = json.load(handle)
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(root / "encoder"))
        model = DecisionBert(encoder, head_layers=int(config.get("head_layers", 1)), dropout=float(config.get("dropout", 0.1)))
        # 磁盘上是 fp16，计算用 fp32。strict=True 保证决策头和编码器的键都在。
        state = {key: value.float() for key, value in load_file(root / "model.safetensors").items()}
        model.load_state_dict(state, strict=True)
        return cls(model, tokenizer, config, resolve_device(device))

    def predict(self, state: State, questions: Mapping[str, Any]) -> Dict[str, Any]:
        """对同一段 state 上的全部题目做一次 batch 前向，再按题型温度解码。"""
        if isinstance(questions, str):
            questions = json.loads(questions)
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions 必须是非空的题目字典")
        rows = []
        order = []
        for index, (qid, question) in enumerate(questions.items()):
            if question.get("type") not in QTYPES:
                raise ValueError("题目 %s 的 type 必须是 choice、score 或 noul" % qid)
            input_ids, markers, _stats = build_sequence(
                self.tokenizer,
                state,
                question,
                max_len=int(self.config["max_len"]),
                head_max_len=int(self.config["head_max_len"]),
            )
            rows.append({
                "input_ids": input_ids,
                "marker_pos": markers,
                "qtype": QTYPES[question["type"]],
                "index": index,
            })
            order.append((qid, question, len(markers)))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        batch = collate(rows, pad_id=pad_id)
        qtype_names = [question["type"] for _qid, question, _width in order]
        with torch.inference_mode():
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["marker_pos"].to(self.device),
                batch["marker_mask"].to(self.device),
                batch["qtype"].to(self.device),
            )
            logits = scale_logits(logits, qtype_names, self.config.get("temperatures") or {})
        # 温度在 logit 上，softmax 在解码时才做，这样 choice / score / noul 共用同一套概率。
        logits = logits.float().cpu().numpy()
        answers = {}
        for row, (qid, question, width) in zip(logits, order):
            answers[qid] = distribution_answer(question, probabilities_from_logits(row, width))
        return {
            "model": self.config.get("model_name", "bert"),
            "answers": answers,
        }

    def predict_gold(self, questions: Mapping[str, Any], gold: Mapping[str, Any]) -> Dict[str, Any]:
        """用和预测相同的解码器把金标变成答案，方便并排对比，不再走模型。"""
        return {"model": "gold", "answers": gold_as_answers(questions, gold)}


def save_checkpoint(path: Union[str, Path], model: DecisionBert, tokenizer, config: Mapping[str, Any]) -> None:
    """写出可被 DecisionAgent.load 读回的目录。权重以 fp16 保存，减小检查点体积。"""
    from safetensors.torch import save_file

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(root / "encoder")
    tokenizer.save_pretrained(root / "tokenizer")
    tensors = {key: value.detach().cpu().half().contiguous() for key, value in model.state_dict().items()}
    save_file(tensors, root / "model.safetensors")
    with open(root / "config.json", "w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, ensure_ascii=False, indent=2)


def format_answers(result: Mapping[str, Any]) -> str:
    """终端上一行一道题。noul 不打印 confidence，因为答案对象里没有这个字段。"""
    lines = []
    for qid, answer in result["answers"].items():
        kind = answer["type"]
        if kind == "choice":
            lines.append("%s\tchoice\t%s\tconfidence=%.3f" % (qid, answer["choice"], answer["confidence"]))
        elif kind == "score":
            lines.append("%s\tscore\t%.3f\tconfidence=%.3f" % (qid, answer["score"], answer["confidence"]))
        else:
            lines.append("%s\tnoul\t%.3f" % (qid, answer["noul"]))
    return "\n".join(lines)


def dump_result(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)
