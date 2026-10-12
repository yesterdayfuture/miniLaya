"""核心约定的回归测试：序列里每个选项一个 [MASK]、Laya 答案字段、金标归一化、指标，以及小 BERT 能拟合软标签。"""
import torch
from transformers import BertConfig, BertModel

from minilaya.batch import collate
from minilaya.metrics import chance_accuracy, decision_scores, ece_score, summarize
from minilaya.model import DecisionBert, soft_ce_loss
from minilaya.predict import DecisionAgent
from minilaya.schema import distribution_answer, gold_as_answers, target_from_gold
from minilaya.sequence import build_sequence


class StubTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    mask_token = "[MASK]"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        pieces = text.split()
        return {"input_ids": [10 + (index % 40) for index, _word in enumerate(pieces)] or [10]}


def test_sequence_keeps_one_mask_per_option():
    question = {
        "type": "choice",
        "instructions": "Which team handles this",
        "criteria": {"billing": "invoices and refunds", "technical": "bugs and outages"},
    }
    ids, markers, stats = build_sequence(StubTokenizer(), "please refund the duplicate charge", question, max_len=64, head_max_len=40)
    assert stats["options"] == 2
    assert len(markers) == 2
    assert ids[0] == 1
    assert ids[-1] == 2
    assert ids[markers[0]] == 3
    assert ids[markers[1]] == 3
    assert stats["state_tokens_dropped"] == 0


def test_answers_match_laya_fields():
    choice = {
        "type": "choice",
        "instructions": "route",
        "criteria": {"billing": "money", "other": "else"},
    }
    score = {"type": "score", "instructions": "urgency", "criteria": ["low", "mid", "high"]}
    noul = {"type": "noul", "instructions": "refund?"}
    choice_answer = distribution_answer(choice, [0.2, 0.8])
    score_answer = distribution_answer(score, [0.2, 0.3, 0.5])
    noul_answer = distribution_answer(noul, [0.25, 0.75])
    assert choice_answer["choice"] == "other"
    assert set(choice_answer) == {"type", "choice", "probabilities", "confidence"}
    assert abs(score_answer["score"] - 1.3) < 1e-9
    assert "confidence" in score_answer
    assert set(noul_answer) == {"type", "noul", "probabilities"}
    assert abs(noul_answer["noul"] - 0.75) < 1e-9

    gold = gold_as_answers(
        {"urgent": noul},
        {"urgent": {"type": "noul", "probabilities": {"false": 0.2, "true": 0.8}}},
    )
    assert abs(gold["urgent"]["noul"] - 0.8) < 1e-9


def test_target_normalizes_and_score_index_fallback():
    question = {"type": "score", "instructions": "risk", "criteria": ["none", "low", "high"]}
    target = target_from_gold(question, {"probabilities": {"0": 1, "1": 1, "2": 2}})
    assert abs(sum(target) - 1) < 1e-9
    assert abs(target[2] - 0.5) < 1e-9


def test_metrics_of_a_perfect_and_uniform_prediction():
    perfect = decision_scores([0.0, 1.0], [0.0, 1.0])
    assert perfect["correct"] == 1
    assert perfect["brier"] == 0
    assert perfect["kl"] == 0
    uniform = decision_scores([0.5, 0.5], [1.0, 0.0])
    assert abs(uniform["brier"] - 0.5) < 1e-9
    assert abs(chance_accuracy([[1, 0, 0], [1, 0]]) - ((1 / 3 + 0.5) / 2)) < 1e-9
    summary = summarize([
        {"correct": 1, "brier": 0.0, "kl": 0.0, "confidence": 0.9, "qtype": "choice", "workflow": "a"},
        {"correct": 0, "brier": 0.5, "kl": 0.2, "confidence": 0.9, "qtype": "choice", "workflow": "a"},
    ])
    assert summary["accuracy"] == 0.5
    assert abs(ece_score(summary and __import__("numpy").array([0.9, 0.9]), __import__("numpy").array([1.0, 0.0])) - 0.4) < 1e-9


def test_tiny_bert_learns_a_soft_target_and_emits_typed_answers():
    config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    model = DecisionBert(BertModel(config), head_layers=1, dropout=0.0)
    ids = [1, 5, 6, 2, 3, 7, 3, 8, 2, 9, 10, 2]
    row = {"input_ids": ids, "marker_pos": [4, 7], "qtype": 0, "target": [0.05, 0.95], "index": 0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    first = last = None
    for step in range(50):
        batch = collate([row])
        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["marker_pos"],
            batch["marker_mask"],
            batch["qtype"],
        )
        loss = soft_ce_loss(logits, batch["target"], batch["marker_mask"])
        if step == 0:
            first = float(loss)
        last = float(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert last < first * 0.5

    agent = DecisionAgent(
        model,
        StubTokenizer(),
        {
            "max_len": 64,
            "head_max_len": 40,
            "model_name": "tiny",
            "temperatures": {"choice": 1.0, "score": 1.0, "noul": 1.0},
        },
        torch.device("cpu"),
    )
    result = agent.predict(
        "please refund the duplicate",
        {
            "department": {"type": "choice", "instructions": "Which team", "criteria": {"billing": "money", "other": "else"}},
            "urgent": {"type": "noul", "instructions": "Is it a refund"},
            "level": {"type": "score", "instructions": "How urgent", "criteria": ["low", "high"]},
        },
    )
    assert set(result["answers"]) == {"department", "urgent", "level"}
    assert abs(sum(result["answers"]["department"]["probabilities"].values()) - 1) < 1e-5
    assert 0.0 <= result["answers"]["urgent"]["noul"] <= 1.0
    assert result["model"] == "tiny"
