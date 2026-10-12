"""在 typed-decisions 上微调 BERT，并只在官方 test 上打分。

流程：先在 test 上算均匀分布和先验，确认评分器；再训练；每个 epoch 用未调温度的
校准 soft CE 选权重；最后在校准集上拟合温度，对 test 打一次分。
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from minilaya.batch import collate
from minilaya.calibrate import fit_temperatures, scale_logits
from minilaya.data import (
    Case,
    Example,
    encode_cases,
    limit_cases,
    load_cases,
    split_calibration,
)
from minilaya.metrics import (
    PUBLISHED,
    chance_accuracy,
    decision_scores,
    summarize,
    uniform_probability,
)
from minilaya.model import DecisionBert, soft_ce_loss
from minilaya.predict import DecisionAgent, format_answers, probabilities_from_logits, resolve_device, save_checkpoint
from minilaya.schema import QTYPES, target_from_gold


class EncodedDataset(Dataset):
    """已经编码好的决策。qtype 在这里从字符串收成 Embedding 用的整数。"""
    def __init__(self, examples: Sequence[Example]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.examples[index]
        return {
            "input_ids": item.input_ids,
            "marker_pos": item.marker_pos,
            "qtype": QTYPES[item.qtype],
            "target": item.target,
            "index": index,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: DecisionBert, encoder_lr: float, head_lr: float, weight_decay: float):
    """编码器与决策头用不同学习率。一维参数（bias、LayerNorm）不做权重衰减。"""
    groups = {
        (False, False): {"params": [], "lr": encoder_lr, "weight_decay": weight_decay},
        (False, True): {"params": [], "lr": encoder_lr, "weight_decay": 0.0},
        (True, False): {"params": [], "lr": head_lr, "weight_decay": weight_decay},
        (True, True): {"params": [], "lr": head_lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = not name.startswith("encoder.")
        groups[(is_head, param.ndim < 2)]["params"].append(param)
    return torch.optim.AdamW([group for group in groups.values() if group["params"]])


def pack_by_length(examples: Sequence[Example], batch_size: int, seed: int) -> List[Example]:
    """按长度打包，使一个 batch 落在同一个长度桶里，短句不会被 batch 里的长句拉长。

    DataLoader 之后必须 shuffle=False，否则打包会被打散。这里先洗牌的是 batch，不是单条样本。
    """
    order = sorted(range(len(examples)), key=lambda i: (len(examples[i].input_ids), i))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    random.Random(seed).shuffle(batches)
    return [examples[i] for batch in batches for i in batch]


def _loader(
    examples: Sequence[Example],
    batch_size: int,
    pad_id: int,
    shuffle: bool,
    seed: int,
    fixed_len: int,
    bucket: int = 64,
) -> DataLoader:
    """shuffle=True 表示按长度打包后再打乱 batch 顺序。评估传 False，保持与 examples 相同的行序。"""
    rows = pack_by_length(examples, batch_size, seed) if shuffle else examples
    return DataLoader(
        EncodedDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_id=pad_id, fixed_len=fixed_len, bucket=bucket),
    )


def collect_output(
    model: DecisionBert,
    examples: Sequence[Example],
    device: torch.device,
    batch_size: int,
    pad_id: int,
    fixed_len: Optional[int] = None,
):
    """按 examples 的顺序收集 logit。调用方靠这个顺序把 logit 和金标对齐，所以这里不能洗牌。"""
    model.eval()
    length = fixed_len or max(len(item.input_ids) for item in examples)
    loader = _loader(examples, batch_size, pad_id, shuffle=False, seed=0, fixed_len=length)
    logits_parts = []
    target_parts = []
    mask_parts = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            logits_parts.append(logits.float().cpu())
            target_parts.append(batch["target"])
            mask_parts.append(batch["marker_mask"])
    return _cat_marker(logits_parts, 0.0), _cat_marker(target_parts, 0.0), _cat_marker(mask_parts, False)


def _cat_marker(parts: Sequence[torch.Tensor], pad_value) -> torch.Tensor:
    """不同 batch 的选项数不同，直接 cat 会在第 2 维对不上。先补到本批评估里的最大选项数。

    补出来的 logit 填 0、目标填 0、mask 填 False。后续损失和指标都用真实选项宽度，
    这些填充位不参与 softmax。
    """
    if not parts:
        raise RuntimeError("评估没有产生输出")
    width = max(int(part.size(1)) for part in parts)
    padded = []
    for part in parts:
        gap = width - int(part.size(1))
        if gap == 0:
            padded.append(part)
            continue
        fill = part.new_full((part.size(0), gap), pad_value)
        padded.append(torch.cat((part, fill), dim=1))
    return torch.cat(padded, dim=0)


def mean_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float(soft_ce_loss(logits, target, mask).item())


def rows_from_logits(
    examples: Sequence[Example],
    logits: torch.Tensor,
    temperatures: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """把 logit 收成指标行。temperatures 为空时相当于 T=1。只取每行前 len(target) 个 logit。"""
    names = [item.qtype for item in examples]
    scaled = logits if not temperatures else scale_logits(logits, names, temperatures)
    array = scaled.float().cpu().numpy()
    rows = []
    for index, example in enumerate(examples):
        width = len(example.target)
        probability = probabilities_from_logits(array[index], width)
        scores = decision_scores(probability, example.target)
        answer = None
        rows.append({
            "id": example.case_id,
            "workflow": example.workflow,
            "qid": example.qid,
            "qtype": example.qtype,
            "probability": probability,
            "target": list(example.target),
            "correct": scores["correct"],
            "brier": scores["brier"],
            "kl": scores["kl"],
            "confidence": scores["confidence"],
            "answer": answer,
        })
    return rows


def baseline_rows(cases: Sequence[Case], probability_of) -> List[Dict[str, Any]]:
    rows = []
    for case in cases:
        for qid, question in case.questions.items():
            target = target_from_gold(question, case.gold[qid])
            probability = probability_of(case, qid, target)
            scores = decision_scores(probability, target)
            rows.append({
                "workflow": case.workflow,
                "qid": qid,
                "qtype": question["type"],
                **scores,
            })
    return rows


def prior_probabilities(train_cases: Sequence[Case]):
    """忽略输入的基线：同一工作流、同一题目 id 的训练软标签取平均。

    校准集来自官方 train，也算进这个平均。test 上没有见过的键退回均匀分布。
    """
    totals: Dict[tuple, np.ndarray] = {}
    counts: Dict[tuple, int] = defaultdict(int)
    for case in train_cases:
        for qid, question in case.questions.items():
            target = np.asarray(target_from_gold(question, case.gold[qid]), dtype=np.float64)
            key = (case.workflow, qid)
            totals[key] = target if key not in totals else totals[key] + target
            counts[key] += 1
    means = {key: totals[key] / counts[key] for key in totals}

    def probability_of(case: Case, qid: str, target: Sequence[float]) -> List[float]:
        mean = means.get((case.workflow, qid))
        if mean is None or len(mean) != len(target):
            return uniform_probability(len(target))
        return mean.tolist()

    return probability_of


def render_table(title: str, rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [title, "%-32s %8s %8s %8s %8s %6s" % ("模型", "准确率", "KL", "Brier", "ECE", "n")]
    for row in rows:
        def cell(value: Any) -> str:
            if value is None:
                return "%8s" % "-"
            return "%8.3f" % float(value)

        lines.append(
            "%-32s %s %s %s %s %6s"
            % (
                row["model"],
                cell(row.get("accuracy")),
                cell(row.get("kl")),
                cell(row.get("brier")),
                cell(row.get("ece")),
                "-" if row.get("n") is None else str(row["n"]),
            )
        )
    return "\n".join(lines)


def group_lines(summary: Mapping[str, Any], key: str) -> List[str]:
    lines = []
    block = summary.get("by_" + key) or {}
    for name, metrics in block.items():
        lines.append(
            "  %-28s 准确率 %.3f  KL %.3f  Brier %.3f  ECE %.3f  n=%d"
            % (name, metrics["accuracy"], metrics["kl"], metrics["brier"], metrics["ece"], metrics["n"])
        )
    return lines


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.cls_token
    return tokenizer


def build_encoder_model(model_name: str, head_layers: int, dropout: float) -> DecisionBert:
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(model_name)
    return DecisionBert(encoder, head_layers=head_layers, dropout=dropout)


def train(args) -> Dict[str, Any]:
    """完整训练。返回写入 benchmark.json 的那份字典。"""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("加载 %s [%s]" % ("LocalLLaMA/typed-decisions", "all"))
    train_cases = load_cases("train")
    test_cases = load_cases("test")
    if args.limit:
        train_cases = limit_cases(train_cases, args.limit)
        test_cases = limit_cases(test_cases, args.limit)

    # 将训练集先按 工作流 进行分组，然后每组 按照frac进行划分。分为训练集和校准集。
    fit_cases, calib_cases = split_calibration(train_cases, frac=args.calib_frac, seed=args.seed)
    # 基线在编码和训练之前就算完。KL / Brier 若和数据集卡差一截，说明评分器有问题，不必等训练结束。
    print(
        "案例  训练 %d  校准 %d  测试 %d  设备 %s"
        % (len(fit_cases), len(calib_cases), len(test_cases), device)
    )

    uniform_summary = summarize(
        baseline_rows(test_cases, lambda _case, _qid, target: uniform_probability(len(target))),
        group_keys=(),
    )
    chance = chance_accuracy(
        target_from_gold(question, case.gold[qid])
        for case in test_cases
        for qid, question in case.questions.items()
    )
    prior_summary = summarize(baseline_rows(test_cases, prior_probabilities(train_cases)), group_keys=())
    print(
        "指标自检  均匀分布 KL %.3f（数据集卡 0.444）  Brier %.3f（数据集卡 0.238）  机会准确率 %.3f（数据集卡 0.308）"
        % (uniform_summary["kl"], uniform_summary["brier"], chance)
    )
    print(
        "指标自检  训练集先验准确率 %.3f（数据集卡 0.470）  KL %.3f  Brier %.3f"
        % (prior_summary["accuracy"], prior_summary["kl"], prior_summary["brier"])
    )

    tokenizer = load_tokenizer(args.model)
    print("编码决策，max_len=%d head_max_len=%d" % (args.max_len, args.head_max_len))
    fit_examples, fit_stats = encode_cases(fit_cases, tokenizer, args.max_len, args.head_max_len)
    calib_examples, calib_stats = encode_cases(calib_cases, tokenizer, args.max_len, args.head_max_len)
    test_examples, test_stats = encode_cases(test_cases, tokenizer, args.max_len, args.head_max_len)
    for name, stats in ("训练", fit_stats), ("校准", calib_stats), ("测试", test_stats):
        print("  %s决策 %d，state 被截断 %d" % (name, stats["decisions"], stats["state_truncated"]))

    model = build_encoder_model(args.model, args.head_layers, args.dropout)
    if getattr(model.encoder, "config", None) is not None:
        model.encoder.config.use_cache = False
    if device.type == "mps":
        # batch 8、序列 512 时，不 checkpoint 会在 MPS 统一内存水位上卡住而不是报 OOM。
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoder_params = sum(param.numel() for name, param in model.named_parameters() if name.startswith("encoder."))
    head_params = sum(param.numel() for name, param in model.named_parameters() if not name.startswith("encoder."))
    print("参数  编码器 %.1fM  决策头 %.1fM" % (encoder_params / 1e6, head_params / 1e6))
    model.to(device)
    pad_id = tokenizer.pad_token_id or 0
    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    steps_per_epoch = max(1, (len(fit_examples) + args.batch_size - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup))
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    best_loss = None
    best_state = None
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        loader = _loader(fit_examples, args.batch_size, pad_id, shuffle=True, seed=args.seed + epoch, fixed_len=args.max_len)
        running = 0.0
        seen = 0
        for step, batch in enumerate(loader, start=1):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
            loss = soft_ce_loss(logits, batch["target"].to(device), batch["marker_mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * batch["input_ids"].size(0)
            seen += batch["input_ids"].size(0)
            if device.type == "mps":
                # 同步后再打日志，避免打印时 GPU 还停在上一 step。
                torch.mps.synchronize()
            if step == 1 or step % args.log_every == 0 or step == steps_per_epoch:
                print(
                    "  epoch %d/%d  step %d/%d  loss %.4f  len %d"
                    % (epoch + 1, args.epochs, step, steps_per_epoch, running / seen, batch["input_ids"].size(1)),
                    flush=True,
                )
        calib_logits, calib_target, calib_mask = collect_output(
            model, calib_examples, device, args.batch_size, pad_id, fixed_len=args.max_len
        )
        calib_loss = mean_loss(calib_logits, calib_target, calib_mask)
        # 选模用未调温度的校准损失。温度要等最佳权重量好后再拟合，不能每个 epoch 都在同一批 logit 上挑 T。
        history.append({"epoch": epoch + 1, "train_loss": running / max(seen, 1), "calib_loss": calib_loss})
        print("epoch %d  训练损失 %.4f  校准损失 %.4f" % (epoch + 1, history[-1]["train_loss"], calib_loss))
        if best_loss is None or calib_loss < best_loss:
            best_loss = calib_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            # 中途崩溃时还能从这里恢复权重。正式检查点仍是训练结束后的 safetensors。
            torch.save(best_state, output / "best_state.pt")

    if best_state is None:
        raise RuntimeError("训练没有产生权重")
    model.load_state_dict(best_state)
    # 用最佳权重重新跑校准集，再拟合温度。test 的 logit 只在这一步之后计算一次。
    calib_logits, calib_target, calib_mask = collect_output(
        model, calib_examples, device, args.batch_size, pad_id, fixed_len=args.max_len
    )
    temperatures = fit_temperatures(calib_logits, calib_target, calib_mask, [item.qtype for item in calib_examples])
    print("校准温度  " + "  ".join("%s=%.3f" % (name, temperatures[name]) for name in ("choice", "score", "noul")))

    raw_logits, _target, _mask = collect_output(
        model, test_examples, device, args.batch_size, pad_id, fixed_len=args.max_len
    )
    raw_rows = rows_from_logits(test_examples, raw_logits)
    calibrated_rows = rows_from_logits(test_examples, raw_logits, temperatures)
    raw_summary = summarize(raw_rows)
    summary = summarize(calibrated_rows)
    print(render_metrics(summary, raw_summary, uniform_summary, prior_summary, chance, len(test_examples)))
    print("按题型")
    print("\n".join(group_lines(summary, "qtype")))
    print("按工作流")
    print("\n".join(group_lines(summary, "workflow")))

    config = {
        "model_name": args.model,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "head_layers": args.head_layers,
        "dropout": args.dropout,
        "temperatures": temperatures,
        "dataset": "LocalLLaMA/typed-decisions",
        "dataset_config": "all",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_calib_loss": best_loss,
    }
    save_checkpoint(output, model, tokenizer, config)
    benchmark = {
        "model_name": args.model,
        "dataset": "LocalLLaMA/typed-decisions",
        "split": "test",
        "n_cases_train": len(fit_cases),
        "n_cases_calibration": len(calib_cases),
        "n_cases_test": len(test_cases),
        "n_decisions_test": len(test_examples),
        "temperatures": temperatures,
        "history": history,
        "test_calibrated": summary,
        "test_raw": raw_summary,
        "baseline_uniform": {"chance_accuracy": chance, **{key: uniform_summary[key] for key in ("kl", "brier", "ece", "n")}},
        "baseline_prior": {key: prior_summary[key] for key in ("accuracy", "kl", "brier", "ece", "n")},
        "published": PUBLISHED,
        "seconds": time.time() - started,
        "note": "模型在 train 上微调，校准温度来自 train 内部留出集。测试集只用于最终打分。准确率与零样本 Jev 不可直接比高低。",
    }
    with open(output / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, ensure_ascii=False, indent=2)
    with open(output / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in calibrated_rows:
            handle.write(json.dumps({key: row[key] for key in ("id", "workflow", "qid", "qtype", "probability", "target", "correct")}, ensure_ascii=False) + "\n")

    demo = test_cases[0]
    agent = DecisionAgent.load(output, device=args.device)
    predicted = agent.predict(demo.state, demo.questions)
    gold = agent.predict_gold(demo.questions, demo.gold)
    print("\n示例 %s / %s" % (demo.workflow, demo.id))
    print("预测")
    print(format_answers(predicted))
    print("金标")
    print(format_answers(gold))
    with open(output / "demo_predict.json", "w", encoding="utf-8") as handle:
        json.dump({"id": demo.id, "workflow": demo.workflow, "state": demo.state, "prediction": predicted, "gold": gold}, handle, ensure_ascii=False, indent=2)
    print("检查点 %s" % output)
    return benchmark


def render_metrics(summary, raw_summary, uniform_summary, prior_summary, chance, n_test) -> str:
    rows = [
        {
            "model": "本次 BERT（校准后）",
            "accuracy": summary["accuracy"],
            "kl": summary["kl"],
            "brier": summary["brier"],
            "ece": summary["ece"],
            "n": n_test,
        },
        {
            "model": "本次 BERT（温度=1）",
            "accuracy": raw_summary["accuracy"],
            "kl": raw_summary["kl"],
            "brier": raw_summary["brier"],
            "ece": raw_summary["ece"],
            "n": n_test,
        },
        {
            "model": "本次均匀分布",
            "accuracy": chance,
            "kl": uniform_summary["kl"],
            "brier": uniform_summary["brier"],
            "ece": uniform_summary["ece"],
            "n": n_test,
        },
        {
            "model": "本次先验",
            "accuracy": prior_summary["accuracy"],
            "kl": prior_summary["kl"],
            "brier": prior_summary["brier"],
            "ece": prior_summary["ece"],
            "n": n_test,
        },
    ]
    for item in PUBLISHED:
        rows.append({
            "model": item["model"] + " / " + item["kind"],
            "accuracy": item["accuracy"],
            "kl": item["kl"],
            "brier": item["brier"],
            "ece": item["ece"],
            "n": None,
        })
    return render_table("typed-decisions 测试集", rows)


def evaluate_checkpoint(args) -> Dict[str, Any]:
    """用检查点里保存的温度，在官方 test 上重打分。不再训练，也不重新拟合温度。"""
    from minilaya.predict import DecisionAgent

    agent = DecisionAgent.load(args.checkpoint, device=args.device)
    test_cases = load_cases("test")
    if args.limit:
        test_cases = limit_cases(test_cases, args.limit)
    examples, stats = encode_cases(
        test_cases,
        agent.tokenizer,
        int(agent.config["max_len"]),
        int(agent.config["head_max_len"]),
    )
    print("测试决策 %d，state 被截断 %d" % (stats["decisions"], stats["state_truncated"]))
    logits, _target, _mask = collect_output(
        agent.model,
        examples,
        agent.device,
        args.batch_size,
        agent.tokenizer.pad_token_id or 0,
        fixed_len=int(agent.config["max_len"]),
    )
    rows = rows_from_logits(examples, logits, agent.config.get("temperatures") or {})
    summary = summarize(rows)
    print(
        "准确率 %.3f  KL %.3f  Brier %.3f  ECE %.3f  n=%d"
        % (summary["accuracy"], summary["kl"], summary["brier"], summary["ece"], summary["n"])
    )
    print("按题型")
    print("\n".join(group_lines(summary, "qtype")))
    print("按工作流")
    print("\n".join(group_lines(summary, "workflow")))
    return summary
