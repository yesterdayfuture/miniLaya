"""Pack one typed question into the BERT sequence Laya scores at [MASK] markers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from minilaya.schema import State, render_options, serialize_state


def _encode(tokenizer, text: str) -> List[int]:
    """只做分词，不在这里截断。截断策略由 build_sequence 统一决定，避免选项标记被词表上限切掉。"""
    mask = getattr(tokenizer, "mask_token", None)
    # 正文里如果出现字面量 [MASK]，会变成假的打分位置，先换成空格。
    if mask and mask in text:
        text = text.replace(mask, " ")
    # 截断发生在 build_sequence。若留着 tokenizer 自带的 512 上限，长 state 会在切片前就报警。
    limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(limit, int) and limit < 10**6:
        tokenizer.model_max_length = 10**6
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids)


def build_sequence(
    tokenizer,
    state: State,
    question: Mapping[str, Any],
    max_len: int = 512,
    head_max_len: int = 256,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """返回 token id、每个选项的 [MASK] 下标，以及截断统计。

    布局与 Laya 相同：
    [CLS] instructions [SEP] [MASK] option [MASK] option ... [SEP] state [SEP]

    每个选项都必须留下自己的 [MASK]。超长时先缩短说明和选项文本，再截 state 的尾部。
    head_max_len 只限制题干（说明 + 选项），把剩余位置留给 state。
    """
    if max_len < 8:
        raise ValueError("max_len 过小: %s" % max_len)
    options = render_options(question)
    n_options = len(options)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    if None in (cls_id, sep_id, mask_id):
        raise ValueError("tokenizer 需要 cls、sep、mask，BERT 词表满足这一点")

    # 固定开销：CLS、说明后的 SEP、每个选项一个 MASK、选项段后的 SEP。选项正文另算。
    # 末尾的 state SEP 不占 head_max_len，它放在整段序列的最后。
    overhead = 3 + n_options
    if overhead > head_max_len:
        raise ValueError(
            "head_max_len=%s 放不下 %s 个选项，请增大 head_max_len" % (head_max_len, n_options)
        )

    instructions = str(question.get("instructions") or "Answer the question.")
    instruction_ids = _encode(tokenizer, instructions)
    text_budget = head_max_len - overhead
    # 每个选项至少留一个 token 的标签，其余预算再给说明。选项一个都不能删。
    min_option_text = min(text_budget, n_options)
    instruction_cap = max(0, text_budget - min_option_text)
    instruction_ids = instruction_ids[:instruction_cap]
    option_budget = text_budget - len(instruction_ids)
    base, extra = divmod(option_budget, n_options)
    option_caps = [base + (1 if index < extra else 0) for index in range(n_options)]

    ids: List[int] = [cls_id]
    ids.extend(instruction_ids)
    ids.append(sep_id)
    markers: List[int] = []
    option_tokens_used = 0
    for option, cap in zip(options, option_caps):
        markers.append(len(ids))
        ids.append(mask_id)
        piece = _encode(tokenizer, option)[:cap]
        ids.extend(piece)
        option_tokens_used += len(piece)
    ids.append(sep_id)
    if len(ids) > head_max_len:
        raise RuntimeError("题干超出 head_max_len，这是序列构造错误")

    head_len = len(ids)
    state_ids = _encode(tokenizer, serialize_state(state))
    # 留 1 个位置给结尾 SEP。state 放不下时从尾部丢掉，题干和标记保持完整。
    room = max(0, max_len - head_len - 1)
    used_state = state_ids[:room]
    ids = (ids + used_state + [sep_id])[:max_len]
    markers = [marker for marker in markers if marker < len(ids)]
    if len(markers) != n_options:
        raise RuntimeError("选项标记被截断，请增大 max_len 或减小 head_max_len")
    stats = {
        "options": n_options,
        "head_len": head_len,
        "option_tokens": option_tokens_used,
        "state_tokens": len(state_ids),
        "state_tokens_used": len(used_state),
        "state_tokens_dropped": len(state_ids) - len(used_state),
    }
    return ids, markers, stats
