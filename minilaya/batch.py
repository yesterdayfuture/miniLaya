"""把变长决策行补成一个 BERT batch。

序列长度按桶补齐，是为了让 MPS 反复看到同一批形状。每个 batch 一种全新长度时，
Apple GPU 会在第二次前向卡住。选项数仍按本 batch 的最大值补齐，评估时再在 batch 之间对齐。
"""

from __future__ import annotations

from typing import Dict, List, Mapping

import torch


def collate(
    rows: List[Mapping],
    pad_id: int = 0,
    fixed_len: int | None = None,
    bucket: int = 0,
) -> Dict[str, torch.Tensor]:
    """把若干决策补成张量。

    fixed_len 是硬上限，通常等于 max_len。bucket > 0 时不补到上限，
    而是补到不小于本 batch 最长序列的下一个 bucket 倍数，避免短句被拉到 512。
    补出来的选项位置下标写 0，并用 marker_mask=False 标成无效；
    损失和 softmax 都看这张 mask，不会把填充位当真实选项。
    """
    if not rows:
        raise ValueError("空 batch")
    longest = max(len(row["input_ids"]) for row in rows)
    cap = longest if fixed_len is None else fixed_len
    if longest > cap:
        raise ValueError("序列长度 %s 超过上限 %s" % (longest, cap))
    if bucket > 0:
        max_len = ((longest + bucket - 1) // bucket) * bucket
        max_len = min(max_len, cap)
    else:
        max_len = cap
    max_k = max(len(row["marker_pos"]) for row in rows)
    has_target = "target" in rows[0]
    input_ids, attention, marker_pos, marker_mask, qtype, index = [], [], [], [], [], []
    target = []
    for row in rows:
        length = len(row["input_ids"])
        width = len(row["marker_pos"])
        input_ids.append(list(row["input_ids"]) + [pad_id] * (max_len - length))
        attention.append([1] * length + [0] * (max_len - length))
        marker_pos.append(list(row["marker_pos"]) + [0] * (max_k - width))
        marker_mask.append([True] * width + [False] * (max_k - width))
        qtype.append(int(row["qtype"]))
        index.append(int(row["index"]))
        if has_target:
            target.append(list(row["target"]) + [0.0] * (max_k - width))
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "marker_pos": torch.tensor(marker_pos, dtype=torch.long),
        "marker_mask": torch.tensor(marker_mask, dtype=torch.bool),
        "qtype": torch.tensor(qtype, dtype=torch.long),
        "index": torch.tensor(index, dtype=torch.long),
    }
    if has_target:
        batch["target"] = torch.tensor(target, dtype=torch.float)
    return batch
