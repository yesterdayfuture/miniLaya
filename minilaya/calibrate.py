"""在留出集上为每种题型拟合一个标量温度。

温度 T 作用在 logit 上：softmax(logit / T)。T > 1 把分布拉平，T < 1 把分布变尖。
T 必须为正，因此 argmax 不变，准确率不变，变的是 KL、Brier 和 ECE。
测试集不能传进 fit_temperatures，否则温度会偷看最终分数。
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from minilaya.model import masked_log_softmax

MIN_COUNT = 8
TEMP_MIN = 0.5
TEMP_MAX = 5.0
# 46 个点覆盖 [0.5, 5.0]，步长 0.1。网格足够密，不必再做连续优化。


def fit_temperatures(
    logits: torch.Tensor,
    target: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: Sequence[str],
) -> Dict[str, float]:
    """按题型网格搜索使 soft CE 最小的温度。

    某一题型的校准样本少于 MIN_COUNT 时保持 1.0，避免几个样本把温度推到网格端点。
    """
    temperatures = {"choice": 1.0, "score": 1.0, "noul": 1.0}
    if logits.numel() == 0:
        return temperatures
    names = list(qtype)
    grid = np.linspace(TEMP_MIN, TEMP_MAX, 46)
    for name in temperatures:
        index = [i for i, item in enumerate(names) if item == name]
        if len(index) < MIN_COUNT:
            continue
        chosen = logits[index]
        chosen_target = target[index]
        chosen_mask = marker_mask[index]
        best_loss = None
        best_temp = 1.0
        for temperature in grid:
            scaled = chosen / float(temperature)
            log_prob = masked_log_softmax(scaled, chosen_mask)
            loss = -(chosen_target * log_prob * chosen_mask.float()).sum(dim=-1).mean()
            value = float(loss.item())
            if best_loss is None or value < best_loss:
                best_loss = value
                best_temp = float(temperature)
        temperatures[name] = best_temp
    return temperatures


def scale_logits(logits: torch.Tensor, qtype: Sequence[str], temperatures: Mapping[str, float]) -> torch.Tensor:
    """按每行的题型除以对应温度。未知题型用 1。下限 1e-3 防止除零。"""
    factors = [float(temperatures.get(name, 1.0)) for name in qtype]
    scale = logits.new_tensor(factors).unsqueeze(1).clamp_min(1e-3)
    return logits / scale
