"""单轮决策上的 RLCD。

奖励与 Laya 的 proper_reward 一致：对数分 + 0.5 × 球面分；score 题再减 1.0 × 排名概率分。
动作是报出的分布。在选项 logit 上加零均值高斯噪声，得到一组扰动报告，
用 REINFORCE 把高于组内均值的报告拉向当前 logit。

typed-decisions 没有多轮轨迹，也没有 act / escalate 标签，所以这里不含 TD(λ) 和动作头。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from minilaya.schema import QTYPES


def masked_probs(logits: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    """只在真实选项上做 softmax。填充位的概率是 0。"""
    # 填充位若留着 0，softmax 会分走概率。-1e4 把它们压到接近 0，又不会变成 NaN。
    return torch.softmax(logits.masked_fill(~marker_mask, -1e4), dim=-1)


def proper_reward(
    probability: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    marker_mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """严格恰当评分。probability 可以是 [N, K] 或 [G, N, K]，返回去掉最后一维后的奖励。

    对数项下限 -9.21，约等于 log(1e-4)，避免某一个近零概率把整批梯度拉爆。
    球面分是 target·q / ||q||。排名概率分只加在 score 上，比较的是累积分布。

    三项都是严格恰当的：期望奖励唯一地在「报告等于金标」时最大。
    choice / noul 的选项没有顺序，所以不加排名概率分。
    """
    mask = marker_mask.to(dtype=probability.dtype)
    # 填充位概率清零，后面的范数和累积和都不会算进它们。
    reported = probability * mask
    # clamp 两次：先避免 log(0)，再把 log(1e-12) 抬到 -9.21。金标为 0 的项乘上去仍是 0。
    log_q = torch.log(reported.clamp_min(1e-12)).clamp_min(log_floor)
    # 软标签时这是 sum_k g_k log q_k，和 soft CE 是同一项，只是符号相反（这里是奖励，越大越好）。
    log_score = (target * log_q).sum(dim=-1)
    # 球面分有界。全对的 one-hot 得 1，均匀分布得 1/sqrt(K)。分母夹住，防止全零向量。
    spherical = (target * reported).sum(dim=-1) / reported.norm(dim=-1).clamp_min(1e-9)
    reward = log_score + w_sph * spherical
    # is_score 是 0/1。choice 和 noul 乘上去是 0，排名项消失。
    is_score = (qtype == QTYPES["score"]).to(device=probability.device, dtype=probability.dtype)
    # 至少按 2 档算，避免 K=1 时除以 0。真实 score 题的等级数都大于 1。
    width = marker_mask.sum(dim=-1).clamp(min=2).to(dtype=probability.dtype)
    # 累积分布：第 k 位是「等级 ≤ k」的概率。相邻等级的差比跳到两端的差小。
    cdf_q = torch.cumsum(reported, dim=-1)
    cdf_t = torch.cumsum(target * mask, dim=-1)
    # 除以 K-1，和 Laya 的定义一致。最后一档两边的累积都是 1，那一项贡献 0。
    rps = (((cdf_q - cdf_t) ** 2) * mask).sum(dim=-1) / (width - 1)
    # RPS 越小越好，所以从奖励里减掉。
    return reward - w_rps * rps * is_score


def noise_std_at(step: int, total_steps: int, start: float, end: float) -> float:
    """第 1 步用 start，最后一步用 end，中间线性插值。

    step 从 1 计。总步数为 1 时直接用终点，避免除以 0。
    """
    if total_steps <= 1:
        return float(end)
    progress = (step - 1) / (total_steps - 1)
    return float(start + (end - start) * progress)


def rlcd_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
    group_size: int = 4,
    noise_std: float = 1.0,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """一组噪声报告的 REINFORCE 损失。noise 为 None 时现场采样标准正态，形状 [G, N, K]。

    策略是「当前 logit 为均值、noise_std 为标准差」的高斯。采样出的 logit 再 softmax 成报告。
    对数密度只对当前 logit 求导，奖励不回传。优势是该题 G 个奖励减去它们的均值。
    优势为正时，梯度把 logit 推向对应的扰动；为负时推开。

    高斯对数密度里不依赖参数的常数（log σ、2π）被丢掉了，梯度不受影响。
    """
    if group_size < 2:
        raise ValueError("RLCD 至少需要 2 个噪声样本才能做组内均值")
    # 标准差贴地，避免后面除以 0。调用方的退火终点是 0.3，正常不会撞上。
    std = max(float(noise_std), 1e-6)
    if noise is None:
        # 标准正态。后面再乘 std，得到 N(0, std^2)。
        samples = torch.randn(
            (group_size, logits.size(0), logits.size(1)),
            device=logits.device,
            dtype=logits.dtype,
        )
    else:
        # 测试传入固定噪声，形状必须是 [组, batch, 选项]。
        samples = noise.to(device=logits.device, dtype=logits.dtype)
        if samples.shape != (group_size, logits.size(0), logits.size(1)):
            raise ValueError("noise 的形状应为 [group, batch, options]，实际是 %s" % (tuple(samples.shape),))
    # 填充位不加噪声。这些位置的 logit 已经是 -1e4，扰动它们没有意义。
    perturbation = (samples * std).masked_fill(~marker_mask, 0.0)
    # 采样点本身不带梯度。策略对数密度通过 (z - logit) 连回当前 logit。
    sampled = logits.detach() + perturbation
    reported = masked_probs(sampled, marker_mask)
    reward = proper_reward(reported, target, qtype, marker_mask)
    # dim=0 是组。每道题用自己的 G 个奖励做基线，不同题之间不比较。
    advantage = reward - reward.mean(dim=0, keepdim=True)
    # log N(z; μ, σ) 对选项求和，只留真实选项。residual = (z - μ) / σ。
    residual = (sampled - logits) / std
    log_prob = -0.5 * (residual.square() * marker_mask.float()).sum(dim=-1)
    # advantage 切断梯度：奖励只当系数，不把 proper_reward 的导数传回来。
    # 对损失做梯度下降时，优势 > 0 的样本会让 μ 靠近 z。
    loss = -(advantage.detach() * log_prob).mean()
    return loss, {
        "reward": float(reward.detach().mean()),
        "advantage_abs": float(advantage.detach().abs().mean()),
    }


def mean_proper_reward(
    logits: torch.Tensor,
    target: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
) -> float:
    """未加噪声的报告上的平均奖励。选 epoch 用这个，而不是再采样一次。

    采样奖励随噪声浮动，同一份权重前后两次会不一样。干净分布上的奖励可以比较。
    """
    reported = masked_probs(logits, marker_mask)
    return float(proper_reward(reported, target, qtype, marker_mask).mean())
