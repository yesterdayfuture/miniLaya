"""RLCD 奖励和策略梯度的方向。不加载 BERT，也不读数据集。"""
import math

import torch

from minilaya.rlcd import proper_reward, rlcd_loss
from minilaya.schema import QTYPES


def test_proper_reward_matches_log_plus_spherical_and_rps():
    # 完全命中的 one-hot：对数分是 log(1)=0，球面分是 1/1=1，奖励 = 0 + 0.5×1。
    perfect = torch.tensor([[1.0, 0.0]])
    choice = torch.tensor([QTYPES["choice"]])
    mask = torch.tensor([[True, True]])
    reward = proper_reward(perfect, perfect, choice, mask)
    assert abs(float(reward) - 0.5) < 1e-6

    # 完全猜错：对数分被夹在 -9.21，球面分是 0。choice 题没有排名项。
    wrong = torch.tensor([[0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0]])
    missed = proper_reward(wrong, target, choice, mask)
    assert abs(float(missed) - (-9.21)) < 1e-6

    # 均匀分布：log(0.5) + 0.5 × (0.5 / ||[0.5, 0.5]||)。
    uniform = torch.tensor([[0.5, 0.5]])
    expected = math.log(0.5) + 0.5 * (0.5 / math.sqrt(0.5))
    got = proper_reward(uniform, target, choice, mask)
    assert abs(float(got) - expected) < 1e-6

    # 金标在等级 0。把质量放在等级 1 的 RPS 是 0.5，放在等级 2 的 RPS 是 1。
    # 奖励 = -9.21 - RPS，所以偏一档比偏两档更高。
    score = torch.tensor([QTYPES["score"]])
    wide = torch.tensor([[True, True, True]])
    gold = torch.tensor([[1.0, 0.0, 0.0]])
    adjacent = proper_reward(torch.tensor([[0.0, 1.0, 0.0]]), gold, score, wide)
    distant = proper_reward(torch.tensor([[0.0, 0.0, 1.0]]), gold, score, wide)
    assert float(adjacent) > float(distant)
    assert abs(float(adjacent) - (-9.21 - 0.5)) < 1e-5
    assert abs(float(distant) - (-9.21 - 1.0)) < 1e-5


def test_reinforce_pulls_logits_toward_the_better_report():
    # 当前 logit 是 [0, 0]，金标是第二项。形状 [组, batch, 选项]。
    # 第一组噪声 [0, 2] 把质量推向正确选项，第二组 [2, 0] 推向错误选项。
    # 梯度下降应抬高第二项 logit（梯度为负）、压低第一项 logit（梯度为正）。
    logits = torch.nn.Parameter(torch.zeros(1, 2))
    target = torch.tensor([[0.0, 1.0]])
    mask = torch.tensor([[True, True]])
    qtype = torch.tensor([QTYPES["choice"]])
    noise = torch.tensor([[[0.0, 2.0]], [[2.0, 0.0]]])
    loss, stats = rlcd_loss(logits, target, mask, qtype, group_size=2, noise_std=1.0, noise=noise)
    assert stats["advantage_abs"] > 0
    loss.backward()
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 0] > 0
