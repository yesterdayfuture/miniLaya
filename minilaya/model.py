"""BERT encoder plus a small head that scores one logit per option marker."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DecisionBert(nn.Module):
    """预训练 BERT，加题型嵌入，再在每个选项标记上打一个 logit。

    编码器是双向的，所以每个选项在同一次前向里都能看到 state 和其他选项。
    不自回归生成文本。logit 只从 [MASK] 读取。后面的正温度不改变 argmax。
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        # BERT-base 的 768 维用 12 个头。隐藏维不能整除时退回 1 头，避免层构造失败。
        heads = max(1, hidden // 64)
        if hidden % heads != 0:
            heads = 1
        self.head = None
        if head_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=4 * hidden,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            # nested tensor 在变长 batch 上会改写实现，MPS 上容易出形状问题，这里关掉。
            self.head = nn.TransformerEncoder(layer, num_layers=head_layers, enable_nested_tensor=False)
        # 三种题型各一个向量，加到该题所有 token 上，让同一套打分头区分 choice / score / noul。
        self.type_emb = nn.Embedding(3, hidden)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # 决策头默认不 checkpoint。编码器的 checkpoint 在 train.py 里按设备打开。
        self.head_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # qtype 形状是 [batch]，扩成 [batch, 1, hidden] 后加到每个位置。
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            # PyTorch 的 key padding mask 里 True 表示“不要看”。BERT 的 attention_mask 相反，1 是有效 token。
            padding = ~attention_mask.bool()
            if self.head_checkpointing and self.training and torch.is_grad_enabled():
                for layer in self.head.layers:
                    # 默认参数把当前层和 mask 绑进闭包。循环变量若直接引用，所有层会共用最后一层。
                    def run(src, layer=layer, padding=padding):
                        return layer(src, src_key_padding_mask=padding)

                    hidden = checkpoint(run, hidden, use_reentrant=False)
            else:
                hidden = self.head(hidden, src_key_padding_mask=padding)
        # gather 需要 [batch, options, hidden]。补齐出来的 marker 下标是 0，靠 marker_mask 丢掉。
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        gathered = torch.gather(hidden, 1, index)
        logits = self.scorer(gathered).squeeze(-1).float()
        # 无效选项填一个很大的负数，softmax 后概率接近 0，且不会变成 NaN。
        return logits.masked_fill(~marker_mask, -1e4)


def masked_log_softmax(logits: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    """只在真实选项上归一化。补齐位置即使 logit 是 0，也不能分走概率。"""
    return torch.log_softmax(logits.masked_fill(~marker_mask, -1e4), dim=-1)


def soft_ce_loss(logits: torch.Tensor, target: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    """金标分布上的交叉熵，不是 one-hot。target 已经按选项对齐并归一化。"""
    log_prob = masked_log_softmax(logits, marker_mask)
    per_row = -(target * log_prob * marker_mask.float()).sum(dim=-1)
    return per_row.mean()
