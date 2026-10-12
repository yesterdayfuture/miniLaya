#!/usr/bin/env python3
"""Render a vertical explainer: dataset, architecture, fine-tune, results.

Output: out/minilaya-explainer.mp4
Voice: macOS Tingting. On-screen type is the caption track.

    .venv/bin/python scripts/make_explainer_video.py --stills
    .venv/bin/python scripts/make_explainer_video.py
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out" / "pydeps"))

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
MX = 56
BOTTOM = 1668
FPS = 20
VOICE = "Tingting"
RATE = 198

BG = (11, 12, 16, 255)
CARD = (22, 24, 32, 255)
INK = (244, 244, 242, 255)
MUTED = (168, 174, 184, 255)
PINK = (251, 114, 153, 255)
YELLOW = (255, 214, 10, 255)
MINT = (61, 220, 151, 255)
BLUE = (126, 182, 255, 255)
CORAL = (255, 90, 54, 255)
LINE = (44, 49, 64, 255)
TRACK = (38, 42, 54, 255)

PING = "/System/Library/Fonts/PingFang.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"
SC = {"regular": 2, "medium": 5, "semibold": 8, "light": 11}


def font(size: int, weight: str = "medium", mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        index = 1 if weight in {"medium", "semibold", "bold"} else 0
        return ImageFont.truetype(MENLO, size, index=index)
    return ImageFont.truetype(PING, size, index=SC[weight])


def ease(t: float, delay: float = 0.0, dur: float = 0.38) -> float:
    x = (t - delay) / dur
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return 1 - (1 - x) ** 3


def wrap(text: str, face: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for ch in text:
        trial = line + ch
        if face.getlength(trial) <= max_w:
            line = trial
        else:
            if line:
                lines.append(line)
            line = ch
    if line:
        lines.append(line)
    if len(lines) >= 2 and len(lines[-1]) <= 2 and len(lines[-2]):
        lines[-1] = lines[-2][-1] + lines[-1]
        lines[-2] = lines[-2][:-1]
    return lines or [""]


def text_size(face: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
    box = font_bbox(face, text)
    return box[2] - box[0], box[3] - box[1]


def font_bbox(face: ImageFont.FreeTypeFont, text: str) -> tuple[int, int, int, int]:
    return ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=face)


def blit(base: Image.Image, layer: Image.Image, alpha: float, shift: int = 0) -> None:
    if alpha <= 0:
        return
    if shift:
        moved = Image.new("RGBA", base.size, (0, 0, 0, 0))
        moved.alpha_composite(layer, (0, shift))
        layer = moved
    if alpha < 0.999:
        r, g, b, a = layer.split()
        a = a.point(lambda p: int(p * alpha))
        layer = Image.merge("RGBA", (r, g, b, a))
    base.alpha_composite(layer)


def rounded(d: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=2) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def content_w() -> int:
    return W - 2 * MX


# ---------------------------------------------------------------------------
# Shots. Each one is a hard cut, in Bilibili knowledge-short rhythm.
# ---------------------------------------------------------------------------

SHOTS = [
    {
        "id": "hook",
        "kicker": "开场",
        "title": "同一套官方测试",
        "sub": "2,000 道决策 · 四个工作流",
        "voice": "同一套官方测试，两千道决策。三个模型，三个分数。",
        "draw": "hook",
    },
    {
        "id": "scores",
        "kicker": "开场",
        "title": "谁更高",
        "sub": "Jev 是零样本。Laya 和 miniLaya 都见过这批训练标签。",
        "voice": "miniLaya，零点五五九。Jev，零点七二七。Laya，零点七六六。",
        "draw": "scores",
    },
    {
        "id": "flows",
        "kicker": "01   数据集",
        "title": "四个工作流",
        "sub": "只加载配置 all。四个工作流不要再单独加一遍。",
        "voice": "数据来自 typed-decisions。四个工作流：发票处理、安全事件、客服，和智能体轨迹。",
        "draw": "flows",
    },
    {
        "id": "counts",
        "kicker": "01   数据集",
        "title": "案例怎么切",
        "sub": "按整个案例切分，不把同一案例的五道题拆开。",
        "voice": "训练集一千二百个案例，测试集四百个。每个案例五道题，测试一共两千条决策。",
        "draw": "counts",
    },
    {
        "id": "types",
        "kicker": "01   数据集",
        "title": "只有三种题",
        "sub": "选项顺序固定。金标概率要对齐到这个顺序。",
        "voice": "题只有三种。choice 是多选一，score 是等级，noul 是真或假。",
        "draw": "types",
    },
    {
        "id": "gold",
        "kicker": "01   数据集",
        "title": "金标是教师分布",
        "sub": "明显高于 0.735，通常是在拟合教师，不是更接近事实。",
        "voice": "金标不是标准答案。教师模型采样三次，取平均概率。自一致准确率是零点七三五。",
        "draw": "gold",
    },
    {
        "id": "sequence",
        "kicker": "01   数据集",
        "title": "装进一条序列",
        "sub": "选项一个都不能丢。超长时从 state 尾部截断。",
        "voice": "题干在前，每个选项前面放一个 MASK，state 接在最后。测试里大约百分之三被截断。",
        "draw": "sequence",
    },
    {
        "id": "stack",
        "kicker": "02   模型结构",
        "title": "只在 MASK 上打分",
        "sub": "一次前向看完所有选项，不逐字生成。",
        "voice": "打分只读 MASK。编码器之后加上题型嵌入，再过一层 Transformer，每个选项出一个 logit。",
        "draw": "stack",
    },
    {
        "id": "models",
        "kicker": "02   模型结构",
        "title": "两边的骨干",
        "sub": "结构同类。容量和上下文不同。",
        "voice": "miniLaya 是 bert-base，大约一亿一千万参数，上下文五百一十二。Laya 是 ModernBERT-large，四亿两千万参数，上下文一千零二十四。",
        "draw": "models",
    },
    {
        "id": "loss",
        "kicker": "03   微调",
        "title": "软交叉熵",
        "sub": "金标已经是分布，所以不对 one-hot。",
        "voice": "miniLaya 的损失是软交叉熵。预测分布要靠近教师分布，而不是只猜中一个选项。",
        "draw": "loss",
    },
    {
        "id": "split",
        "kicker": "03   微调",
        "title": "校准集先留出",
        "sub": "按工作流分层，按案例切开。测试集不参与选模型。",
        "voice": "开训之前，按案例留出百分之十做校准。训练一千零八十个案例。官方测试集只在最后打一次分。",
        "draw": "split",
    },
    {
        "id": "temp",
        "kicker": "03   微调",
        "title": "温度不改准确率",
        "sub": "正的标量温度不改变 argmax。它只动 KL、Brier 和 ECE。",
        "voice": "温度按题型拟合。choice 一点一，score 一点三，noul 一点零。最大项不变，准确率就不变。",
        "draw": "temp",
    },
    {
        "id": "rlcd",
        "kicker": "03   微调",
        "title": "Laya 用 RLCD",
        "sub": "动作是一整组概率。期望奖励最大的报告，是诚实的分布。",
        "voice": "Laya 的奖励是适当评分规则。对数分，加上球面分。等级题再减去排名概率分。旁边仍然有一项软交叉熵。",
        "draw": "rlcd",
    },
    {
        "id": "overall",
        "kicker": "04   效果",
        "title": "官方测试",
        "sub": "KL、Brier 越低越好。Laya 的模型卡没有公布 KL。",
        "voice": "miniLaya 准确率零点五五九，KL 零点二五三，Brier 零点一四四。",
        "draw": "overall",
    },
    {
        "id": "bytype",
        "kicker": "04   效果",
        "title": "分题型",
        "sub": "三题型都低。noul 的差距最大。Jev 没有分题型数字。",
        "voice": "分题型看，Laya 是零点七三三、零点八五七、零点七二三。miniLaya 是零点五二七、零点六一五、零点五四零。",
        "draw": "bytype",
    },
    {
        "id": "trace",
        "kicker": "04   效果",
        "title": "智能体轨迹最弱",
        "sub": "低于全体先验，比按训练集平均来猜更差。",
        "voice": "四个工作流里，差距最大的是智能体轨迹。零点四二二，低于训练集先验零点四八一。",
        "draw": "trace",
    },
    {
        "id": "speed",
        "kicker": "04   效果",
        "title": "速度不是同一块表",
        "sub": "机器不同，检查点也不同。比的是热身之后的单题。",
        "voice": "苹果 M1 上，热身之后一道题的中位时间是三十三点七毫秒。Laya 英文检查点在 T4 上公布的是三十九点五毫秒。",
        "draw": "speed",
    },
    {
        "id": "next",
        "kicker": "下一步",
        "title": "先动这三件事",
        "sub": "温度调完了。准确率还是 0.559。",
        "voice": "下一步把训练加长，换成 ModernBERT-base，并把工作流名字写进题干。",
        "draw": "next",
    },
]


def new_canvas() -> Image.Image:
    im = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.ellipse((640, -280, 1380, 460), fill=(251, 114, 153, 36))
    d.ellipse((-360, 1280, 520, 2140), fill=(255, 214, 10, 28))
    d.rectangle((0, 0, W, 8), fill=PINK)
    return im


def draw_chrome(d: ImageDraw.ImageDraw, shot: dict, progress: float, sub_alpha: float) -> int:
    d.rounded_rectangle((MX, 36, W - MX, 48), radius=6, fill=TRACK)
    x1 = MX + int(content_w() * max(0.0, min(1.0, progress)))
    if x1 > MX + 8:
        d.rounded_rectangle((MX, 36, x1, 48), radius=6, fill=PINK)
    d.text((MX, 68), shot["kicker"], font=font(28, "semibold"), fill=PINK)
    d.text((W - MX, 70), "miniLaya  ×  Laya", font=font(24, "medium"), fill=MUTED, anchor="ra")
    y = 118
    face = font(60, "semibold")
    for line in wrap(shot["title"], face, content_w()):
        d.text((MX, y), line, font=face, fill=INK)
        y += 76
    return y + 18


def draw_sub(d: ImageDraw.ImageDraw, text: str) -> None:
    box = (MX, 1704, W - MX, 1868)
    rounded(d, box, 28, (16, 18, 24, 255), LINE, 2)
    d.rectangle((MX, 1732, MX + 8, 1840), fill=YELLOW)
    face = font(34, "medium")
    lines = wrap(text, face, content_w() - 64)
    yy = 1736
    for line in lines[:3]:
        d.text((MX + 36, yy), line, font=face, fill=INK)
        yy += 44


def panel(base: Image.Image, t: float, delay: float, painter) -> None:
    a = ease(t, delay)
    if a <= 0:
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    painter(ImageDraw.Draw(layer))
    blit(base, layer, a, shift=int((1 - a) * 28))


def shot_hook(im, y0, t):
    def paint(d):
        d.text((MX, y0 + 40), "DECISIONS", font=font(28, "medium", mono=True), fill=MUTED)
        d.text((MX, y0 + 90), "2,000", font=font(210, "bold", mono=True), fill=YELLOW)
        d.text((MX, y0 + 340), "道决策", font=font(64, "semibold"), fill=INK)
        rows = [("4", "个工作流"), ("5", "题 / 案例"), ("1", "次官方 test")]
        x = MX
        for num, label in rows:
            rounded(d, (x, y0 + 480, x + 300, y0 + 680), 28, CARD, LINE)
            d.text((x + 28, y0 + 510), num, font=font(72, "bold", mono=True), fill=INK)
            d.text((x + 28, y0 + 600), label, font=font(30, "medium"), fill=MUTED)
            x += 320

    panel(im, t, 0.0, paint)


def shot_scores(im, y0, t):
    rows = [
        ("miniLaya", "0.559", MINT, "bert-base · 见过训练集"),
        ("Jev 1.13.0", "0.727", BLUE, "零样本 · 无分题型"),
        ("Laya", "0.766", YELLOW, "ModernBERT-large · RLCD"),
    ]

    def paint_one(i, name, num, color, note):
        def paint(d):
            y = y0 + i * 250
            rounded(d, (MX, y, W - MX, y + 230), 32, CARD, color, 3)
            d.text((MX + 36, y + 36), name, font=font(36, "semibold"), fill=color)
            d.text((W - MX - 36, y + 28), num, font=font(92, "bold", mono=True), fill=INK, anchor="ra")
            d.text((MX + 36, y + 150), note, font=font(30, "regular"), fill=MUTED)

        panel(im, t, 0.08 * i, paint)

    for i, row in enumerate(rows):
        paint_one(i, *row)


def shot_flows(im, y0, t):
    items = [
        ("01", "发票处理", "invoice"),
        ("02", "安全事件", "security"),
        ("03", "客服", "customer"),
        ("04", "智能体轨迹", "agent trace"),
    ]
    gap = 20
    w = (content_w() - gap) // 2
    h = 250

    def paint(d):
        for i, (num, name, en) in enumerate(items):
            r, c = divmod(i, 2)
            x = MX + c * (w + gap)
            y = y0 + r * (h + gap)
            rounded(d, (x, y, x + w, y + h), 32, CARD, LINE)
            d.text((x + 28, y + 28), num, font=font(28, "bold", mono=True), fill=YELLOW)
            d.text((x + 28, y + 88), name, font=font(46, "semibold"), fill=INK)
            d.text((x + 28, y + 170), en, font=font(28, "medium", mono=True), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_counts(im, y0, t):
    rows = [
        ("1,200", "训练案例", "四工作流各 300"),
        ("400", "测试案例", "官方 test"),
        ("2,000", "测试决策", "400 × 5 道题"),
    ]

    def paint(d):
        for i, (num, name, note) in enumerate(rows):
            y = y0 + i * 230
            a = ease(t, i * 0.08)
            if a <= 0:
                continue
            rounded(d, (MX, y, W - MX, y + 210), 32, CARD, LINE)
            d.text((MX + 36, y + 48), num, font=font(86, "bold", mono=True), fill=YELLOW if i == 2 else INK)
            d.text((MX + 520, y + 58), name, font=font(40, "semibold"), fill=INK)
            d.text((MX + 520, y + 120), note, font=font(28, "regular"), fill=MUTED)

    # Single panel; internal stagger is via ease on the same layer, so draw direct.
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    paint(ImageDraw.Draw(layer))
    blit(im, layer, 1.0)


def shot_types(im, y0, t):
    rows = [
        ("choice", "多选一", "继续，或转人工", MINT),
        ("score", "等级", "输出期望分数", YELLOW),
        ("noul", "真或假", "P(true)，没有 confidence", BLUE),
    ]

    def paint(d):
        for i, (name, kind, ex, color) in enumerate(rows):
            y = y0 + i * 230
            rounded(d, (MX, y, W - MX, y + 210), 32, CARD, LINE)
            d.rectangle((MX, y + 28, MX + 10, y + 182), fill=color)
            d.text((MX + 40, y + 36), name, font=font(40, "bold", mono=True), fill=color)
            d.text((MX + 40, y + 100), kind, font=font(40, "semibold"), fill=INK)
            d.text((MX + 40, y + 154), ex, font=font(28, "regular"), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_gold(im, y0, t):
    def paint(d):
        rounded(d, (MX, y0, W - MX, y0 + 360), 32, CARD, LINE)
        d.text((MX + 36, y0 + 36), "教师 × 3", font=font(32, "medium"), fill=MUTED)
        d.text((MX + 36, y0 + 100), "平均成分布", font=font(64, "semibold"), fill=INK)
        d.text((MX + 36, y0 + 200), "约 4B 参数  ·  温度 0.7", font=font(32, "regular"), fill=MUTED)
        d.text((MX + 36, y0 + 260), "不是客观标准答案", font=font(32, "medium"), fill=PINK)
        rounded(d, (MX, y0 + 400, W - MX, y0 + 760), 32, CARD, YELLOW, 3)
        d.text((MX + 36, y0 + 440), "教师自一致", font=font(34, "medium"), fill=YELLOW)
        d.text((MX + 36, y0 + 510), "0.735", font=font(140, "bold", mono=True), fill=INK)
        d.text((MX + 36, y0 + 680), "这份标签的上限参考", font=font(32, "regular"), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_sequence(im, y0, t):
    tokens = [
        ("[CLS]", "special"),
        ("题干", "text"),
        ("[SEP]", "special"),
        ("[MASK]", "mask"),
        ("继续", "text"),
        ("[MASK]", "mask"),
        ("转人工", "text"),
        ("[SEP]", "special"),
        ("state", "text"),
        ("[SEP]", "special"),
    ]

    def paint(d):
        rounded(d, (MX, y0, W - MX, y0 + 250), 32, CARD, LINE)
        x, y = MX + 28, y0 + 36
        limit = W - MX - 28
        for label, kind in tokens:
            face = font(30, "semibold", mono=kind != "text")
            tw, _ = text_size(face, label)
            box_w = tw + 36
            if x + box_w > limit:
                x = MX + 28
                y += 88
            fill = YELLOW if kind == "mask" else (PINK if kind == "special" else INK)
            bg = (42, 36, 12, 255) if kind == "mask" else (42, 28, 36, 255) if kind == "special" else (32, 36, 48, 255)
            rounded(d, (x, y, x + box_w, y + 68), 16, bg, None, 0)
            d.text((x + 18, y + 14), label, font=face, fill=fill)
            x += box_w + 12
        notes = [
            ("MASK", "logit 只从这里读"),
            ("3%", "测试决策被截断"),
            ("512", "bert-base 的上限"),
        ]
        x = MX
        for num, label in notes:
            rounded(d, (x, y0 + 280, x + 312, y0 + 480), 28, CARD, LINE)
            d.text((x + 24, y0 + 308), num, font=font(48, "bold", mono=True), fill=YELLOW)
            d.text((x + 24, y0 + 390), label, font=font(26, "medium"), fill=MUTED)
            x += 328

    panel(im, t, 0.0, paint)


def shot_stack(im, y0, t):
    steps = [
        ("1", "BERT 最后一层", "整段双向编码"),
        ("2", "加上题型嵌入", "choice 0 · score 1 · noul 2"),
        ("3", "Transformer × 1", "norm-first，padding 挡住"),
        ("4", "每个 MASK 一个 logit", "softmax 得到选项分布"),
    ]

    def paint(d):
        for i, (num, title, note) in enumerate(steps):
            y = y0 + i * 175
            rounded(d, (MX, y, W - MX, y + 155), 28, CARD, LINE)
            d.ellipse((MX + 24, y + 36, MX + 100, y + 112), fill=YELLOW if i == 3 else PINK)
            d.text((MX + 62, y + 74), num, font=font(32, "bold", mono=True), fill=BG, anchor="mm")
            d.text((MX + 124, y + 36), title, font=font(36, "semibold"), fill=INK)
            d.text((MX + 124, y + 92), note, font=font(28, "regular"), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_models(im, y0, t):
    blocks = [
        (MINT, "miniLaya", "bert-base-uncased", [("110M", "参数"), ("512", "上下文"), ("soft CE", "损失")]),
        (YELLOW, "Laya typed-decisions", "ModernBERT-large", [("421M", "参数"), ("1024", "上下文"), ("RLCD", "损失")]),
    ]

    def paint(d):
        for i, (color, name, enc, stats) in enumerate(blocks):
            y = y0 + i * 360
            rounded(d, (MX, y, W - MX, y + 340), 32, CARD, color, 3)
            d.text((MX + 32, y + 28), name, font=font(36, "semibold"), fill=color)
            d.text((MX + 32, y + 84), enc, font=font(28, "medium", mono=True), fill=MUTED)
            x = MX + 32
            for num, label in stats:
                d.text((x, y + 160), num, font=font(42, "bold", mono=True), fill=INK)
                d.text((x, y + 220), label, font=font(24, "regular"), fill=MUTED)
                x += 300

    panel(im, t, 0.0, paint)


def shot_loss(im, y0, t):
    def paint(d):
        rounded(d, (MX, y0, W - MX, y0 + 420), 32, CARD, YELLOW, 3)
        d.text((MX + 36, y0 + 36), "LOSS", font=font(26, "bold", mono=True), fill=YELLOW)
        d.text((MX + 36, y0 + 110), "− Σ  gold · log p", font=font(52, "bold", mono=True), fill=INK)
        d.text((MX + 36, y0 + 220), "gold 是教师概率", font=font(34, "medium"), fill=INK)
        d.text((MX + 36, y0 + 280), "只在真实选项上求和", font=font(34, "medium"), fill=MUTED)
        d.text((MX + 36, y0 + 340), "填充位不进 softmax", font=font(34, "medium"), fill=MUTED)
        rows = [("准确率", "argmax 是否一致"), ("KL / Brier", "整组概率像不像"), ("ECE", "高置信时是否对")]
        for i, (name, note) in enumerate(rows):
            x = MX + i * 324
            rounded(d, (x, y0 + 460, x + 308, y0 + 680), 24, CARD, LINE)
            d.text((x + 20, y0 + 490), name, font=font(28, "semibold"), fill=YELLOW)
            face = font(30, "medium")
            yy = y0 + 560
            for line in wrap(note, face, 260):
                d.text((x + 20, yy), line, font=face, fill=INK)
                yy += 42

    panel(im, t, 0.0, paint)


def shot_split(im, y0, t):
    def paint(d):
        d.text((MX, y0), "官方 train  1,200 个案例", font=font(32, "medium"), fill=MUTED)
        bar = (MX, y0 + 70, W - MX, y0 + 150)
        rounded(d, bar, 28, YELLOW, None, 0)
        cut = MX + int(content_w() * 0.9)
        d.rounded_rectangle((MX, y0 + 70, cut, y0 + 150), radius=28, fill=MINT)
        d.rectangle((cut - 28, y0 + 70, cut, y0 + 150), fill=MINT)
        d.text((MX + 28, y0 + 90), "拟合 90%", font=font(32, "semibold"), fill=BG)
        d.text((MX, y0 + 190), "1,080", font=font(64, "bold", mono=True), fill=MINT)
        d.text((MX + 270, y0 + 214), "案例", font=font(36, "semibold"), fill=MINT)
        d.text((MX + 520, y0 + 190), "120", font=font(64, "bold", mono=True), fill=YELLOW)
        d.text((MX, y0 + 280), "拿来更新权重", font=font(30, "regular"), fill=MUTED)
        d.text((MX + 520, y0 + 280), "校准，先留出", font=font(30, "regular"), fill=MUTED)
        rounded(d, (MX, y0 + 400, W - MX, y0 + 620), 32, CARD, PINK, 3)
        d.text((MX + 36, y0 + 440), "TEST  LOCKED", font=font(28, "bold", mono=True), fill=PINK)
        d.text((MX + 36, y0 + 500), "400 案例  ·  2,000 决策", font=font(40, "semibold"), fill=INK)
        d.text((MX + 36, y0 + 560), "只在最后打一次分", font=font(30, "regular"), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_temp(im, y0, t):
    rows = [("choice", "1.1", MINT), ("score", "1.3", YELLOW), ("noul", "1.0", BLUE)]

    def paint(d):
        for i, (name, temp, color) in enumerate(rows):
            y = y0 + i * 200
            rounded(d, (MX, y, W - MX, y + 180), 32, CARD, LINE)
            d.text((MX + 36, y + 56), name, font=font(40, "semibold"), fill=color)
            d.text((W - MX - 36, y + 40), temp, font=font(84, "bold", mono=True), fill=INK, anchor="ra")

    panel(im, t, 0.0, paint)


def shot_rlcd(im, y0, t):
    rows = [
        ("对数分", "log q(y)", "和软交叉熵是同一项"),
        ("球面分 × 0.5", "q(y) / ||q||", "惩罚把概率堆在一项上"),
        ("排名概率分", "只在 score 上减", "偏三个等级，比偏一档更亏"),
        ("怎么更新", "4 个噪声样本", "高于组内均值的，拉向当前 logit"),
    ]

    def paint(d):
        for i, (title, formula, note) in enumerate(rows):
            y = y0 + i * 195
            rounded(d, (MX, y, W - MX, y + 178), 28, CARD, LINE)
            d.text((MX + 28, y + 22), title, font=font(32, "semibold"), fill=YELLOW if i >= 2 else INK)
            d.text((MX + 28, y + 74), formula, font=font(36, "semibold"), fill=MINT)
            d.text((MX + 28, y + 124), note, font=font(26, "regular"), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_overall(im, y0, t):
    def paint(d):
        rounded(d, (MX, y0, W - MX, y0 + 420), 32, CARD, MINT, 3)
        d.text((MX + 36, y0 + 28), "miniLaya", font=font(32, "semibold"), fill=MINT)
        d.text((MX + 36, y0 + 90), "0.559", font=font(150, "bold", mono=True), fill=INK)
        d.text((MX + 36, y0 + 270), "准确率", font=font(32, "medium"), fill=MUTED)
        pair = [("0.253", "KL"), ("0.144", "Brier"), ("0.061", "ECE")]
        for i, (num, name) in enumerate(pair):
            x = MX + i * 324
            rounded(d, (x, y0 + 470, x + 308, y0 + 690), 24, CARD, LINE)
            d.text((x + 24, y0 + 500), num, font=font(48, "bold", mono=True), fill=INK)
            d.text((x + 24, y0 + 580), name, font=font(28, "medium"), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_bytype(im, y0, t):
    groups = [
        ("choice", 0.527, 0.733),
        ("noul", 0.615, 0.857),
        ("score", 0.540, 0.723),
    ]

    def paint(d):
        max_w = 560
        for i, (name, mini, laya) in enumerate(groups):
            y = y0 + i * 250
            d.text((MX, y), name, font=font(32, "semibold", mono=True), fill=INK)
            for j, (val, color, label) in enumerate(((mini, MINT, "mini"), (laya, YELLOW, "Laya"))):
                yy = y + 52 + j * 78
                d.text((MX, yy + 8), label, font=font(24, "medium"), fill=MUTED)
                x0 = MX + 110
                rounded(d, (x0, yy, x0 + max_w, yy + 44), 12, TRACK, None, 0)
                fill_w = int(max_w * val)
                if fill_w > 8:
                    d.rounded_rectangle((x0, yy, x0 + fill_w, yy + 44), radius=12, fill=color)
                d.text((x0 + max_w + 16, yy + 4), f"{val:.3f}", font=font(32, "bold", mono=True), fill=INK)

    panel(im, t, 0.0, paint)


def shot_trace(im, y0, t):
    def paint(d):
        d.text((MX, y0), "agent trace", font=font(30, "medium", mono=True), fill=CORAL)
        d.text((MX, y0 + 40), "0.422", font=font(160, "bold", mono=True), fill=CORAL)
        rows = [("0.481", "训练集先验", MUTED), ("0.422", "miniLaya", CORAL), ("0.730", "Laya", YELLOW)]
        for i, (num, name, color) in enumerate(rows):
            y = y0 + 280 + i * 150
            rounded(d, (MX, y, W - MX, y + 130), 28, CARD, LINE)
            d.text((MX + 28, y + 36), name, font=font(32, "medium"), fill=MUTED)
            d.text((W - MX - 28, y + 28), num, font=font(52, "bold", mono=True), fill=color, anchor="ra")

    panel(im, t, 0.0, paint)


def shot_speed(im, y0, t):
    cards = [
        (MINT, "33.7 ms", "miniLaya", "Apple M1  ·  MPS", "热身 20 次的 p50  ·  1 题"),
        (YELLOW, "39.5 ms", "Laya 英文检查点", "Tesla T4", "公布值  ·  不是 typed-decisions"),
    ]

    def paint(d):
        for i, (color, num, name, device, note) in enumerate(cards):
            y = y0 + i * 340
            rounded(d, (MX, y, W - MX, y + 310), 32, CARD, color, 3)
            d.text((MX + 32, y + 28), name, font=font(32, "semibold"), fill=color)
            d.text((MX + 32, y + 90), num, font=font(84, "bold", mono=True), fill=INK)
            d.text((MX + 32, y + 200), device, font=font(30, "medium"), fill=INK)
            d.text((MX + 32, y + 248), note, font=font(26, "regular"), fill=MUTED)

    panel(im, t, 0.0, paint)


def shot_next(im, y0, t):
    rows = [
        ("1", "加长训练", "校准损失到第 3 个 epoch 还在降"),
        ("2", "换 ModernBERT-base", "上下文拉到 1024，M1 上还能训"),
        ("3", "写上工作流名", "智能体轨迹现在低于先验"),
    ]

    def paint(d):
        for i, (num, title, note) in enumerate(rows):
            y = y0 + i * 230
            rounded(d, (MX, y, W - MX, y + 210), 32, CARD, LINE)
            d.text((MX + 32, y + 48), num, font=font(72, "bold", mono=True), fill=YELLOW)
            d.text((MX + 140, y + 48), title, font=font(40, "semibold"), fill=INK)
            d.text((MX + 140, y + 120), note, font=font(28, "regular"), fill=MUTED)

    panel(im, t, 0.0, paint)


DRAW = {
    "hook": shot_hook,
    "scores": shot_scores,
    "flows": shot_flows,
    "counts": shot_counts,
    "types": shot_types,
    "gold": shot_gold,
    "sequence": shot_sequence,
    "stack": shot_stack,
    "models": shot_models,
    "loss": shot_loss,
    "split": shot_split,
    "temp": shot_temp,
    "rlcd": shot_rlcd,
    "overall": shot_overall,
    "bytype": shot_bytype,
    "trace": shot_trace,
    "speed": shot_speed,
    "next": shot_next,
}


def render_frame(shot: dict, local_t: float, progress: float) -> Image.Image:
    im = new_canvas()
    d = ImageDraw.Draw(im)
    y0 = draw_chrome(d, shot, progress, 1.0)
    draw_sub(d, shot["sub"])
    DRAW[shot["draw"]](im, y0, local_t)
    return im.convert("RGB")


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        text=True,
    )
    return float(out.strip())


def synth(shot: dict, voice_dir: Path, rate: int) -> Path:
    voice_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{VOICE}|{rate}|pad0.28|{shot['voice']}".encode()).hexdigest()[:12]
    wav = voice_dir / f"{shot['id']}-{key}.wav"
    if wav.exists() and wav.stat().st_size > 1000:
        return wav
    aiff = wav.with_suffix(".aiff")
    subprocess.check_call(["say", "-v", VOICE, "-r", str(rate), "-o", str(aiff), shot["voice"]])
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(aiff), "-ar", "48000", "-ac", "1",
            "-af", "apad=pad_dur=0.28",
            str(wav),
        ]
    )
    aiff.unlink(missing_ok=True)
    return wav


def build_audio(wavs: list[Path], durs: list[float], dest: Path, tail: float) -> None:
    lst = dest.with_suffix(".txt")
    lines = []
    for wav in wavs:
        lines.append(f"file '{wav}'")
    lst.write_text("\n".join(lines) + "\n")
    concat = dest.with_name("concat.wav")
    subprocess.check_call(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(concat)]
    )
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(concat),
            "-af", f"apad=pad_dur={tail}",
            str(dest),
        ]
    )


def encode(frames, audio: Path, dest: Path, fps: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-i", str(audio),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(dest),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError("ffmpeg exited %s" % code)


def iter_frames(durs: list[float], fps: int, tail: float):
    spans = []
    t = 0.0
    for shot, dur in zip(SHOTS, durs):
        spans.append((shot, t, t + dur))
        t += dur
    total = t + tail
    n = math.ceil(total * fps)
    for i in range(n):
        now = i / fps
        shot, start, end = spans[-1]
        for cand, a, b in spans:
            if a <= now < b:
                shot, start, end = cand, a, b
                break
        local = 0.0 if end <= start else (now - start) / (end - start)
        local = max(0.0, min(1.0, local))
        progress = min(1.0, now / total)
        yield render_frame(shot, local, progress)


def write_stills(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(SHOTS)
    for i, shot in enumerate(SHOTS):
        frame = render_frame(shot, 1.0, (i + 1) / n)
        path = out_dir / f"{i:02d}-{shot['id']}.png"
        frame.save(path)
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stills", action="store_true")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--rate", type=int, default=RATE)
    parser.add_argument("--out", default=str(ROOT / "out" / "minilaya-explainer.mp4"))
    args = parser.parse_args()
    if args.stills:
        write_stills(ROOT / "out" / "stills")
        return
    voice_dir = ROOT / "out" / "voice"
    wavs = []
    durs = []
    for shot in SHOTS:
        wav = synth(shot, voice_dir, args.rate)
        wavs.append(wav)
        durs.append(probe_duration(wav))
        print("%s  %.2fs" % (shot["id"], durs[-1]))
    tail = 0.9
    audio = ROOT / "out" / "voice" / "narration.wav"
    build_audio(wavs, durs, audio, tail)
    dest = Path(args.out)
    encode(iter_frames(durs, args.fps, tail), audio, dest, args.fps)
    print(dest)
    print("duration %.1fs" % probe_duration(dest))


if __name__ == "__main__":
    main()
