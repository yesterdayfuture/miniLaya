"""BERT 类型化决策模型，在 LocalLLaMA/typed-decisions 上微调。

load(path) 返回 DecisionAgent。predict(state, questions) 的 answers 与 Laya 对齐。
"""

__version__ = "0.1.0"


def load(path: str, device: str = "auto"):
    """加载检查点。延迟导入，避免 `import minilaya` 时立刻拉起模型代码。"""
    from minilaya.predict import DecisionAgent

    return DecisionAgent.load(path, device=device)
