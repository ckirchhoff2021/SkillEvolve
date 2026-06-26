"""
Evaluate/Gate 模块 - 验证门控
在 held-out 验证集上评估候选技能，决定是否接受编辑
类比神经网络的验证集 early-stopping
"""
import json
import os
from typing import Dict, Any, Tuple


class GateResult:
    """门控结果"""
    ACCEPT = "ACCEPT"
    ACCEPT_NEW_BEST = "ACCEPT(new best)"
    REJECT = "REJECT"
    
    def __init__(self, decision: str, val_score: float, delta: float):
        self.decision = decision
        self.val_score = val_score
        self.delta = delta
    
    def __str__(self):
        return f"{self.decision} (score={self.val_score:.4f}, delta={self.delta:+.4f})"


def evaluate_gate(
    candidate_skill: str,
    current_skill: str,
    current_val_score: float,
    best_val_score: float,
    rollout_fn,
    validate_samples: list,
) -> GateResult:
    """
    验证门控：在 held-out 验证集上评估候选技能
    
    Args:
        candidate_skill: 候选技能文档
        current_skill: 当前技能文档
        current_val_score: 当前技能验证集分数
        best_val_score: 历史最佳验证集分数
        rollout_fn: rollout 函数，接收 (skill, samples) -> results
        validate_samples: 验证集样本列表
    
    Returns:
        GateResult: 门控结果
    """
    # 在验证集上 rollout 候选技能
    results = rollout_fn(candidate_skill, validate_samples)
    
    # 计算候选技能分数
    correct = sum(1 for r in results if r.get("correct", False))
    candidate_val_score = correct / len(results) if results else 0.0
    
    delta = candidate_val_score - current_val_score
    
    if candidate_val_score > best_val_score:
        return GateResult(GateResult.ACCEPT_NEW_BEST, candidate_val_score, delta)
    elif candidate_val_score > current_val_score:
        return GateResult(GateResult.ACCEPT, candidate_val_score, delta)
    else:
        return GateResult(GateResult.REJECT, candidate_val_score, delta)


def simple_score(results: list) -> Tuple[float, float]:
    """
    简单评分：计算硬匹配准确率和软匹配准确率
    
    Args:
        results: 推理结果列表
    
    Returns:
        (hard_score, soft_score)
    """
    if not results:
        return 0.0, 0.0
    
    hard_correct = sum(1 for r in results if r.get("hard_correct", False))
    soft_correct = sum(1 for r in results if r.get("soft_correct", False))
    
    hard_score = hard_correct / len(results)
    soft_score = soft_correct / len(results)
    
    return hard_score, soft_score
