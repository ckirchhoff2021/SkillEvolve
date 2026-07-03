"""
Select 模块 - 梯度裁剪
类比神经网络的梯度裁剪：按重要性排序候选编辑，裁剪到 top-L
L = 编辑预算（学习率）
"""

import re
from tools.common import gen_patch_text


def rank_and_select(
    optimizer_model,
    current_skill: str,
    merged_patch: dict,
    edit_budget: int,
) -> list:
    """
    对合并后的 patch 中的编辑按重要性排序，裁剪到 top-L
    
    Args:
        optimizer_model: 优化器模型
        current_skill: 当前技能文档
        merged_patch: 合并后的编辑集合
        edit_budget: 本步允许的最大编辑数（学习率）
    
    Returns:
        ranked_patch: 裁剪后的编辑列表（top-L）
    """
    edits = merged_patch.get("edits", [])
    
    if not edits:
        return []
    
    if len(edits) <= edit_budget:
        return edits
    
    # 需要裁剪，使用 LLM 排序
    edits_text = "\n\n".join([
        gen_patch_text(i, e)
        for i, e in enumerate(edits)
    ])
    
    prompt = open("prompts/rank.md", "r").read()
    prompt = prompt.format(current_skill, edits_text, edit_budget)
    
    result = optimizer_model.infer_with_text(prompt, "", temperature=0.1)
    output = result.get("result", "")
    
    # 解析排序后的编辑
    from gradient.reflect import _parse_patches
    return _parse_patches(output, "ranked")[:edit_budget]
