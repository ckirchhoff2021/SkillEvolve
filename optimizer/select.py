"""
Select 模块 - 梯度裁剪
类比神经网络的梯度裁剪：按重要性排序候选编辑，裁剪到 top-L
L = 编辑预算（学习率）
"""
import re


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
        f"""### 候选编辑 {i+1}
操作: {e.get('op', 'N/A')}
类型: {e.get('type', 'N/A')}
原因: {e.get('reason', 'N/A')}
{f'锚点: {e.get("anchor", "")}' if 'anchor' in e else ''}
{f'目标: {e.get("target", "")}' if 'target' in e else ''}
内容: {e.get('content', '')}"""
        for i, e in enumerate(edits)
    ])
    
    prompt = f"""你是视觉问答技能优化专家。以下有多个候选编辑，请按重要性从高到低排序，并只保留前 {edit_budget} 条最重要的编辑。

## 当前技能文档
{current_skill[:3000]}

## 候选编辑（共{len(edits)}条，预算上限: {edit_budget}）
{edits_text}

## 任务要求
1. 按重要性排序（失败驱动的编辑通常更重要）
2. 优先保留能修复关键错误的编辑
3. 避免保留重复或冲突的编辑
4. 只输出前 {edit_budget} 条

请按以下格式输出：
<selected>
<rank>1</rank>
<edit op="...">...</edit>
<rank>2</rank>
<edit op="...">...</edit>
...
</selected>
"""
    
    result = optimizer_model.infer_with_text(prompt, "", temperature=0.1)
    output = result.get("result", "")
    
    # 解析排序后的编辑
    from gradient.reflect import _parse_patches
    return _parse_patches(output, "ranked")[:edit_budget]
