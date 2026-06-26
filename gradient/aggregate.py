"""
Aggregate 模块 - 分层合并
将多个 minibatch 产出的 patch 合并为单一连贯的 merged_patch
失败驱动的编辑优先于成功驱动的
"""
import re


def aggregate_patches(
    optimizer_model,
    current_skill: str,
    failure_patches: list,
    success_patches: list,
) -> dict:
    """
    分层合并所有 patch 为最终的 merged_patch
    
    Args:
        optimizer_model: 优化器模型
        current_skill: 当前技能文档
        failure_patches: 失败驱动的编辑列表
        success_patches: 成功驱动的编辑列表
    
    Returns:
        merged_patch: 合并后的编辑字典，包含 edits 列表和 metadata
    """
    all_patches = failure_patches + success_patches
    
    if not all_patches:
        return {"edits": [], "source_counts": {"failure": 0, "success": 0}}
    
    # 如果有多个 patch，需要合并去重
    if len(all_patches) <= 3:
        # patch 少，直接去重即可
        merged = _deduplicate_patches(all_patches)
    else:
        # patch 多，使用 LLM 合并
        merged = _llm_merge_patches(optimizer_model, current_skill, all_patches)
    
    # 确保失败驱动的 patch 排在前面
    failure_first = sorted(merged, key=lambda x: (0 if x.get("type") == "failure" else 1))
    
    return {
        "edits": failure_first,
        "source_counts": {
            "failure": len(failure_patches),
            "success": len(success_patches),
        },
    }


def _deduplicate_patches(patches: list) -> list:
    """简单去重：移除重复的编辑"""
    seen = set()
    unique = []
    for p in patches:
        # 用 op + content/target 作为唯一标识
        key = (p.get("op", ""), p.get("content", "")[:100], p.get("target", "")[:100])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _llm_merge_patches(
    optimizer_model,
    current_skill: str,
    patches: list,
) -> list:
    """使用 LLM 合并多个 patch"""
    
    patches_text = "\n\n".join([
        f"""### Patch {i+1}
操作: {p.get('op', 'N/A')}
类型: {p.get('type', 'N/A')}
原因: {p.get('reason', 'N/A')}
{f'锚点: {p.get("anchor", "")}' if 'anchor' in p else ''}
{f'目标: {p.get("target", "")}' if 'target' in p else ''}
内容: {p.get('content', '')}"""
        for i, p in enumerate(patches)
    ])
    
    prompt = f"""你是视觉问答技能优化专家。以下有多个独立生成的编辑建议，请将它们合并为一个连贯的编辑集合。

## 当前技能文档
{current_skill[:3000]}

## 待合并的编辑（共{len(patches)}条）
{patches_text}

## 任务要求
1. 合并重复的编辑
2. 消除冲突的编辑（同一位置的不同编辑，保留更重要的）
3. 合并后保留的编辑必须使用以下格式：
<patch>
{chr(10)}<edit op="append|insert_after|replace|delete">
{chr(10)}<anchor>锚点文本或 N/A</anchor>
{chr(10)}<target>要替换/删除的原文（仅replace/delete需要）</target>
{chr(10)}<content>编辑内容（仅append/insert_after/replace需要）</content>
{chr(10)}</edit>
{chr(10)}<edit op="...">...</edit>
{chr(10)}</patch>

只输出 <patch> 块，不要输出其他内容。
"""
    
    result = optimizer_model.infer_with_text(prompt, "", temperature=0.2)
    output = result.get("result", "")
    
    # 解析合并后的 patch
    from gradient.reflect import _parse_patches
    return _parse_patches(output, "merged")
