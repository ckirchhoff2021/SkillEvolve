"""
Aggregate 模块 - 分层合并
将多个 minibatch 产出的 patch 合并为单一连贯的 merged_patch
失败驱动的编辑优先于成功驱动的
"""
import re
from tools.common import gen_patch_text


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
    
    patches_text = "\n\n".join([gen_patch_text(i, p) for i, p in enumerate(patches)])
    
    prompt = open("prompts/apply.md", "r").read()
    prompt = prompt.format(current_skill, patches_text)

    result = optimizer_model.infer_with_text(prompt, "", temperature=0.2)
    output = result.get("result", "")
    
    # 解析合并后的 patch
    from gradient.reflect import _parse_patches
    return _parse_patches(output, "merged")
