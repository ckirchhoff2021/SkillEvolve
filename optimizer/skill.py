"""
Update 模块 - 编辑应用（optimizer.step）
将排序裁剪后的编辑应用到技能文档，生成候选技能
"""
import re


def apply_patch(
    current_skill: str,
    ranked_patch: list,
) -> str:
    """
    将编辑应用到当前技能文档
    
    Args:
        current_skill: 当前技能文档内容
        ranked_patch: 排序裁剪后的编辑列表
    
    Returns:
        candidate_skill: 应用编辑后的候选技能文档
    """
    skill = current_skill
    
    for edit in ranked_patch:
        op = edit.get("op", "")
        
        if op == "append":
            content = edit.get("content", "")
            if content:
                skill = skill.rstrip() + "\n\n" + content
        
        elif op == "insert_after":
            anchor = edit.get("anchor", "")
            content = edit.get("content", "")
            if anchor and content:
                # 在锚点后插入
                idx = skill.find(anchor)
                if idx != -1:
                    insert_pos = idx + len(anchor)
                    skill = skill[:insert_pos] + "\n\n" + content + skill[insert_pos:]
        
        elif op == "replace":
            target = edit.get("target", "")
            content = edit.get("content", "")
            if target and content:
                if target in skill:
                    skill = skill.replace(target, content, 1)
        
        elif op == "delete":
            target = edit.get("target", "")
            if target and target in skill:
                skill = skill.replace(target, "", 1)
    
    return skill


def rewrite_skill(
    optimizer_model,
    current_skill: str,
    ranked_patch: list,
) -> str:
    """
    整篇改写模式：将编辑意图融入后重写完整技能文档
    
    Args:
        optimizer_model: 优化器模型
        current_skill: 当前技能文档
        ranked_patch: 编辑列表
    
    Returns:
        candidate_skill: 改写后的完整技能文档
    """
    edits_text = "\n".join([
        f"- [{e.get('op', '')}] {e.get('reason', '')}: {e.get('content', '')[:200]}"
        for e in ranked_patch
    ])
    
    prompt = f"""你是视觉问答技能优化专家。请根据以下编辑意图，重新编写完整的技能文档。

## 当前技能文档
{current_skill}

## 编辑意图
{edits_text}

## 任务要求
1. 将编辑意图完整融入技能文档
2. 保持原有结构和格式
3. 确保逻辑严谨，不引入矛盾
4. 输出完整的 Markdown 技能文档

请输出完整的技能文档。
"""
    
    result = optimizer_model.infer_with_text(prompt, "", temperature=0.2)
    return result.get("result", current_skill)
