"""
Reflect 模块 - 文本梯度生成
类比神经网络的反向传播：把打分后的轨迹转化为结构化编辑（patch）
将失败/成功轨迹分组为 minibatch，分析归纳出更稳健的规律
"""
import re
from typing import List, Dict, Any


EditOp = ["append", "insert_after", "replace", "delete"]


def reflect_failures(
    optimizer_model,
    current_skill: str,
    failures: List[Dict[str, Any]],
    step_buffer: List[str] = None,
    meta_skill: str = "",
) -> List[Dict[str, Any]]:
    """
    对失败轨迹进行反思，生成 failure_patches
    
    Args:
        optimizer_model: 优化器模型实例
        current_skill: 当前技能文档
        failures: 失败案例列表，每个包含 {question, answer, prediction, image_path, thought}
        step_buffer: 历史被拒编辑列表（负面上下文）
        meta_skill: 优化器侧记忆
    
    Returns:
        failure_patches: 结构化编辑列表
    """
    if not failures:
        return []
    
    # 限制数量避免 prompt 过长
    samples = failures[:5]
    
    buffer_text = ""
    if step_buffer:
        buffer_text = "## 历史被拒编辑（避免重复）\n" + "\n".join([f"- {b}" for b in step_buffer[-5:]])
    
    meta_text = f"## 优化器记忆\n{meta_skill}" if meta_skill else ""
    
    failure_examples = "\n\n".join([
        f"""### 失败案例 {i+1}
问题: {s.get('question', 'N/A')}
真实答案: {s.get('answer', 'N/A')}
模型输出: {s.get('prediction', 'N/A')}"""
        for i, s in enumerate(samples)
    ])
    
    prompt = f"""你是视觉问答技能优化专家。请分析以下失败案例，指出当前技能的不足，并提出具体的编辑建议。

## 当前技能文档
{current_skill[:3000]}

{buffer_text}

{meta_text}

## 失败案例分析（共{len(failures)}个失败案例，展示前{len(samples)}个）
{failure_examples}

## 任务要求
请分析这些失败案例的共同模式，提出针对当前技能的编辑建议。

每个编辑建议必须使用以下格式之一：
- append: 在文档末尾追加新规则
- insert_after: 在指定锚点文本后插入新内容
- replace: 替换指定范围的文本
- delete: 删除指定范围的文本

请按以下格式输出：
<patch>
{chr(10)}<edit op="append">
{chr(10)}<anchor>N/A</anchor>
{chr(10)}<content>要追加的内容</content>
{chr(10)}</edit>
{chr(10)}<edit op="insert_after">
{chr(10)}<anchor>锚点文本（需要匹配的原文）</anchor>
{chr(10)}<content>要插入的内容</content>
{chr(10)}</edit>
{chr(10)}<edit op="replace">
{chr(10)}<target>要替换的原文</target>
{chr(10)}<content>替换后的内容</content>
{chr(10)}</edit>
{chr(10)}<edit op="delete">
{chr(10)}<target>要删除的原文</target>
{chr(10)}</edit>
{chr(10)}<edit_reason>编辑原因说明</edit_reason>
{chr(10)}</patch>

只输出结构化编辑，不要输出完整文档。每条编辑需要包含 op、anchor/target、content 和 edit_reason。
最多提出 5 条编辑建议。
"""
    
    result = optimizer_model.infer_with_text(prompt, "", temperature=0.3)
    output = result.get("result", "")
    
    return _parse_patches(output, "failure")


def reflect_successes(
    optimizer_model,
    current_skill: str,
    successes: List[Dict[str, Any]],
    meta_skill: str = "",
) -> List[Dict[str, Any]]:
    """
    对成功轨迹进行反思，生成 success_patches（强化性梯度）
    """
    if not successes:
        return []
    
    samples = successes[:5]
    meta_text = f"## 优化器记忆\n{meta_skill}" if meta_skill else ""
    
    success_examples = "\n\n".join([
        f"""### 成功案例 {i+1}
问题: {s.get('question', 'N/A')}
答案: {s.get('answer', 'N/A')}
模型输出: {s.get('prediction', 'N/A')}"""
        for i, s in enumerate(samples)
    ])
    
    prompt = f"""你是视觉问答技能优化专家。以下案例中模型回答正确，请分析成功原因，提取值得固化的规则。

## 当前技能文档
{current_skill[:3000]}

{meta_text}

## 成功案例分析（共{len(successes)}个成功案例，展示前{len(samples)}个）
{success_examples}

## 任务要求
请分析这些成功案例的共同模式，提出将有效规则固化为技能文档的编辑建议。

每个编辑建议必须使用以下格式之一：
- append: 在文档末尾追加新规则
- insert_after: 在指定锚点文本后插入新内容
- replace: 替换指定范围的文本（强化现有规则）

请按以下格式输出：
<patch>
{chr(10)}<edit op="append">
{chr(10)}<anchor>N/A</anchor>
{chr(10)}<content>要追加的内容</content>
{chr(10)}</edit>
{chr(10)}<edit op="insert_after">
{chr(10)}<anchor>锚点文本</anchor>
{chr(10)}<content>要插入的内容</content>
{chr(10)}</edit>
{chr(10)}<edit op="replace">
{chr(10)}<target>要替换的原文</target>
{chr(10)}<content>替换后的内容</content>
{chr(10)}</edit>
{chr(10)}<edit_reason>编辑原因说明</edit_reason>
{chr(10)}</patch>

只输出结构化编辑。最多提出 3 条编辑建议。
"""
    
    result = optimizer_model.infer_with_text(prompt, "", temperature=0.3)
    output = result.get("result", "")
    
    return _parse_patches(output, "success")


def _parse_patches(text: str, patch_type: str) -> List[Dict[str, Any]]:
    """解析 LLM 输出的 patch 格式为结构化数据"""
    patches = []
    
    # 提取 <patch>...</patch> 块
    patch_blocks = re.findall(r'<patch>(.*?)</patch>', text, re.DOTALL)
    if not patch_blocks:
        return patches
    
    for block in patch_blocks:
        # 提取各个 edit
        edits = re.findall(r'<edit op="(\w+)">(.*?)</edit>', block, re.DOTALL)
        
        # 提取 edit_reason（在最后一个 edit 之后）
        reason_match = re.search(r'<edit_reason>(.*?)</edit_reason>', block, re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else ""
        
        for op, content in edits:
            anchor_match = re.search(r'<anchor>(.*?)</anchor>', content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)
            target_match = re.search(r'<target>(.*?)</target>', content, re.DOTALL)
            
            edit = {
                "op": op,
                "type": patch_type,
                "reason": reason,
            }
            
            if anchor_match:
                edit["anchor"] = anchor_match.group(1).strip()
            if content_match:
                edit["content"] = content_match.group(1).strip()
            if target_match:
                edit["target"] = target_match.group(1).strip()
            
            patches.append(edit)
    
    return patches
