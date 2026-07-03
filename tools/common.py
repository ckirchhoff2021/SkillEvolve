from typing import List, Dict, Any
import re

def gen_patch_text(index: int, patch: Dict[str, Any]) -> str:
    """
    生成 patch 文本，用于 LLM 合并
    """
    title = f"### Patch {index+1}"
    op = f"操作: {patch.get("op", "N/A")}"
    patch_type = f"类型: {patch.get('type', 'N/A')}"
    reason = f"原因: {patch.get('reason', 'N/A')}"
    anchor = f"锚点: {patch.get('anchor', '')}" if 'anchor' in patch else ''
    target = f"目标: {patch.get('target', '')}" if 'target' in patch else ''
    content = f"内容: {patch.get('content', '')}"
    
    return f"{title}\n{op}\n{patch_type}\n{reason}\n{anchor}\n{target}\n{content}"


def extract_result(completion, pattern=r'<result>(.*?)</result>'):
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""