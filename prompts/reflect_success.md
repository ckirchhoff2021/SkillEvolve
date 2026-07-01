你是技能优化专家。以下案例中模型回答正确，请分析成功原因，提取值得固化的规则。

## 当前技能文档
{}

{}

## 成功案例分析
{}

## 任务要求
请分析这些成功案例的共同模式，提出将有效规则固化为技能文档的编辑建议。

每个编辑建议必须使用以下格式之一：
- append: 在文档末尾追加新规则
- insert_after: 在指定锚点文本后插入新内容
- replace: 替换指定范围的文本（强化现有规则）

请按以下格式输出：
<patch>
<edit op="append">
<anchor>N/A</anchor>
<content>要追加的内容</content>
</edit>
<edit op="insert_after">
<anchor>锚点文本</anchor>
<content>要插入的内容</content>
</edit>
<edit op="replace">
<target>要替换的原文</target>
<content>替换后的内容</content>
</edit>
<edit_reason>编辑原因说明</edit_reason>
</patch>

只输出结构化编辑。最多提出 3 条编辑建议。