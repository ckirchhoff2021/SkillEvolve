"""
DocVQA Rollout 模块
使用目标模型（冻结）+ 当前技能文档对 DocVQA 数据进行推理
"""
import os
import re
import json
from datasets import load_dataset


class DocVQARollout:
    """DocVQA 数据上的 rollout"""
    
    def __init__(self, target_model, data_root: str = "/home/chenxiang.101/datas/"):
        """
        Args:
            target_model: 目标模型实例（用于推理的冻结模型）
            data_root: DocVQA 数据根目录
        """
        self.model = target_model
        self.data_root = data_root
        self.data_dir = os.path.join(data_root, "DocVQA")
        
        # 加载验证集（DocVQA 通常用验证集作为训练+验证数据）
        self.dataset = load_dataset(self.data_dir, "DocVQA")["validation"]
        
        # 加载索引
        sample_path =  "datas/docvqa/sample.json"
        import json
        with open(sample_path, "r") as f:
            samples = json.load(f)
        self.train_indices = samples["train"]
        self.validate_indices = samples["validate"]
    
    def _extract_answer(self, text: str) -> str:
        """从模型输出中提取答案"""
        # 尝试多种提取模式
        patterns = [
            r'<result>(.*?)</result>',
            r'<answer>(.*?)</answer>',
            r'答案[:：]\s*(.+)',
            r'Answer[:：]\s*(.+)',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        # 默认返回最后一段非空文本
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else text.strip()
    
    def _normalize_answer(self, ans: str) -> str:
        """标准化答案用于比较"""
        return re.sub(r'[^\w\s]', '', ans.lower()).strip()
    
    def rollout_batch(self, skill_prompt: str, indices: list) -> list:
        """
        对一批数据进行 rollout
        
        Args:
            skill_prompt: 当前技能文档
            indices: 数据索引列表
        
        Returns:
            results: 推理结果列表
        """
        results = []
        
        for i, idx in enumerate(indices):
            data = self.dataset[idx]
            question = data["question"]
            image = data.get("image", "")
            user_prompt = f"问题: {question}\n请回答以上问题。"
            
            try:
                result = self.model.infer_with_image(image, skill_prompt, user_prompt=user_prompt)
            except Exception as e:
                print(f"[Rollout] Error on idx {idx}: {e}")
                result = {"result": f"Error: {e}"}
          
            output = result.get("result", "")
            prediction = self._extract_answer(output)
            
            # 评分
            pred_norm = self._normalize_answer(prediction)
            # DocVQA 答案可能有多个
            ground_truths = data.get("answers", [data.get("answer", "")])
            correct = any(self._normalize_answer(gt) == pred_norm for gt in ground_truths)
            
            results.append({
                "index": idx,
                "question": question,
                "answer": ground_truths[0],
                "prediction": prediction,
                "full_output": json.dumps(result, ensure_ascii=False),
                "hard_correct": correct,
                "soft_correct": correct,  # DocVQA 用硬匹配
            })
            
            if (i + 1) % 10 == 0:
                print(f"  [Rollout] Processed {i+1}/{len(indices)}")
        
        return results
    
    def rollout_train(self, skill_prompt: str) -> list:
        """在训练集上 rollout"""
        return self.rollout_batch(skill_prompt, self.train_indices)
    
    def rollout_validate(self, skill_prompt: str) -> list:
        """在验证集上 rollout"""
        return self.rollout_batch(skill_prompt, self.validate_indices)
