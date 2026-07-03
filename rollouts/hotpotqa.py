import os
import datasets
import json
import numpy as np
from tools.retrieve import retrieve
from tools.common import extract_result


class HotpotQARollout(object):
    def __init__(self, target_model, dataset_config: dict):
        """
        Args:
            target_model: 目标模型实例（用于推理的冻结模型）
            dataset_config: 数据集配置
        """
        self.model = target_model
        self.dataset_config = dataset_config
        
        dataset_info = dataset_config["dataset_info"]
        dataset_path = dataset_info["dataset_path"]
        sub_dir = dataset_info["sub_dir"]
        self.dataset = datasets.load_dataset(
            dataset_path,
            sub_dir
        )    
        self.train_datas = self.dataset["train"]
        self.validate_datas = self.dataset["validation"]
        
        self.train_indices = np.random.choice(range(len(self.train_datas)), size=dataset_info['train_sample_num'], replace=False)
        self.validate_indices = np.random.choice(range(len(self.validate_datas)), size=dataset_info['validate_sample_num'], replace=False)
        
        self.instruction = open('prompts/hotpot/instruction.md', 'r').read()
        self.judge_prompt = open('prompts/hotpot/judge.md', 'r').read()
        self.max_iterations = 6
        
        
    @property
    def train(self):
        return self.train_datas
    
    @property
    def validate(self):
        return self.validate_datas
    
    @staticmethod
    def build_supporting_facts(facts, contexts):
        titles = facts['title']
        indices = facts['sent_id']
        supporting_facts = []
        for i in range(len(titles)):
            sentences = contexts[titles[i]]
            supporting_facts.append(sentences[indices[i]])
        return {
            'titles': titles,
            'sent_id': indices,
            'sents': supporting_facts,
        }
    
    @staticmethod
    def build_context(context):
        titles = context['title']
        sentences = context['sentences']
        outputs = {titles[i]: sentences[i] for i in range(len(titles))}
        return outputs
    
    
    def get_item(self, idx: int, split: str = "train"):
        if split == "train":
            x = self.train_datas[idx]
        else:
            x = self.validate_datas[idx]
    
        input_id = x['id']
        question = x['question']
        answer = x['answer']
        input_type = x['type']
        level = x['level']
        
        context = self.build_context(x['context'])
        supporting_facts = self.build_supporting_facts(x['supporting_facts'], context)
        
        return {
            'input_id': input_id,
            'input_type': input_type,
            'level': level,
            'question': question,
            'answer': answer,
            'context': context,
            'supporting_facts': supporting_facts,
        }
        
    def rollout_batch(self, skill_prompt: str, indices: list[int], train: bool = True) -> list:
        """
        对一批数据进行 rollout
        
        Args:
            skill_prompt: 当前技能文档
            indices: 数据索引列表
            train: 是否在训练集上 rollout
        
        Returns:
            results: 推理结果列表
        """
        
        results = []
        n_retrv = n_valid_retrv = 0
        for i, idx in enumerate(indices):
            try:
                data = self.get_item(idx, split="train" if train else "validate")
            except Exception as e:
                print(f"[Rollout] Error on idx {idx}: {e}")
                continue
            
            question = data["question"]
            answer = data['answer']    
            
            docs = list()
            for title in data['context']:
                sentences = data['context'][title]
                sents = '\n'.join(sentences)
                doc = f"{title}:\n {sents}"
                docs.append(doc)
            
            contexts = ''
            step = 0
            preds = '<error>'    
            while step < self.max_iterations:
                inputs = f"Contexts:  \n{contexts} \nQuestion: {question}"
                result = self.model.infer_with_text(inputs, skill_prompt=self.instruction)
                res = result.get("result", "")
                if '<result>' in res:
                    preds = extract_result(res, pattern=r'<result>(.*?)</result>')
                    break
                
                elif '<retrieve>' in res and len(docs) > 0:
                    query = extract_result(res, pattern=r'<retrieve>(.*?)</retrieve>')
                    retrieved = retrieve(docs, query, top_k=1)
                    doc = retrieved[0]['document'] 
                    idx = retrieved[0]['doc_idx']  
                    docs.pop(idx)               
                    contexts += f"\nstep {step}: \nQuery: {query} \nRetrieved: \n{doc}\n"
                
                    title = doc.split(':')[0]
                    n_retrv += 1
                    if title in data['supporting_facts']['titles']:
                        n_valid_retrv += 1
            
                else:
                    preds = '<error>'
                    break  
            
                step += 1 
            
            if preds == '<error>':  
                score = 0            
            else:
                judge = self.model.infer_with_text(f"Question: {question}\nAnswer: {answer}\nPrediction: {preds}", skill_prompt=self.judge_prompt)
                judge = judge.get("result", "")
                score = 1 if judge.strip() == 'Yes' else 0
                
            results.append({
                "index": idx,
                "question": question,
                "answer": answer,
                "score": score,
                "contexts": contexts,
                "prediction": res,
            })
            
            if (i + 1) % 10 == 0:
                print(f"  [Rollout] Processed {i+1}/{len(indices)}")
            
        scores = [item['score'] for item in results]
        print(f'Accuracy: {np.mean(scores)}')
        print(f'Retrieved Accuracy hit@1: {n_valid_retrv/n_retrv:.4f}')
        
        return results

    def rollout_train(self, skill_prompt: str) -> list:
        """在训练集上 rollout"""
        return self.rollout_batch(skill_prompt, self.train_indices, train=True)
    
    def rollout_validate(self, skill_prompt: str) -> list:
        """在验证集上 rollout"""
        return self.rollout_batch(skill_prompt, self.validate_indices, train=False)
    
    def rollout_step(self, skill_prompt: str, step_idx: int, max_steps: int) -> list:
        """单step rollout"""
        indices = self.train_indices
        batch_size = len(indices) // max_steps
        start = step_idx * batch_size
        end = min(start + batch_size, len(indices))
        return self.rollout_batch(skill_prompt, indices[start:end])
