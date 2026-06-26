"""
Trainer - 主训练循环
基于 SkillOpt ReflACT 六阶段流水线:
  Rollout → Reflect → Aggregate → Select → Update → Evaluate

每个 epoch 对训练集进行 step 级优化，每 step 执行完整六阶段流水线
"""
import os
import json
import hashlib
import time
from typing import List, Dict, Any

from rollouts.docvqa import DocVQARollout
from gradient.reflect import reflect_failures, reflect_successes
from gradient.aggregate import aggregate_patches
from optimizer.select import rank_and_select
from optimizer.skill import apply_patch, rewrite_skill
from optimizer.scheduler import Scheduler, build_scheduler
from evaluation.gate import simple_score, GateResult


class SkillTrainer:
    """
    Skill 进化训练器 - 完整 ReflACT 六阶段流水线编排
    
    核心流程:
      1. Rollout: 用当前技能驱动冻结模型对训练集推理，获取打分结果
      2. Reflect: 对失败/成功轨迹分组分析，生成结构化编辑 patch
      3. Aggregate: 分层合并多个 patch 为单一连贯的 merged_patch
      4. Select: 按重要性排序裁剪编辑，限制到 edit_budget
      5. Update: 应用编辑到技能文档，生成候选技能
      6. Evaluate/Gate: 在验证集上评估候选，决定 ACCEPT/REJECT
    """
    
    def __init__(
        self,
        target_model,
        optimizer_model,
        rollout: DocVQARollout,
        initial_skill_file: str,
        scheduler: Scheduler = None,
        max_epochs: int = 10,
        max_steps_per_epoch: int = 5,
        update_mode: str = "edit",  # "edit" or "rewrite"
        output_dir: str = None,
        early_stop_patience: int = 3,
        threshold: float = 0.95,
    ):
        self.target_model = target_model
        self.optimizer_model = optimizer_model
        self.rollout = rollout
        self.scheduler = scheduler or Scheduler(mode="constant", max_lr=3)
        self.max_epochs = max_epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.update_mode = update_mode
        self.output_dir = output_dir
        self.early_stop_patience = early_stop_patience
        self.threshold = threshold
        
        with open(initial_skill_file, "r", encoding="utf-8") as f:
            self.current_skill = f.read()
        
        self.best_skill = self.current_skill
        self.best_val_score = 0.0
        self.best_train_score = 0.0
        self.step_buffer = []
        self.meta_skill = ""
        self.step_count = 0
        self.no_improve_count = 0
        self.sel_cache = {}
        self.history = []
    
    def _hash_skill(self, skill: str) -> str:
        return hashlib.md5(skill.encode()).hexdigest()[:12]
    
    def _save_skill(self, skill: str, filename: str):
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(skill)
            print(f"  [Save] 技能已保存到: {path}")
    
    def _save_checkpoint(self, epoch: int, step: int, state: dict):
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            ckpt_path = os.path.join(self.output_dir, f"ckpt_epoch{epoch}_step{step}.json")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
    
    def train(self):
        print("=" * 60)
        print("SkillEvolve 训练开始")
        print(f"最大 Epoch: {self.max_epochs}, 每 Epoch 最大 Step: {self.max_steps_per_epoch}")
        print(f"更新模式: {self.update_mode}, 学习率模式: {self.scheduler.mode}")
        print(f"初始技能: {len(self.current_skill)} 字符")
        print("=" * 60)
        
        # 初始评估
        print("\n--- 初始技能评估 ---")
        train_results = self.rollout.rollout_train(self.current_skill)
        val_results = self.rollout.rollout_validate(self.current_skill)
        
        train_hard, train_soft = simple_score(train_results)
        val_hard, val_soft = simple_score(val_results)
        
        print(f"训练集: hard={train_hard:.4f}, soft={train_soft:.4f}")
        print(f"验证集: hard={val_hard:.4f}, soft={val_soft:.4f}")
        
        self.best_val_score = val_hard
        self.best_train_score = train_hard
        
        for epoch in range(self.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            print(f"{'='*60}")
            print(f"当前最佳验证集精度: {self.best_val_score:.4f}")
            
            epoch_no_improve = True
            
            for step in range(self.max_steps_per_epoch):
                print(f"\n--- Step {step + 1}/{self.max_steps_per_epoch} ---")
                self.step_count += 1
                
                edit_budget = self.scheduler.step(val_hard, self.best_val_score if step == 0 else None)
                print(f"  学习率(编辑预算): {edit_budget}")
                
                # ====== 六阶段流水线 ======
                
                # [1/6] Rollout
                print("  [1/6] Rollout: 在训练集上推理...")
                train_results = self.rollout.rollout_train(self.current_skill)
                train_hard, train_soft = simple_score(train_results)
                print(f"    训练集精度: hard={train_hard:.4f}, soft={train_soft:.4f}")
                
                if train_hard >= self.threshold:
                    print(f"  达到阈值 {self.threshold}, 停止训练")
                    self._save_skill(self.current_skill, "best_skill.md")
                    return self.current_skill
                
                failures = [r for r in train_results if not r.get("hard_correct", False)]
                successes = [r for r in train_results if r.get("hard_correct", False)]
                print(f"    失败: {len(failures)}, 成功: {len(successes)}")
                
                # [2/6] Reflect
                print("  [2/6] Reflect: 生成文本梯度...")
                failure_patches = reflect_failures(
                    self.optimizer_model,
                    self.current_skill,
                    failures,
                    step_buffer=self.step_buffer,
                    meta_skill=self.meta_skill,
                )
                success_patches = reflect_successes(
                    self.optimizer_model,
                    self.current_skill,
                    successes,
                    meta_skill=self.meta_skill,
                )
                print(f"    失败 patch: {len(failure_patches)}, 成功 patch: {len(success_patches)}")
                
                if not failure_patches and not success_patches:
                    print("  [Skip] 无可用 patch，跳过本 step")
                    continue
                
                # [3/6] Aggregate
                print("  [3/6] Aggregate: 分层合并 patch...")
                merged_patch = aggregate_patches(
                    self.optimizer_model,
                    self.current_skill,
                    failure_patches,
                    success_patches,
                )
                print(f"    合并后编辑数: {len(merged_patch['edits'])}")
                
                # [4/6] Select
                print("  [4/6] Select: 梯度裁剪...")
                ranked_patch = rank_and_select(
                    self.optimizer_model,
                    self.current_skill,
                    merged_patch,
                    edit_budget,
                )
                print(f"    裁剪后编辑数: {len(ranked_patch)}")
                
                if not ranked_patch:
                    print("  [Skip] 裁剪后无编辑，跳过本 step")
                    continue
                
                # [5/6] Update
                print("  [5/6] Update: 应用编辑...")
                if self.update_mode == "rewrite":
                    candidate_skill = rewrite_skill(
                        self.optimizer_model,
                        self.current_skill,
                        ranked_patch,
                    )
                else:
                    candidate_skill = apply_patch(self.current_skill, ranked_patch)
                
                skill_hash = self._hash_skill(candidate_skill)
                print(f"    候选技能哈希: {skill_hash}")
                
                # 候选缓存
                if skill_hash in self.sel_cache:
                    cached_score = self.sel_cache[skill_hash]
                    print(f"  [Cache] 命中缓存，验证分数: {cached_score:.4f}")
                    candidate_val_score = cached_score
                else:
                    # [6/6] Evaluate/Gate
                    print("  [6/6] Evaluate/Gate: 验证门控...")
                    val_results = self.rollout.rollout_validate(candidate_skill)
                    candidate_val_score, _ = simple_score(val_results)
                    print(f"    候选验证集精度: {candidate_val_score:.4f}")
                    self.sel_cache[skill_hash] = candidate_val_score
                
                # 门控决策
                if candidate_val_score > self.best_val_score:
                    decision = GateResult.ACCEPT_NEW_BEST
                    delta = candidate_val_score - self.best_val_score
                elif candidate_val_score > val_hard:
                    decision = GateResult.ACCEPT
                    delta = candidate_val_score - val_hard
                else:
                    decision = GateResult.REJECT
                    delta = candidate_val_score - val_hard
                
                print(f"    Gate: {decision} (score={candidate_val_score:.4f}, delta={delta:+.4f})")
                
                if decision in [GateResult.ACCEPT, GateResult.ACCEPT_NEW_BEST]:
                    self.current_skill = candidate_skill
                    self.best_val_score = max(self.best_val_score, candidate_val_score)
                    epoch_no_improve = False
                    self.no_improve_count = 0
                    
                    if decision == GateResult.ACCEPT_NEW_BEST:
                        self.best_skill = candidate_skill
                        self._save_skill(candidate_skill, "best_skill.md")
                        print(f"  *** 新的最佳技能! 验证精度: {self.best_val_score:.4f} ***")
                    print(f"  [ACCEPT] 技能已更新")
                else:
                    self.step_buffer.append(
                        f"Step{self.step_count}: 验证分数{candidate_val_score:.4f}未超过当前{val_hard:.4f}"
                    )
                    print(f"  [REJECT] 编辑被拒绝，加入负面上下文")
            
            self._save_checkpoint(epoch, step, {
                "epoch": epoch,
                "current_skill": self.current_skill,
                "best_val_score": self.best_val_score,
                "step_buffer": self.step_buffer,
            })
            
            if epoch_no_improve:
                self.no_improve_count += 1
                print(f"  [Warning] Epoch {epoch+1} 无验证集提升 (patience: {self.no_improve_count}/{self.early_stop_patience})")
                if self.no_improve_count >= self.early_stop_patience:
                    print(f"  [Early Stop] 连续 {self.early_stop_patience} 个 epoch 无提升，停止训练")
                    break
            else:
                self.no_improve_count = 0
        
        print(f"\n{'='*60}")
        print("训练完成")
        print(f"{'='*60}")
        print(f"最佳验证集精度: {self.best_val_score:.4f}")
        print(f"对应训练集精度: {self.best_train_score:.4f}")
        print(f"总 step 数: {self.step_count}")
        
        self._save_skill(self.best_skill, "final_skill.md")
        
        if self.output_dir:
            summary = {
                "best_val_score": self.best_val_score,
                "best_train_score": self.best_train_score,
                "total_steps": self.step_count,
                "final_skill_length": len(self.best_skill),
                "history": self.history,
            }
            summary_path = os.path.join(self.output_dir, "training_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"训练摘要保存到: {summary_path}")
        
        return self.best_skill
