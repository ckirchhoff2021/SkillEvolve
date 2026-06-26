"""
Scheduler 模块 - 学习率调度
"学习率" = 每步允许的最大编辑数（edit budget）
支持 constant、linear、cosine、autonomous 四种模式
"""


class Scheduler:
    """学习率调度器"""
    
    def __init__(self, mode: str = "constant", max_lr: int = 5, min_lr: int = 1, total_steps: int = 100):
        """
        Args:
            mode: 调度模式 (constant/linear/cosine/autonomous)
            max_lr: 最大编辑预算
            min_lr: 最小编辑预算
            total_steps: 总步数
        """
        self.mode = mode
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.current_lr = max_lr
    
    def step(self, val_accuracy: float = None, prev_val_accuracy: float = None) -> int:
        """
        计算下一步的学习率（编辑预算）
        
        Args:
            val_accuracy: 当前验证集精度（autonomous 模式需要）
            prev_val_accuracy: 上一步验证集精度（autonomous 模式需要）
        
        Returns:
            edit_budget: 本步允许的最大编辑数
        """
        self.current_step += 1
        
        if self.mode == "constant":
            self.current_lr = self.max_lr
        
        elif self.mode == "linear":
            progress = self.current_step / max(self.total_steps, 1)
            self.current_lr = max(
                self.min_lr,
                int(self.max_lr - (self.max_lr - self.min_lr) * progress)
            )
        
        elif self.mode == "cosine":
            import math
            progress = self.current_step / max(self.total_steps, 1)
            self.current_lr = max(
                self.min_lr,
                int(self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress)))
            )
        
        elif self.mode == "autonomous":
            self.current_lr = self._autonomous_lr(val_accuracy, prev_val_accuracy)
        
        return self.current_lr
    
    def _autonomous_lr(self, val_acc: float, prev_val_acc: float) -> int:
        """
        自主动态调整学习率
        - 验证集精度提升 → 降低学习率（收敛阶段）
        - 验证集精度不变 → 保持学习率
        - 验证集精度下降 → 提高学习率（探索更多编辑）
        """
        if val_acc is None or prev_val_acc is None:
            return self.max_lr
        
        if val_acc > prev_val_acc:
            # 精度提升，降低学习率
            self.current_lr = max(self.min_lr, self.current_lr - 1)
        elif val_acc < prev_val_acc:
            # 精度下降，提高学习率
            self.current_lr = min(self.max_lr, self.current_lr + 1)
        # 精度不变，保持当前学习率
        
        return self.current_lr


def build_scheduler(cfg: dict) -> Scheduler:
    """
    从配置构建调度器
    
    Args:
        cfg: 配置字典，包含 lr_mode, max_lr, min_lr, total_steps
    
    Returns:
        Scheduler 实例
    """
    return Scheduler(
        mode=cfg.get("lr_mode", "constant"),
        max_lr=cfg.get("max_lr", 5),
        min_lr=cfg.get("min_lr", 1),
        total_steps=cfg.get("total_steps", 100),
    )
