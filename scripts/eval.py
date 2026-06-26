import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Prompt评估脚本"""
import os
from dotenv import load_dotenv
from skillevolve.config import load_config, flatten_config
from skillevolve.optimizer import VisualPromptOptimizer
from skillevolve.types import DatasetItem
from skillevolve.utils.logging import setup_logger

logger = setup_logger("skillevolve.eval")

def load_dataset(data_dir: str) -> List[DatasetItem]:
    """加载数据集"""
    from scripts.train import load_dataset as _load_dataset
    return _load_dataset(data_dir)

def main():
    load_dotenv()
    
    import argparse
    parser = argparse.ArgumentParser(description="SkillEvolve: Prompt Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--prompt", type=str, required=True, help="待评估的prompt")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    flat_cfg = flatten_config(cfg)
    
    # 加载数据集
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("数据集为空")
        return
    
    # 初始化优化器
    optimizer = VisualPromptOptimizer(flat_cfg)
    
    # 评估
    result = optimizer.evaluate_prompt(args.prompt, dataset)
    
    logger.info("=" * 50)
    logger.info("评估结果:")
    logger.info(f"总样本数: {result.total_samples}")
    logger.info(f"正确样本数: {result.correct_samples}")
    logger.info(f"准确率: {result.accuracy:.4f}")
    logger.info(f"平均耗时: {result.latency_avg:.2f}s")
    logger.info(f"错误样本数: {len(result.error_cases)}")
    logger.info("=" * 50)

if __name__ == "__main__":
    from typing import List
    main()
