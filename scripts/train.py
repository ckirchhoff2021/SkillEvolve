import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime
from dotenv import load_dotenv

from skillevolve.config import load_config, flatten_config
from skillevolve.types import DatasetItem, OptimizationResult, OptimizationIteration
from skillevolve.utils.logging import setup_logger

logger = setup_logger("skillevolve.train")

def load_dataset(data_dir: str) -> list[DatasetItem]:
    """加载数据集，支持标准annotations.json格式"""
    dataset = []
    ann_path = os.path.join(data_dir, "annotations.json")
    
    if not os.path.exists(ann_path):
        logger.warning(f"标注文件不存在: {ann_path}，将返回空数据集")
        return dataset
    
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            anns = json.load(f)
        
        for ann in anns:
            img_path = os.path.join(data_dir, ann["image"])
            if not os.path.exists(img_path):
                logger.warning(f"图片不存在: {img_path}，跳过该样本")
                continue
            dataset.append(DatasetItem(
                image_path=img_path,
                label=ann["label"],
                metadata=ann.get("metadata", {})
            ))
        
        logger.info(f"成功加载 {len(dataset)} 个样本 from {data_dir}")
        return dataset
    
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        return dataset

def save_result(result, save_dir: str):
    """保存优化结果"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存完整结果
    result_dict = {
        "best_prompt": result.best_prompt,
        "best_accuracy": result.best_accuracy,
        "total_iterations": result.total_iterations,
        "config": result.config,
        "iterations": [
            {
                "iteration": iter.iteration,
                "prompt": iter.prompt,
                "train_accuracy": iter.train_accuracy,
                "val_accuracy": iter.val_accuracy,
                "negative_feedback_count": iter.negative_feedback_count,
                "new_prompt": iter.new_prompt,
                "new_val_accuracy": iter.new_val_accuracy,
                "is_best": iter.is_best
            } for iter in result.iterations
        ]
    }
    
    result_path = os.path.join(save_dir, f"optimization_result_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    # 保存最优prompt
    best_prompt_path = os.path.join(save_dir, f"best_prompt_{timestamp}.txt")
    with open(best_prompt_path, "w", encoding="utf-8") as f:
        f.write(result.best_prompt)
    
    logger.info(f"💾 结果已保存到目录: {save_dir}")
    logger.info(f"📄 完整优化记录: {result_path}")
    logger.info(f"🏆 最优prompt: {best_prompt_path}")

def main():
    load_dotenv()
    
    import argparse
    parser = argparse.ArgumentParser(description="SkillEvolve: Visual Prompt Optimization Training Loop")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--initial-prompt", type=str, required=True, help="待优化的初始prompt")
    parser.add_argument("--demo", action="store_true", help="Demo模式，使用模拟数据运行流程（不需要真实数据集和API密钥）")
    args = parser.parse_args()
    
    # Demo模式：模拟运行完整流程，不需要任何依赖
    if args.demo:
        logger.info("🚀 启动Demo模式，将模拟运行完整训练流程")
        import random
        
        # 模拟优化过程
        iterations = []
        initial_acc = random.uniform(0.6, 0.75)
        best_acc = initial_acc
        best_prompt = args.initial_prompt
        
        iterations.append(OptimizationIteration(
            iteration=0,
            prompt=args.initial_prompt,
            train_accuracy=initial_acc - 0.05,
            val_accuracy=initial_acc,
            negative_feedback_count=random.randint(10, 20),
            is_best=True
        ))
        
        logger.info("=" * 60)
        logger.info("开始Skill优化训练流程")
        logger.info(f"初始prompt: {args.initial_prompt[:100]}...")
        logger.info(f"模拟训练集大小: 100, 验证集大小: 50")
        logger.info(f"最大迭代轮数: 3, 目标准确率: 0.95")
        logger.info("=" * 60)
        
        logger.info("\n[Step 1/2] 初始prompt验证集评估")
        logger.info(f"评估完成，准确率: {initial_acc:.4f}, 平均耗时: 0.25s, 错误数: {int((1-initial_acc)*50)}")
        
        for i in range(3):
            logger.info(f"\n\n🚀 第 {i+1}/3 轮训练")
            logger.info(f"当前最优准确率: {best_acc:.4f}, 当前prompt准确率: {best_acc:.4f}")
            
            logger.info("\n[Step 1/4] 训练集评估，收集错误案例")
            train_acc = best_acc - 0.03 + random.uniform(-0.02, 0.02)
            error_count = random.randint(5, 15)
            logger.info(f"评估完成，准确率: {train_acc:.4f}, 平均耗时: 0.23s, 错误数: {error_count}")
            logger.info(f"收集到 {error_count} 个错误案例，开始分析优化")
            
            logger.info("\n[Step 2/4] 调用Judge模型生成优化后prompt")
            new_prompt = f"{args.initial_prompt} [优化版{i+1}]：请特别注意区分猫和狗的耳朵、脸型特征，结果严格用<result>标签包裹"
            logger.info(f"生成新prompt: {new_prompt[:100]}...")
            
            logger.info("\n[Step 3/4] 验证集评估新prompt")
            new_acc = min(best_acc + random.uniform(0.05, 0.15), 0.98)
            logger.info(f"评估完成，准确率: {new_acc:.4f}, 平均耗时: 0.24s, 错误数: {int((1-new_acc)*50)}")
            
            logger.info("\n[Step 4/4] 对比选择最优prompt")
            is_best = new_acc > best_acc
            if is_best:
                best_acc = new_acc
                best_prompt = new_prompt
                logger.info(f"✅ 找到更好的prompt，准确率提升至 {best_acc:.4f}")
            else:
                logger.info(f"⚠️ 新prompt效果没有提升（{new_acc:.4f} <= {best_acc:.4f}），保持原有prompt")
            
            iterations.append(OptimizationIteration(
                iteration=i+1,
                prompt=best_prompt if is_best else iterations[-1].prompt,
                train_accuracy=train_acc,
                val_accuracy=best_acc,
                negative_feedback_count=error_count,
                new_prompt=new_prompt,
                new_val_accuracy=new_acc,
                is_best=is_best
            ))
        
        logger.info("\n" + "="*60)
        logger.info("🏆 优化训练完成")
        logger.info(f"最优准确率: {best_acc:.4f}")
        logger.info(f"最优prompt: {best_prompt}")
        logger.info(f"总迭代轮数: 3")
        logger.info("="*60)
        
        result = OptimizationResult(
            best_prompt=best_prompt,
            best_accuracy=best_acc,
            total_iterations=3,
            iterations=iterations,
            config={"demo": True}
        )
        
        save_result(result, "./outputs/demo/")
        logger.info("✅ Demo模式运行成功！完整训练流程验证通过")
        logger.info("💡 正式运行请准备数据集并配置API密钥后去掉--demo参数即可")
        return
    
    # 正式运行模式才导入optimizer
    from skillevolve.optimizer import VisualPromptOptimizer
    
    logger.info(f"📝 加载配置文件: {args.config}")
    cfg = load_config(args.config)
    flat_cfg = flatten_config(cfg)
    
    # 加载数据集
    logger.info(f"📂 加载训练集: {cfg['data']['train_path']}")
    train_dataset = load_dataset(cfg['data']['train_path'])
    logger.info(f"📂 加载验证集: {cfg['data']['val_path']}")
    val_dataset = load_dataset(cfg['data']['val_path'])
    
    if not train_dataset or not val_dataset:
        logger.error("❌ 训练集或验证集为空，请检查数据路径和标注文件")
        logger.info("💡 如果你只是想测试流程，可以添加 --demo 参数运行Demo模式")
        return
    
    # 初始化优化器
    logger.info("🔧 初始化优化器")
    try:
        optimizer = VisualPromptOptimizer(flat_cfg)
    except Exception as e:
        logger.error(f"❌ 初始化优化器失败: {str(e)}")
        return
    
    # 开始优化训练
    try:
        result = optimizer.optimize(args.initial_prompt, train_dataset, val_dataset)
    except KeyboardInterrupt:
        logger.info("⏹️ 用户手动中断训练")
        return
    except Exception as e:
        logger.error(f"❌ 优化过程出错: {str(e)}", exc_info=True)
        return
    
    # 保存结果
    save_result(result, cfg['output']['save_dir'])
    logger.info("✅ 训练流程全部完成！")

if __name__ == "__main__":
    from typing import List
    main()
