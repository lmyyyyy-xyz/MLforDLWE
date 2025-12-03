#!/usr/bin/env python3
"""
LWE Transformer 主程序

使用Transformer模型求解decisional-LWE问题的主程序入口。
整合了数据生成、模型训练、评估和可视化所有功能。
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入自定义模块
from models import create_standard_lwe_transformer, create_model_from_config
from data import LWEDataset, create_data_loaders, create_dataset_from_generator
from training import setup_training, CrossValidationTrainer
from utils import (
    generate_lwe_problem, create_standard_evaluator, setup_experiment_logging,
    set_random_seed, print_system_info, estimate_training_time,
    print_comparison_results, plot_model_comparison
)
from utils.data_generator import calculate_lwe_parameters, validate_lwe_parameters


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LWE Transformer 训练程序")
    
    # 数据参数
    parser.add_argument("--n", type=int, default=256, help="LWE维度")
    parser.add_argument("--q", type=int, default=8380417, help="模数")
    parser.add_argument("--num-samples", type=int, default=400000, help="总样本数量")
    parser.add_argument("--error-std", type=float, default=0.5, help="误差标准差")
    parser.add_argument("--secret-dist", type=str, default="sparse", 
                       choices=["uniform", "binary", "ternary", "sparse"],
                       help="秘密向量分布")
    parser.add_argument("--error-dist", type=str, default="gaussian",
                       choices=["gaussian", "uniform", "binary"],
                       help="误差分布")
    
    # 模型参数
    parser.add_argument("--model-type", type=str, default="standard",
                       choices=["standard", "large", "small"],
                       help="模型类型")
    parser.add_argument("--d-model", type=int, default=256, help="模型维度")
    parser.add_argument("--nhead", type=int, default=4, help="注意力头数")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="前馈网络维度")
        
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练周期数") 
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="测试集比例")
    
    # 实验设置
    parser.add_argument("--experiment-name", type=str, default="lwe_experiment",
                       help="实验名称")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use-cuda", action="store_true", help="使用GPU")
    parser.add_argument("--cross-validation", action="store_true", help="使用交叉验证")
    parser.add_argument("--k-folds", type=int, default=5, help="交叉验证折数")
    
    # 功能选项
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument("--visualize", action="store_true", help="可视化结果")
    parser.add_argument("--compare-models", action="store_true", help="比较多个模型")
    parser.add_argument("--parameter-sweep", action="store_true", help="参数扫描")
    
    # 文件路径
    parser.add_argument("--load-model", type=str, help="加载预训练模型")
    parser.add_argument("--save-dir", type=str, default="results", help="结果保存目录")
    parser.add_argument("--data-file", type=str, help="使用已有的数据文件")
    
    return parser.parse_args()


def setup_environment(args):
    """设置运行环境"""
    print("=" * 60)
    print("LWE TRANSFORMER 实验设置")
    print("=" * 60)
    
    # 设置随机种子
    set_random_seed(args.random_seed)
    
    # 打印系统信息
    print_system_info()
    
    # 设置设备
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    # 创建保存目录
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    return device, save_dir


def generate_data(args):
    """生成或加载数据"""
    print("\n" + "=" * 40)
    print("数据生成")
    print("=" * 40)
    
    if args.data_file and Path(args.data_file).exists():
        # 从文件加载数据
        from utils.data_generator import load_lwe_dataset
        lwe_samples, uniform_samples, metadata = load_lwe_dataset(args.data_file)
        print(f"从文件加载数据: {args.data_file}")
        if metadata:
            print(f"数据参数: {metadata}")
    else:
        # 生成新数据
        print(f"生成LWE数据:")
        print(f"  维度 n: {args.n}")
        print(f"  模数 q: {args.q}")
        print(f"  样本数: {args.num_samples}")
        print(f"  误差标准差: {args.error_std}")
        print(f"  秘密分布: {args.secret_dist}")
        print(f"  误差分布: {args.error_dist}")
        
        lwe_samples, uniform_samples, secret = generate_lwe_problem(
            n=args.n,
            q=args.q,
            num_samples=args.num_samples,
            error_std=args.error_std,
            secret_distribution=args.secret_dist,
            error_distribution=args.error_dist,
            return_secret=True
        )
        
        # 验证参数安全性
        validation = validate_lwe_parameters(args.n, args.q, args.error_std)
        print("参数安全性验证:")
        for check, result in validation.items():
            status = "✓" if result else "✗"
            print(f"  {check}: {status}")
        
        # 保存数据（可选）
        if args.data_file:
            from utils.data_generator import save_lwe_dataset
            metadata = {
                'n': args.n,
                'q': args.q,
                'error_std': args.error_std,
                'secret_dist': args.secret_dist,
                'error_dist': args.error_dist,
                'num_samples': args.num_samples
            }
            save_lwe_dataset((lwe_samples, uniform_samples), args.data_file, metadata)
    
    return lwe_samples, uniform_samples


def create_model(args, device):
    """创建模型"""
    print("\n" + "=" * 40)
    print("模型创建")
    print("=" * 40)
    
    if args.load_model:
        # 加载预训练模型
        print(f"加载预训练模型: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        
        # 从检查点重建模型
        model_config = checkpoint.get('config', {}).get('model_config', {})
        model = create_model_from_config(
            vocab_size=args.q,
            max_seq_len=args.n + 1,
            n_classes=2,
            **model_config
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"模型已加载，训练周期: {checkpoint.get('epoch', '未知')}")
    else:
        # 创建新模型
        if args.model_type in ["standard", "large", "small"]:
            print(f"创建{args.model_type}模型")
            model = create_model_from_config(
                vocab_size=args.q,
                max_seq_len=args.n + 1,
                n_classes=2,
                model_type=args.model_type
            )
        else:
            # 自定义模型参数
            print("创建自定义模型")
            model = create_standard_lwe_transformer(
                vocab_size=args.q,
                n=args.n,
                n_classes=2
            )
        
        model.to(device)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def train_model(args, model, dataset, device, save_dir):
    """训练模型"""
    print("\n" + "=" * 40)
    print("模型训练")
    print("=" * 40)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        balanced_sampling=True,
        shuffle=True,
        random_seed=args.random_seed
    )
    
    print(f"数据分割:")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本") 
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    
    # 设置实验日志
    experiment_config = {
        'n': args.n,
        'q': args.q,
        'error_std': args.error_std,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio
    }
    
    logger = setup_experiment_logging(
        args.experiment_name,
        config=experiment_config,
        log_dir=str(save_dir)
    )
    
    # 记录模型信息
    logger.log_model_info(model)
    
    # 估计训练时间
    estimate_training_time(model, train_loader, epochs=args.epochs, device=device)
    
    if args.cross_validation:
        # 交叉验证
        print("开始交叉验证...")
        from models import LWETransformer
        
        cv_trainer = CrossValidationTrainer(
            model_class=LWETransformer,
            dataset=dataset,
            k_folds=args.k_folds, 
            config={
                'model_args': {
                    'vocab_size': args.q,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers,
                    'dim_feedforward': args.dim_feedforward,
                    'max_seq_len': args.n + 1,
                    'n_classes': 2
                },
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size
            }
        )
        
        cv_results = cv_trainer.run_cross_validation()
        
        # 记录交叉验证结果
        logger.log_metrics({
            'cv_mean_accuracy': cv_results['mean_val_accuracy'],
            'cv_std_accuracy': cv_results['std_val_accuracy'],
            'cv_mean_loss': cv_results['mean_val_loss'],
            'cv_std_loss': cv_results['std_val_loss']
        })
        
        # 保存交叉验证结果
        import json
        cv_file = save_dir / "cross_validation_results.json"
        with open(cv_file, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        print(f"交叉验证结果已保存到: {cv_file}")
        
        return cv_results, test_loader
    
    else:
        # 常规训练
        # 设置训练器
        training_config = {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'early_stopping_patience': 10,
            'log_dir': str(save_dir / "logs"),
            'checkpoint_dir': str(save_dir / "checkpoints")
        }
        
        trainer = setup_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config_type="default",
            custom_config=training_config
        )
        
        # 开始训练
        history = trainer.train()
        
        # 记录最终指标
        final_metrics = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_accuracy': trainer.best_val_accuracy,
            'best_val_loss': trainer.best_val_loss
        }
        logger.log_metrics(final_metrics)
        
        # 保存最终模型
        model_save_path = save_dir / "final_model.pth"
        trainer.save_checkpoint(best=True)
        print(f"最终模型已保存到: {model_save_path}")
        
        # 完成实验记录
        logger.finish_experiment(summary=final_metrics)
        
        return trainer, test_loader


def evaluate_model(args, model, test_loader, device, save_dir):
    """评估模型"""
    print("\n" + "=" * 40)
    print("模型评估")
    print("=" * 40)
    
    # 创建评估器
    evaluator = create_standard_evaluator(model, device)
    
    # 在测试集上评估
    test_metrics = evaluator.evaluate(test_loader)
    
    print("测试集结果:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  LWE优势: {test_metrics['advantage']:.4f}")
    print(f"  F1分数: {test_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    # 绘制混淆矩阵
    cm_path = save_dir / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(
        test_loader,
        save_path=str(cm_path),
        title=f"Confusion Matrix (Advantage: {test_metrics['advantage']:.4f})"
    )
    
    # 计算置信区间
    confidence_intervals = evaluator.compute_confidence_intervals(test_loader)
    print("\n置信区间 (95%):")
    for metric, (mean, low, high) in confidence_intervals.items():
        print(f"  {metric}: {mean:.4f} [{low:.4f}, {high:.4f}]")
    
    # 保存评估结果
    import json
    eval_file = save_dir / "evaluation_results.json"
    results = {
        'test_metrics': test_metrics,
        'confidence_intervals': confidence_intervals
    }
    with open(eval_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"评估结果已保存到: {eval_file}")
    
    return test_metrics


def compare_multiple_models(args, dataset, device, save_dir):
    """比较多个模型"""
    print("\n" + "=" * 40)
    print("模型比较")
    print("=" * 40)
    
    # 创建测试数据加载器
    _, _, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        train_ratio=0.8,  # 使用更多数据测试
        val_ratio=0.1,
        test_ratio=0.1,
        balanced_sampling=False,
        shuffle=True,
        random_seed=args.random_seed
    )
    
    # 定义要比较的模型
    model_types = ["small", "standard", "large"]
    models = {}
    
    for model_type in model_types:
        print(f"创建 {model_type} 模型...")
        model = create_model_from_config(
            vocab_size=args.q,
            max_seq_len=args.n + 1,
            n_classes=2,
            model_type=model_type
        )
        models[model_type] = model
    
    # 比较模型性能
    from utils.evaluator import compare_models
    comparison_results = compare_models(models, test_loader, device)
    
    # 打印比较结果
    print_comparison_results(comparison_results)
    
    # 绘制比较图
    comparison_plot_path = save_dir / "model_comparison.png"
    plot_model_comparison(
        comparison_results,
        metrics=['accuracy', 'advantage', 'f1_score'],
        save_path=str(comparison_plot_path)
    )
    
    # 保存比较结果
    import json
    comparison_file = save_dir / "model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=float)
    
    print(f"模型比较结果已保存到: {comparison_file}")
    
    return comparison_results


def run_parameter_sweep(args, device, save_dir):
    """运行参数扫描"""
    print("\n" + "=" * 40)
    print("参数扫描")
    print("=" * 40)
    
    # 定义参数范围
    n_range = [5, 10, 15, 20]
    q_range = [97, 257, 521]
    error_std_range = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    # 生成参数扫描数据集
    from utils.data_generator import generate_parameter_sweep_datasets
    param_datasets = generate_parameter_sweep_datasets(
        n_range=n_range,
        q_range=q_range,
        error_std_range=error_std_range,
        samples_per_config=1000
    )
    
    print(f"生成了 {len(param_datasets)} 个参数配置的数据集")
    
    # 创建标准模型
    model = create_standard_lwe_transformer(vocab_size=max(q_range), n=max(n_range))
    
    # 评估参数敏感性
    evaluator = create_standard_evaluator(model, device)
    sensitivity_results = evaluator.evaluate_parameter_sensitivity(
        param_datasets,
        batch_size=args.batch_size
    )
    
    # 绘制敏感性分析图
    for param in ['n', 'q', 'error_std']:
        plot_path = save_dir / f"parameter_sensitivity_{param}.png"
        evaluator.plot_parameter_sensitivity(
            sensitivity_results,
            parameter=param,
            metric='advantage',
            save_path=str(plot_path)
        )
    
    # 保存敏感性分析结果
    import json
    sensitivity_file = save_dir / "parameter_sensitivity.json"
    
    # 转换numpy类型以便JSON序列化
    serializable_results = {}
    for key, result in sensitivity_results.items():
        serializable_results[key] = {
            'metrics': {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                       for k, v in result['metrics'].items()},
            'parameters': result['parameters']
        }
    
    with open(sensitivity_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"参数敏感性分析结果已保存到: {sensitivity_file}")
    
    return sensitivity_results


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果没有指定任何操作，默认进行训练和评估
    if not any([args.train, args.evaluate, args.compare_models, args.parameter_sweep]):
        args.train = True
        args.evaluate = True
    
    try:
        # 设置环境
        device, save_dir = setup_environment(args)
        
        # 生成或加载数据
        lwe_samples, uniform_samples = generate_data(args)
        
        # 创建数据集
        dataset = LWEDataset(lwe_samples, uniform_samples)
        print(f"数据集大小: {len(dataset)}")
        
        # 模型训练
        trainer = None
        test_loader = None
        
        if args.train:
            model = create_model(args, device)
            training_result, test_loader = train_model(args, model, dataset, device, save_dir)
            
            if isinstance(training_result, tuple):  # 交叉验证
                cv_results, test_loader = training_result
            else:  # 常规训练
                trainer = training_result
        
        # 模型评估
        if args.evaluate:
            if trainer is not None:
                # 使用训练器的模型进行评估
                test_metrics = evaluate_model(args, trainer.model, test_loader, device, save_dir)
            else:
                # 创建新模型进行评估
                model = create_model(args, device)
                if test_loader is None:
                    _, _, test_loader = create_data_loaders(
                        dataset=dataset,
                        batch_size=args.batch_size,
                        train_ratio=0.7,
                        val_ratio=0.15,
                        test_ratio=0.15
                    )
                test_metrics = evaluate_model(args, model, test_loader, device, save_dir)
        
        # 模型比较
        if args.compare_models:
            comparison_results = compare_multiple_models(args, dataset, device, save_dir)
        
        # 参数扫描
        if args.parameter_sweep:
            sensitivity_results = run_parameter_sweep(args, device, save_dir)
        
        print("\n" + "=" * 60)
        print("实验完成!")
        print("=" * 60)
        print(f"所有结果已保存到: {save_dir}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()