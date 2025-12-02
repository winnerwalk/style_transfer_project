#!/usr/bin/env python3
"""
生成式神经网络图像风格混合项目主入口
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.data_loader import download_dataset
from train import Trainer
from evaluate import Evaluator, compare_models


def setup_environment():
    """设置环境"""
    # 创建必要的目录
    directories = ['data', 'checkpoints', 'logs', 'samples', 'results', 'output']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    
    print("Environment setup completed!")


def download_all_datasets():
    """下载所有支持的数据集"""
    datasets = [
        'horse2zebra',
        'monet2photo', 
        'cezanne2photo',
        'ukiyoe2photo',
        'vangogh2photo'
    ]
    
    for dataset in datasets:
        try:
            print(f"Downloading {dataset}...")
            download_dataset(dataset, './data')
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")


def train_both_models(dataset='horse2zebra', epochs=200):
    """训练两种模型进行对比"""
    print(f"Training both models on {dataset} dataset...")
    
    # 训练基础模型
    print("\\n=== Training Basic Model ===")
    basic_trainer = Trainer('basic', {
        'batch_size': 4,
        'epochs': epochs,
        'lr': 0.0002,
        'lambda_cycle': 10.0,
        'lambda_identity': 0.5,
        'log_dir': f'./logs/basic_{dataset}',
        'checkpoint_dir': f'./checkpoints/basic_{dataset}',
        'sample_dir': f'./samples/basic_{dataset}'
    })
    
    # 训练优化模型
    print("\\n=== Training Enhanced Model ===")
    enhanced_trainer = Trainer('enhanced', {
        'batch_size': 4,
        'epochs': epochs,
        'lr': 0.0002,
        'lambda_cycle': 10.0,
        'lambda_identity': 0.5,
        'lambda_perceptual': 1.0,
        'log_dir': f'./logs/enhanced_{dataset}',
        'checkpoint_dir': f'./checkpoints/enhanced_{dataset}',
        'sample_dir': f'./samples/enhanced_{dataset}'
    })
    
    return basic_trainer, enhanced_trainer


def run_full_comparison(dataset='horse2zebra'):
    """运行完整的模型对比实验"""
    data_path = f'./data/{dataset}'
    
    # 检查数据是否存在
    if not os.path.exists(data_path):
        print(f"Dataset {dataset} not found. Downloading...")
        download_dataset(dataset, './data')
    
    # 训练模型（如果检查点不存在）
    basic_checkpoint = f'./checkpoints/basic_{dataset}/final.pdparams'
    enhanced_checkpoint = f'./checkpoints/enhanced_{dataset}/final.pdparams'
    
    if not os.path.exists(basic_checkpoint) or not os.path.exists(enhanced_checkpoint):
        print("Training models...")
        train_both_models(dataset)
    
    # 运行对比评估
    print("\\n=== Running Model Comparison ===")
    comparison_results = compare_models(
        basic_checkpoint,
        enhanced_checkpoint,
        data_path,
        './comparison_results'
    )
    
    # 保存对比结果
    with open('./comparison_results/full_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print("Comparison completed!")
    return comparison_results


def generate_demo_images(model_type='enhanced', dataset='horse2zebra'):
    """生成演示图像"""
    checkpoint_path = f'./checkpoints/{model_type}_{dataset}/final.pdparams'
    data_path = f'./data/{dataset}'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    evaluator = Evaluator(model_type)
    evaluator.load_model(checkpoint_path)
    
    # 创建简单的演示数据加载器
    from utils.data_loader import create_dataloaders
    _, test_loader = create_dataloaders(data_path, batch_size=4, image_size=256, max_size=20)
    
    # 生成样本
    generated_images = evaluator.generate_samples(
        test_loader,
        num_samples=16,
        output_dir='./demo_results'
    )
    
    print("Demo images generated!")
    return generated_images


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Style Transfer Project')
    parser.add_argument('--setup', action='store_true', help='Setup environment')
    parser.add_argument('--download', action='store_true', help='Download datasets')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--compare', action='store_true', help='Run comparison')
    parser.add_argument('--demo', action='store_true', help='Generate demo images')
    parser.add_argument('--dataset', type=str, default='horse2zebra', help='Dataset name')
    parser.add_argument('--model', type=str, choices=['basic', 'enhanced'], default='enhanced', help='Model type')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    
    args = parser.parse_args()
    
    # 设置环境
    if args.setup:
        setup_environment()
        return
    
    # 下载数据集
    if args.download:
        download_all_datasets()
        return
    
    # 训练模型
    if args.train:
        train_both_models(args.dataset, args.epochs)
        return
    
    # 运行对比
    if args.compare:
        run_full_comparison(args.dataset)
        return
    
    # 生成演示
    if args.demo:
        generate_demo_images(args.model, args.dataset)
        return
    
    # 默认行为：运行完整流程
    print("Running complete style transfer pipeline...")
    setup_environment()
    
    # 检查数据集
    data_path = f'./data/{args.dataset}'
    if not os.path.exists(data_path):
        print(f"Dataset {args.dataset} not found. Downloading...")
        download_dataset(args.dataset, './data')
    
    # 运行对比实验
    results = run_full_comparison(args.dataset)
    
    # 生成演示图像
    generate_demo_images('enhanced', args.dataset)
    
    print("\\n=== Pipeline Completed ===")
    print("Results saved in:")
    print("- ./comparison_results/")
    print("- ./demo_results/")
    print("- ./output/")


if __name__ == '__main__':
    main()