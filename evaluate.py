import os
import sys
import paddle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cycle_gan_basic import CycleGANBasicModel
from models.cycle_gan_enhanced import CycleGANEnhancedModel
from utils.data_loader import create_dataloaders


class Evaluator:
    """评估器类"""
    
    def __init__(self, model_type='basic', device='gpu'):
        self.model_type = model_type
        self.device = paddle.set_device(device if paddle.is_compiled_with_cuda() else 'cpu')
        
        # 创建模型
        if model_type == 'basic':
            self.model = CycleGANBasicModel()
        else:
            self.model = CycleGANEnhancedModel()
        
        # 初始化LPIPS模型
        self.lpips_model = lpips.LPIPS(net='alex')
        
    def load_model(self, checkpoint_path):
        """加载模型"""
        checkpoint = paddle.load(checkpoint_path)
        self.model.netG_A.set_state_dict(checkpoint['netG_A_state_dict'])
        self.model.netG_B.set_state_dict(checkpoint['netG_B_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    
    def generate_samples(self, dataloader, num_samples=100, output_dir='./results'):
        """生成样本"""
        self.model.netG_A.eval()
        self.model.netG_B.eval()
        
        os.makedirs(output_dir, exist_ok=True)
        
        generated_images = {
            'real_A': [],
            'fake_B': [],
            'real_B': [],
            'fake_A': []
        }
        
        count = 0
        with paddle.no_grad():
            for real_A, real_B in dataloader:
                if count >= num_samples:
                    break
                    
                real_A = paddle.to_tensor(real_A)
                real_B = paddle.to_tensor(real_B)
                
                # 生成图像
                outputs = self.model.forward(real_A, real_B)
                
                # 保存生成结果
                for i in range(len(real_A)):
                    if count >= num_samples:
                        break
                    
                    # 转换为图像
                    real_A_img = self._tensor_to_image(real_A[i])
                    fake_B_img = self._tensor_to_image(outputs['fake_B'][i])
                    real_B_img = self._tensor_to_image(real_B[i])
                    fake_A_img = self._tensor_to_image(outputs['fake_A'][i])
                    
                    # 保存图像
                    Image.fromarray(real_A_img).save(
                        os.path.join(output_dir, f'real_A_{count:03d}.png'))
                    Image.fromarray(fake_B_img).save(
                        os.path.join(output_dir, f'fake_B_{count:03d}.png'))
                    Image.fromarray(real_B_img).save(
                        os.path.join(output_dir, f'real_B_{count:03d}.png'))
                    Image.fromarray(fake_A_img).save(
                        os.path.join(output_dir, f'fake_A_{count:03d}.png'))
                    
                    # 添加到列表
                    generated_images['real_A'].append(real_A_img)
                    generated_images['fake_B'].append(fake_B_img)
                    generated_images['real_B'].append(real_B_img)
                    generated_images['fake_A'].append(fake_A_img)
                    
                    count += 1
        
        print(f"Generated {count} samples to {output_dir}")
        return generated_images
    
    def calculate_metrics(self, generated_images):
        """计算评估指标"""
        metrics = {}
        n_samples = len(generated_images['real_A'])
        
        # PSNR和SSIM
        psnr_A2B = []
        psnr_B2A = []
        ssim_A2B = []
        ssim_B2A = []
        
        # LPIPS
        lpips_A2B = []
        lpips_B2A = []
        
        for i in range(n_samples):
            # A->B方向
            real_A_gray = np.mean(generated_images['real_A'][i], axis=2)
            fake_B_gray = np.mean(generated_images['fake_B'][i], axis=2)
            
            psnr_A2B.append(psnr(real_A_gray, fake_B_gray))
            ssim_A2B.append(ssim(real_A_gray, fake_B_gray))
            
            # B->A方向
            real_B_gray = np.mean(generated_images['real_B'][i], axis=2)
            fake_A_gray = np.mean(generated_images['fake_A'][i], axis=2)
            
            psnr_B2A.append(psnr(real_B_gray, fake_A_gray))
            ssim_B2A.append(ssim(real_B_gray, fake_A_gray))
            
            # LPIPS (需要转换为PyTorch张量)
            real_A_tensor = self._numpy_to_torch(generated_images['real_A'][i])
            fake_B_tensor = self._numpy_to_torch(generated_images['fake_B'][i])
            real_B_tensor = self._numpy_to_torch(generated_images['real_B'][i])
            fake_A_tensor = self._numpy_to_torch(generated_images['fake_A'][i])
            
            with paddle.no_grad():
                lpips_A2B.append(self.lpips_model(real_A_tensor, fake_B_tensor).item())
                lpips_B2A.append(self.lpips_model(real_B_tensor, fake_A_tensor).item())
        
        metrics = {
            'PSNR_A2B': np.mean(psnr_A2B),
            'PSNR_B2A': np.mean(psnr_B2A),
            'SSIM_A2B': np.mean(ssim_A2B),
            'SSIM_B2A': np.mean(ssim_B2A),
            'LPIPS_A2B': np.mean(lpips_A2B),
            'LPIPS_B2A': np.mean(lpips_B2A)
        }
        
        return metrics
    
    def create_comparison_grid(self, generated_images, output_path, num_samples=16):
        """创建对比网格"""
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < num_samples:
                    # 显示A->B转换
                    axes[i, j].imshow(generated_images['real_A'][idx])
                    axes[i, j].set_title('Real A')
                    axes[i, j].axis('off')
                
                # 显示生成结果
                if idx + 16 < len(generated_images['fake_B']):
                    axes[i, j].imshow(generated_images['fake_B'][idx + 16])
                    axes[i, j].set_title('Fake B')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Comparison grid saved to {output_path}")
    
    def _tensor_to_image(self, tensor):
        """将张量转换为图像"""
        img = tensor.numpy().transpose((1, 2, 0))
        img = (img * 0.5 + 0.5) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def _numpy_to_torch(self, numpy_img):
        """将numpy图像转换为PyTorch张量"""
        # 转换为RGB格式
        if numpy_img.shape[2] == 3:
            img = numpy_img
        else:
            img = np.stack([numpy_img] * 3, axis=2)
        
        # 转换为张量
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # 归一化
        img = img.transpose((2, 0, 1))
        
        return paddle.to_tensor(img).unsqueeze(0)


def compare_models(model1_path, model2_path, data_path, output_dir='./comparison'):
    """比较两个模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据加载器
    _, test_loader = create_dataloaders(data_path, batch_size=4, image_size=256)
    
    # 评估基础模型
    print("Evaluating basic model...")
    basic_evaluator = Evaluator('basic')
    basic_evaluator.load_model(model1_path)
    basic_images = basic_evaluator.generate_samples(
        test_loader, num_samples=100, 
        output_dir=os.path.join(output_dir, 'basic_results')
    )
    basic_metrics = basic_evaluator.calculate_metrics(basic_images)
    
    # 评估增强模型
    print("Evaluating enhanced model...")
    enhanced_evaluator = Evaluator('enhanced')
    enhanced_evaluator.load_model(model2_path)
    enhanced_images = enhanced_evaluator.generate_samples(
        test_loader, num_samples=100,
        output_dir=os.path.join(output_dir, 'enhanced_results')
    )
    enhanced_metrics = enhanced_evaluator.calculate_metrics(enhanced_images)
    
    # 创建对比报告
    comparison_report = {
        'Basic Model': basic_metrics,
        'Enhanced Model': enhanced_metrics,
        'Improvement': {}
    }
    
    # 计算改进幅度
    for metric in basic_metrics:
        if metric in enhanced_metrics:
            improvement = ((enhanced_metrics[metric] - basic_metrics[metric]) / 
                          basic_metrics[metric]) * 100
            comparison_report['Improvement'][metric] = improvement
    
    # 保存报告
    import json
    with open(os.path.join(output_dir, 'comparison_report.json'), 'w') as f:
        json.dump(comparison_report, f, indent=4)
    
    # 创建可视化对比
    create_visual_comparison(basic_images, enhanced_images, output_dir)
    
    return comparison_report


def create_visual_comparison(basic_images, enhanced_images, output_dir):
    """创建可视化对比"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 选择样本进行展示
    sample_idx = 0
    
    # 第一行：原始图像
    axes[0, 0].imshow(basic_images['real_A'][sample_idx])
    axes[0, 0].set_title('Real A (Original)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(basic_images['real_B'][sample_idx])
    axes[0, 1].set_title('Real B (Original)')
    axes[0, 1].axis('off')
    
    # 第二行：基础模型结果
    axes[1, 0].imshow(basic_images['fake_B'][sample_idx])
    axes[1, 0].set_title('Basic Model - Fake B')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(basic_images['fake_A'][sample_idx])
    axes[1, 1].set_title('Basic Model - Fake A')
    axes[1, 1].axis('off')
    
    # 第三行：增强模型结果
    axes[2, 0].imshow(enhanced_images['fake_B'][sample_idx])
    axes[2, 0].set_title('Enhanced Model - Fake B')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(enhanced_images['fake_A'][sample_idx])
    axes[2, 1].set_title('Enhanced Model - Fake A')
    axes[2, 1].axis('off')
    
    # 添加空白子图
    for i in range(3):
        for j in range(2, 4):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visual_comparison.png'))
    plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN Models')
    parser.add_argument('--model_type', type=str, choices=['basic', 'enhanced'],
                       required=True, help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = Evaluator(args.model_type)
    evaluator.load_model(args.checkpoint)
    
    # 创建数据加载器
    _, test_loader = create_dataloaders(args.data_path, batch_size=4, image_size=256)
    
    # 生成样本
    generated_images = evaluator.generate_samples(
        test_loader, 
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    # 计算指标
    metrics = evaluator.calculate_metrics(generated_images)
    
    # 打印结果
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 创建对比网格
    evaluator.create_comparison_grid(
        generated_images, 
        os.path.join(args.output_dir, 'comparison_grid.png')
    )
    
    # 保存指标
    import json
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()