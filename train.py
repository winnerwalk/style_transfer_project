import os
import sys
import time
import paddle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualdl import LogWriter

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cycle_gan_basic import CycleGANBasicModel
from models.cycle_gan_enhanced import CycleGANEnhancedModel
from utils.data_loader import create_dataloaders, download_dataset, ImagePool


class Trainer:
    """训练器类"""
    
    def __init__(self, model_type='basic', config=None):
        self.model_type = model_type
        self.config = config or self._get_default_config()
        
        # 设置设备
        self.device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建模型
        if model_type == 'basic':
            self.model = CycleGANBasicModel(
                lambda_cycle=self.config['lambda_cycle'],
                lambda_identity=self.config['lambda_identity']
            )
        else:
            self.model = CycleGANEnhancedModel(
                lambda_cycle=self.config['lambda_cycle'],
                lambda_identity=self.config['lambda_identity'],
                lambda_perceptual=self.config.get('lambda_perceptual', 1.0)
            )
        
        # 创建优化器
        self._create_optimizers()
        
        # 创建学习率调度器
        self._create_lr_schedulers()
        
        # 创建损失历史记录
        self.loss_history = {
            'G_losses': [],
            'D_losses': [],
            'loss_G_A': [],
            'loss_G_B': [],
            'loss_cycle_A': [],
            'loss_cycle_B': [],
            'loss_D_A': [],
            'loss_D_B': []
        }
        
        # 创建图像池
        self.fake_A_pool = ImagePool(self.config['pool_size'])
        self.fake_B_pool = ImagePool(self.config['pool_size'])
        
        # 创建日志记录器
        self.writer = LogWriter(logdir=self.config['log_dir'])
        
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'batch_size': 4,
            'image_size': 256,
            'epochs': 200,
            'lr': 0.0002,
            'beta1': 0.5,
            'lambda_cycle': 10.0,
            'lambda_identity': 0.5,
            'lambda_perceptual': 1.0,
            'pool_size': 50,
            'save_epoch_freq': 5,
            'log_dir': './logs',
            'checkpoint_dir': './checkpoints',
            'sample_dir': './samples'
        }
    
    def _create_optimizers(self):
        """创建优化器"""
        # 生成器优化器
        self.optimizer_G = paddle.optimizer.Adam(
            learning_rate=self.config['lr'],
            beta1=self.config['beta1'],
            beta2=0.999,
            parameters=list(self.model.netG_A.parameters()) + list(self.model.netG_B.parameters())
        )
        
        # 判别器优化器
        self.optimizer_D = paddle.optimizer.Adam(
            learning_rate=self.config['lr'],
            beta1=self.config['beta1'],
            beta2=0.999,
            parameters=list(self.model.netD_A.parameters()) + list(self.model.netD_B.parameters())
        )
    
    def _create_lr_schedulers(self):
        """创建学习率调度器"""
        def lr_lambda(epoch):
            return 1.0 - max(0, epoch - self.config['epochs'] // 2) / (self.config['epochs'] // 2)
        
        self.lr_scheduler_G = paddle.optimizer.lr.LambdaDecay(
            learning_rate=self.config['lr'],
            lr_lambda=lr_lambda
        )
        self.lr_scheduler_D = paddle.optimizer.lr.LambdaDecay(
            learning_rate=self.config['lr'],
            lr_lambda=lr_lambda
        )
        
        # 更新优化器的学习率
        self.optimizer_G.set_lr_scheduler(self.lr_scheduler_G)
        self.optimizer_D.set_lr_scheduler(self.lr_scheduler_D)
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.netG_A.train()
        self.model.netG_B.train()
        self.model.netD_A.train()
        self.model.netD_B.train()
        
        epoch_losses = {
            'G_loss': 0.0,
            'D_loss': 0.0,
            'loss_G_A': 0.0,
            'loss_G_B': 0.0,
            'loss_cycle_A': 0.0,
            'loss_cycle_B': 0.0,
            'loss_D_A': 0.0,
            'loss_D_B': 0.0
        }
        
        num_batches = len(dataloader)
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (real_A, real_B) in enumerate(pbar):
            # 移动到设备
            real_A = paddle.to_tensor(real_A)
            real_B = paddle.to_tensor(real_B)
            
            # 前向传播
            outputs = self.model.forward(real_A, real_B)
            
            # 训练判别器
            self.optimizer_D.clear_grad()
            
            # 使用图像池
            fake_A_pool = self.fake_A_pool.query(outputs['fake_A'])
            fake_B_pool = self.fake_B_pool.query(outputs['fake_B'])
            
            loss_D, D_losses = self.model.get_discriminator_loss(
                real_A, real_B, fake_A_pool, fake_B_pool
            )
            
            loss_D.backward()
            self.optimizer_D.step()
            
            # 训练生成器
            self.optimizer_G.clear_grad()
            
            loss_G, G_losses = self.model.get_generator_loss(
                real_A, real_B,
                outputs['fake_A'], outputs['fake_B'],
                outputs['rec_A'], outputs['rec_B'],
                outputs['idt_A'], outputs['idt_B']
            )
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # 更新损失统计
            epoch_losses['G_loss'] += loss_G.item()
            epoch_losses['D_loss'] += loss_D.item()
            epoch_losses['loss_G_A'] += G_losses['loss_G_A'].item()
            epoch_losses['loss_G_B'] += G_losses['loss_G_B'].item()
            epoch_losses['loss_cycle_A'] += G_losses['loss_cycle_A'].item()
            epoch_losses['loss_cycle_B'] += G_losses['loss_cycle_B'].item()
            epoch_losses['loss_D_A'] += D_losses['loss_D_A'].item()
            epoch_losses['loss_D_B'] += D_losses['loss_D_B'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}'
            })
            
            # 记录到tensorboard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('train/loss_G', loss_G.item(), global_step)
            self.writer.add_scalar('train/loss_D', loss_D.item(), global_step)
            self.writer.add_scalar('train/loss_G_A', G_losses['loss_G_A'].item(), global_step)
            self.writer.add_scalar('train/loss_G_B', G_losses['loss_G_B'].item(), global_step)
            self.writer.add_scalar('train/loss_cycle_A', G_losses['loss_cycle_A'].item(), global_step)
            self.writer.add_scalar('train/loss_cycle_B', G_losses['loss_cycle_B'].item(), global_step)
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def save_samples(self, dataloader, epoch):
        """保存样本图像"""
        self.model.netG_A.eval()
        self.model.netG_B.eval()
        
        with paddle.no_grad():
            # 获取一批样本
            real_A, real_B = next(iter(dataloader))
            real_A = paddle.to_tensor(real_A[:4])  # 只取4个样本
            real_B = paddle.to_tensor(real_B[:4])
            
            # 生成图像
            outputs = self.model.forward(real_A, real_B)
            
            # 创建样本网格
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            for i in range(4):
                # 原始图像
                real_A_img = self._tensor_to_image(real_A[i])
                real_B_img = self._tensor_to_image(real_B[i])
                
                # 生成图像
                fake_B_img = self._tensor_to_image(outputs['fake_B'][i])
                fake_A_img = self._tensor_to_image(outputs['fake_A'][i])
                
                # 显示图像
                axes[i, 0].imshow(real_A_img)
                axes[i, 0].set_title('Real A')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(fake_B_img)
                axes[i, 1].set_title('Fake B')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(real_B_img)
                axes[i, 2].set_title('Real B')
                axes[i, 2].axis('off')
                
                axes[i, 3].imshow(fake_A_img)
                axes[i, 3].set_title('Fake A')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            sample_path = os.path.join(self.config['sample_dir'], f'epoch_{epoch:03d}.png')
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            plt.savefig(sample_path)
            plt.close()
    
    def _tensor_to_image(self, tensor):
        """将张量转换为图像"""
        img = tensor.numpy().transpose((1, 2, 0))
        img = (img * 0.5 + 0.5) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def save_checkpoint(self, epoch):
        """保存检查点（兼容整数epoch和字符串标识）"""
        checkpoint = {
            'epoch': epoch,
            'netG_A_state_dict': self.model.netG_A.state_dict(),
            'netG_B_state_dict': self.model.netG_B.state_dict(),
            'netD_A_state_dict': self.model.netD_A.state_dict(),
            'netD_B_state_dict': self.model.netD_B.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'loss_history': self.loss_history
        }
        
        # 兼容整数epoch和字符串（如'final'）
        if isinstance(epoch, int):
            # 整数epoch：保留3位补零格式（如5→005）
            checkpoint_filename = f'{self.model_type}_checkpoint_epoch_{epoch:03d}.pdparams'
        else:
            # 字符串epoch：直接拼接（如'final'→xxx_final.pdparams）
            checkpoint_filename = f'{self.model_type}_checkpoint_{epoch}.pdparams'
        
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            checkpoint_filename
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        paddle.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = paddle.load(checkpoint_path)
        
        self.model.netG_A.set_state_dict(checkpoint['netG_A_state_dict'])
        self.model.netG_B.set_state_dict(checkpoint['netG_B_state_dict'])
        self.model.netD_A.set_state_dict(checkpoint['netD_A_state_dict'])
        self.model.netD_B.set_state_dict(checkpoint['netD_B_state_dict'])
        self.optimizer_G.set_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.set_state_dict(checkpoint['optimizer_D_state_dict'])
        
        self.loss_history = checkpoint['loss_history']
        
        return checkpoint['epoch']
    
    def plot_loss_history(self):
        """绘制损失历史"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history['G_losses'], label='Generator Loss')
        plt.plot(self.loss_history['D_losses'], label='Discriminator Loss')
        plt.legend()
        plt.title('Overall Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history['loss_G_A'], label='G_A Loss')
        plt.plot(self.loss_history['loss_G_B'], label='G_B Loss')
        plt.legend()
        plt.title('Generator Adversarial Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 3)
        plt.plot(self.loss_history['loss_cycle_A'], label='Cycle A Loss')
        plt.plot(self.loss_history['loss_cycle_B'], label='Cycle B Loss')
        plt.legend()
        plt.title('Cycle Consistency Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 4)
        plt.plot(self.loss_history['loss_D_A'], label='D_A Loss')
        plt.plot(self.loss_history['loss_D_B'], label='D_B Loss')
        plt.legend()
        plt.title('Discriminator Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        
        # 保存图像
        loss_plot_path = os.path.join(self.config['sample_dir'], 'loss_history.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"Loss history plot saved: {loss_plot_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CycleGAN Model')
    parser.add_argument('--model_type', type=str, choices=['basic', 'enhanced'], 
                       default='basic', help='Model type to train')
    parser.add_argument('--dataset', type=str, default='horse2zebra', 
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, 
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 下载数据集
    data_path = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(data_path):
        print(f"Downloading {args.dataset} dataset...")
        data_path = download_dataset(args.dataset, args.data_dir)
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        data_path, 
        batch_size=args.batch_size,
        image_size=256,
        max_size=1000 if args.model_type == 'enhanced' else None
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # 创建配置
    config = {
        'batch_size': args.batch_size,
        'image_size': 256,
        'epochs': args.epochs,
        'lr': args.lr,
        'beta1': 0.5,
        'lambda_cycle': 10.0,
        'lambda_identity': 0.5,
        'lambda_perceptual': 1.0,
        'pool_size': 50,
        'save_epoch_freq': 5,
        'log_dir': f'./logs/{args.model_type}_{args.dataset}',
        'checkpoint_dir': f'./checkpoints/{args.model_type}_{args.dataset}',
        'sample_dir': f'./samples/{args.model_type}_{args.dataset}'
    }
    
    # 创建训练器
    trainer = Trainer(args.model_type, config)
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed training from epoch {start_epoch}")
    
    # 训练循环
    print(f"Starting training {args.model_type} model for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练一个epoch
        epoch_losses = trainer.train_epoch(train_loader, epoch)
        
        # 更新学习率
        trainer.lr_scheduler_G.step()
        trainer.lr_scheduler_D.step()
        
        # 更新损失历史
        trainer.loss_history['G_losses'].append(epoch_losses['G_loss'])
        trainer.loss_history['D_losses'].append(epoch_losses['D_loss'])
        trainer.loss_history['loss_G_A'].append(epoch_losses['loss_G_A'])
        trainer.loss_history['loss_G_B'].append(epoch_losses['loss_G_B'])
        trainer.loss_history['loss_cycle_A'].append(epoch_losses['loss_cycle_A'])
        trainer.loss_history['loss_cycle_B'].append(epoch_losses['loss_cycle_B'])
        trainer.loss_history['loss_D_A'].append(epoch_losses['loss_D_A'])
        trainer.loss_history['loss_D_B'].append(epoch_losses['loss_D_B'])
        
        # 打印损失
        print(f"Generator Loss: {epoch_losses['G_loss']:.4f}")
        print(f"Discriminator Loss: {epoch_losses['D_loss']:.4f}")
        print(f"Cycle A Loss: {epoch_losses['loss_cycle_A']:.4f}")
        print(f"Cycle B Loss: {epoch_losses['loss_cycle_B']:.4f}")
        
        # 保存样本
        if (epoch + 1) % 5 == 0:
            trainer.save_samples(test_loader, epoch + 1)
        
        # 保存检查点
        if (epoch + 1) % config['save_epoch_freq'] == 0:
            trainer.save_checkpoint(epoch + 1)
    
    # 绘制损失历史
    trainer.plot_loss_history()
    
    # 保存最终模型
    trainer.save_checkpoint('final')
    
    print("Training completed!")


if __name__ == '__main__':
    main()