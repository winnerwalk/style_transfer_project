import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class ResidualBlock(nn.Layer):
    """残差块，用于生成器网络（修复尺寸不匹配）"""
    
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, in_channels, 3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2D(in_channels, in_channels, 3, padding=1, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2D(in_channels)
        self.norm2 = nn.InstanceNorm2D(in_channels)
        
    def forward(self, x):
        residual = x  # 保存原始输入（尺寸不变）
        # 去掉重复的 F.pad！conv1 已自带 padding=1 + reflect 填充
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        # 去掉重复的 F.pad！conv2 已自带 padding=1 + reflect 填充
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual  # 现在尺寸一致（均为原始尺寸）


class BasicGenerator(nn.Layer):
    """基础生成器网络 - 使用ResNet架构（修复尺寸不匹配）"""
    
    def __init__(self, input_nc=3, output_nc=3, n_blocks=9):
        super(BasicGenerator, self).__init__()
        
        # 初始卷积层（已自带padding=3+reflect填充，无需手动pad）
        self.conv1 = nn.Conv2D(input_nc, 64, 7, padding=3, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2D(64)
        
        # 下采样层
        self.conv2 = nn.Conv2D(64, 128, 3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2D(128)
        self.conv3 = nn.Conv2D(128, 256, 3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2D(256)
        
        # 残差块
        self.res_blocks = nn.LayerList([ResidualBlock(256) for _ in range(n_blocks)])
        
        # 上采样层
        self.conv_transpose1 = nn.Conv2DTranspose(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.norm4 = nn.InstanceNorm2D(128)
        self.conv_transpose2 = nn.Conv2DTranspose(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.norm5 = nn.InstanceNorm2D(64)
        
        # 输出层（已自带padding=3+reflect填充，无需手动pad）
        self.conv_out = nn.Conv2D(64, output_nc, 7, padding=3, padding_mode='reflect')
        
    def forward(self, x):
        # 初始卷积：删除重复的手动F.pad！conv1已自带padding=3+reflect填充
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        
        # 下采样
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        
        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 上采样
        x = self.conv_transpose1(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.conv_transpose2(x)
        x = self.norm5(x)
        x = F.relu(x)
        
        # 输出层：删除重复的手动F.pad！conv_out已自带padding=3+reflect填充
        x = self.conv_out(x)
        x = F.tanh(x)
        
        return x


class BasicDiscriminator(nn.Layer):
    """基础判别器网络 - 使用PatchGAN架构"""
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(BasicDiscriminator, self).__init__()
        
        # 初始层
        self.conv1 = nn.Conv2D(input_nc, ndf, 4, stride=2, padding=1)
        
        # 中间层
        self.layers = nn.LayerList()
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2D(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1),
                    nn.InstanceNorm2D(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # 最后一层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv_last = nn.Conv2D(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1)
        self.norm_last = nn.InstanceNorm2D(ndf * nf_mult)
        
        # 输出层
        self.conv_out = nn.Conv2D(ndf * nf_mult, 1, 4, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.conv_last(x)
        x = self.norm_last(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv_out(x)
        
        return x


class CycleGANBasicModel:
    """基础CycleGAN模型"""
    
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # 创建生成器和判别器
        self.netG_A = BasicGenerator()  # A -> B
        self.netG_B = BasicGenerator()  # B -> A
        self.netD_A = BasicDiscriminator()  # 判别A域
        self.netD_B = BasicDiscriminator()  # 判别B域
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for net in [self.netG_A, self.netG_B, self.netD_A, self.netD_B]:
            for m in net.sublayers():
                if isinstance(m, nn.Conv2D):
                    m.weight_attr = paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.02)
                    )
                elif isinstance(m, (nn.BatchNorm2D, nn.InstanceNorm2D)):
                    m.weight_attr = paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(mean=1.0, std=0.02)
                    )
                    m.bias_attr = paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(value=0.0)
                    )
    
    def forward(self, real_A, real_B):
        """前向传播"""
        # 生成假图像
        fake_B = self.netG_A(real_A)
        fake_A = self.netG_B(real_B)
        
        # 循环重建
        rec_A = self.netG_B(fake_B)
        rec_B = self.netG_A(fake_A)
        
        # 身份映射
        idt_A = self.netG_B(real_A)
        idt_B = self.netG_A(real_B)
        
        return {
            'fake_B': fake_B,
            'fake_A': fake_A,
            'rec_A': rec_A,
            'rec_B': rec_B,
            'idt_A': idt_A,
            'idt_B': idt_B
        }
        
    def get_generator_loss(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B):
        """计算生成器损失"""
        # 对抗损失
        pred_fake_B = self.netD_B(fake_B)
        loss_G_A = F.mse_loss(pred_fake_B, paddle.ones_like(pred_fake_B))
        
        pred_fake_A = self.netD_A(fake_A)
        loss_G_B = F.mse_loss(pred_fake_A, paddle.ones_like(pred_fake_A))
        
        # 循环一致性损失
        loss_cycle_A = F.l1_loss(rec_A, real_A) * self.lambda_cycle
        loss_cycle_B = F.l1_loss(rec_B, real_B) * self.lambda_cycle
        
        # 身份映射损失
        loss_idt_A = F.l1_loss(idt_A, real_A) * self.lambda_identity
        loss_idt_B = F.l1_loss(idt_B, real_B) * self.lambda_identity
        
        # 总生成器损失
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        
        return loss_G, {
            'loss_G_A': loss_G_A,
            'loss_G_B': loss_G_B,
            'loss_cycle_A': loss_cycle_A,
            'loss_cycle_B': loss_cycle_B,
            'loss_idt_A': loss_idt_A,
            'loss_idt_B': loss_idt_B
        }
        
    def get_discriminator_loss(self, real_A, real_B, fake_A, fake_B):
        """计算判别器损失"""
        # 判别器A的损失
        pred_real_A = self.netD_A(real_A)
        loss_D_real_A = F.mse_loss(pred_real_A, paddle.ones_like(pred_real_A))
        
        pred_fake_A = self.netD_A(fake_A.detach())
        loss_D_fake_A = F.mse_loss(pred_fake_A, paddle.zeros_like(pred_fake_A))
        
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        
        # 判别器B的损失
        pred_real_B = self.netD_B(real_B)
        loss_D_real_B = F.mse_loss(pred_real_B, paddle.ones_like(pred_real_B))
        
        pred_fake_B = self.netD_B(fake_B.detach())
        loss_D_fake_B = F.mse_loss(pred_fake_B, paddle.zeros_like(pred_fake_B))
        
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        
        # 总判别器损失
        loss_D = loss_D_A + loss_D_B
        
        return loss_D, {
            'loss_D_A': loss_D_A,
            'loss_D_B': loss_D_B
        }