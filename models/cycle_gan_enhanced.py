import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class AttentionBlock(nn.Layer):
    """自注意力机制模块"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.query_conv = nn.Conv2D(channels, channels // 8, 1)
        self.key_conv = nn.Conv2D(channels, channels // 8, 1)
        self.value_conv = nn.Conv2D(channels, channels, 1)
        self.gamma = paddle.create_parameter(
            shape=[1], dtype='float32', 
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 生成query, key, value
        proj_query = self.query_conv(x).reshape([batch_size, -1, height * width])
        proj_key = self.key_conv(x).reshape([batch_size, -1, height * width])
        proj_value = self.value_conv(x).reshape([batch_size, -1, height * width])
        
        # 计算注意力权重
        attention = paddle.bmm(proj_query.transpose([0, 2, 1]), proj_key)
        attention = F.softmax(attention, axis=-1)
        
        # 应用注意力
        out = paddle.bmm(proj_value, attention.transpose([0, 2, 1]))
        out = out.reshape([batch_size, channels, height, width])
        
        # 残差连接
        out = self.gamma * out + x
        return out


class EnhancedResidualBlock(nn.Layer):
    """增强型残差块，包含注意力机制"""
    
    def __init__(self, in_channels, use_attention=True):
        super(EnhancedResidualBlock, self).__init__()
        self.use_attention = use_attention
        
        # 保持卷积层的padding设置，确保尺寸不变
        self.conv1 = nn.Conv2D(in_channels, in_channels, 3, padding=1, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2D(in_channels)
        self.conv2 = nn.Conv2D(in_channels, in_channels, 3, padding=1, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2D(in_channels)
        
        if use_attention:
            self.attention = AttentionBlock(in_channels)
            
        # 自适应权重
        self.alpha = paddle.create_parameter(
            shape=[1], dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.1)
        )
        
    def forward(self, x):
        residual = x  # 保存原始输入（尺寸不变）
        
        # 第一个卷积（移除手动pad，依赖conv1自带的padding）
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        
        # 第二个卷积（移除手动pad，依赖conv2自带的padding）
        out = self.conv2(out)
        out = self.norm2(out)
        
        # 注意力机制
        if self.use_attention:
            out = self.attention(out)
        
        # 自适应残差连接（现在尺寸匹配）
        out = self.alpha * out + residual
        
        return out


class EnhancedGenerator(nn.Layer):
    """增强型生成器网络（移除密集连接块）"""
    
    def __init__(self, input_nc=3, output_nc=3, n_blocks=9, use_attention=True):
        super(EnhancedGenerator, self).__init__()
        self.use_attention = use_attention
        
        # 初始卷积层
        self.conv1 = nn.Conv2D(input_nc, 64, 7, padding=3, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2D(64)
        
        # 下采样层
        self.conv2 = nn.Conv2D(64, 128, 4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2D(128)
        self.conv3 = nn.Conv2D(128, 256, 4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2D(256)
        
        # 增强型残差块（直接使用下采样后的256通道）
        self.res_blocks = nn.LayerList([
            EnhancedResidualBlock(256, use_attention) 
            for _ in range(n_blocks)
        ])
        
        # 注意力机制
        if use_attention:
            self.mid_attention = AttentionBlock(256)  # 通道数改为256
        
        # 上采样层
        self.conv_transpose1 = nn.Conv2DTranspose(256, 128, 4, stride=2, padding=1, output_padding=0)
        self.norm4 = nn.InstanceNorm2D(128)
        self.conv_transpose2 = nn.Conv2DTranspose(128, 64, 4, stride=2, padding=1, output_padding=0)
        self.norm5 = nn.InstanceNorm2D(64)
        
        # 输出层
        self.conv_out1 = nn.Conv2D(64, 32, 3, padding=1, padding_mode='reflect')
        self.norm_out = nn.InstanceNorm2D(32)
        self.conv_out2 = nn.Conv2D(32, output_nc, 7, padding=3, padding_mode='reflect')
        
        # 跳连连接权重
        self.skip_weight = paddle.create_parameter(
            shape=[1], dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.5)
        )
        
    def forward(self, x):
        # 保存输入用于跳连连接
        input_skip = x
        
        # 初始卷积：移除手动pad，依赖conv1自带的padding=3和padding_mode='reflect'
        x = self.conv1(x)  # 不再手动pad，直接使用卷积层自带的padding
        x = self.norm1(x)
        x = F.relu(x)
        skip1 = x  # 此时skip1尺寸正确
        
        # 下采样（保持不变）
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        skip2 = x  # 保存跳连连接
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        
        # 增强型残差块
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 中间注意力（保持不变）
        if self.use_attention:
            x = self.mid_attention(x)
        
        # 上采样（保持不变）
        x = self.conv_transpose1(x)
        x = self.norm4(x)
        x = F.relu(x)
        # 添加跳连连接（此时x与skip2尺寸匹配）
        x = self.skip_weight * x + (1 - self.skip_weight) * skip2
        
        x = self.conv_transpose2(x)
        x = self.norm5(x)
        x = F.relu(x)
        # 添加跳连连接（现在x与skip1尺寸匹配）
        x = self.skip_weight * x + (1 - self.skip_weight) * skip1
        
        # 输出层：同样移除多余的手动pad，依赖卷积层自带padding
        x = self.conv_out1(x)  # conv_out1已设置padding=1
        x = self.norm_out(x)
        x = F.relu(x)
        
        x = self.conv_out2(x)  # conv_out2已设置padding=3
        
        # 添加输入跳连连接
        x = 0.9 * F.tanh(x) + 0.1 * input_skip
        
        return x


class MultiScaleDiscriminator(nn.Layer):
    """多尺度判别器（保持不变）"""
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3, num_D=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_D = num_D
        
        self.discriminators = nn.LayerList()
        for i in range(num_D):
            self.discriminators.append(
                SingleScaleDiscriminator(input_nc, ndf, n_layers)
            )
        
        self.downsample = nn.AvgPool2D(3, stride=2, padding=1)
        
    def forward(self, x):
        results = []
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            results.append(discriminator(x))
        return results


class SingleScaleDiscriminator(nn.Layer):
    """单尺度判别器（保持不变）"""
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(SingleScaleDiscriminator, self).__init__()
        
        # 初始层
        self.conv1 = nn.Conv2D(input_nc, ndf, 4, stride=2, padding=1)
        
        # 中间层 - 使用谱归一化提高稳定性
        self.layers = nn.LayerList()
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2D(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1)
                    ),
                    nn.InstanceNorm2D(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # 最后一层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.conv_last = nn.utils.spectral_norm(
            nn.Conv2D(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1)
        )
        self.norm_last = nn.InstanceNorm2D(ndf * nf_mult)
        
        # 输出层
        self.conv_out = nn.Conv2D(ndf * nf_mult, 1, 4, stride=1, padding=1)
        
        # 特征金字塔
        self.feature_pyramid = nn.LayerList([
            nn.Conv2D(ndf * mult, 256, 1) for mult in [1, 2, 4, 8]
        ])
        
    def forward(self, x):
        features = []
        
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        features.append(self.feature_pyramid[0](x))
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.feature_pyramid) - 1:
                features.append(self.feature_pyramid[i+1](x))
        
        x = self.conv_last(x)
        x = self.norm_last(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv_out(x)
        
        return x, features


class CycleGANEnhancedModel:
    """增强型CycleGAN模型（保持不变）"""
    
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5, lambda_perceptual=1.0):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_perceptual = lambda_perceptual
        
        # 创建生成器和判别器
        self.netG_A = EnhancedGenerator()  # A -> B
        self.netG_B = EnhancedGenerator()  # B -> A
        self.netD_A = MultiScaleDiscriminator()  # 判别A域
        self.netD_B = MultiScaleDiscriminator()  # 判别B域
        
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
        # 多尺度对抗损失
        loss_G_A = 0
        loss_G_B = 0
        
        # 判别器A的损失
        fake_A_preds = self.netD_A(fake_A)
        for pred in fake_A_preds:
            loss_G_A += F.mse_loss(pred[0], paddle.ones_like(pred[0]))
        loss_G_A /= len(fake_A_preds)
        
        # 判别器B的损失
        fake_B_preds = self.netD_B(fake_B)
        for pred in fake_B_preds:
            loss_G_B += F.mse_loss(pred[0], paddle.ones_like(pred[0]))
        loss_G_B /= len(fake_B_preds)
        
        # 循环一致性损失
        loss_cycle_A = F.l1_loss(rec_A, real_A) * self.lambda_cycle
        loss_cycle_B = F.l1_loss(rec_B, real_B) * self.lambda_cycle
        
        # 身份映射损失
        loss_idt_A = F.l1_loss(idt_A, real_A) * self.lambda_identity
        loss_idt_B = F.l1_loss(idt_B, real_B) * self.lambda_identity
        
        # 感知损失（基于判别器特征）
        loss_perceptual = self._compute_perceptual_loss(real_A, real_B, fake_A, fake_B)
        
        # 总生成器损失
        loss_G = (loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + 
                 loss_idt_A + loss_idt_B + loss_perceptual)
        
        return loss_G, {
            'loss_G_A': loss_G_A,
            'loss_G_B': loss_G_B,
            'loss_cycle_A': loss_cycle_A,
            'loss_cycle_B': loss_cycle_B,
            'loss_idt_A': loss_idt_A,
            'loss_idt_B': loss_idt_B,
            'loss_perceptual': loss_perceptual
        }
        
    def get_discriminator_loss(self, real_A, real_B, fake_A, fake_B):
        """计算判别器损失"""
        loss_D_A = 0
        loss_D_B = 0
        
        # 判别器A的损失
        real_A_preds = self.netD_A(real_A)
        fake_A_preds = self.netD_A(fake_A.detach())
        
        for real_pred, fake_pred in zip(real_A_preds, fake_A_preds):
            loss_D_real = F.mse_loss(real_pred[0], paddle.ones_like(real_pred[0]))
            loss_D_fake = F.mse_loss(fake_pred[0], paddle.zeros_like(fake_pred[0]))
            loss_D_A += (loss_D_real + loss_D_fake) * 0.5
        loss_D_A /= len(real_A_preds)
        
        # 判别器B的损失
        real_B_preds = self.netD_B(real_B)
        fake_B_preds = self.netD_B(fake_B.detach())
        
        for real_pred, fake_pred in zip(real_B_preds, fake_B_preds):
            loss_D_real = F.mse_loss(real_pred[0], paddle.ones_like(real_pred[0]))
            loss_D_fake = F.mse_loss(fake_pred[0], paddle.zeros_like(fake_pred[0]))
            loss_D_B += (loss_D_real + loss_D_fake) * 0.5
        loss_D_B /= len(real_B_preds)
        
        # 总判别器损失
        loss_D = loss_D_A + loss_D_B
        
        return loss_D, {
            'loss_D_A': loss_D_A,
            'loss_D_B': loss_D_B
        }
    
    def _compute_perceptual_loss(self, real_A, real_B, fake_A, fake_B):
        """计算感知损失"""
        # 使用判别器的中间特征作为感知损失
        with paddle.no_grad():
            _, real_A_features = self.netD_A(real_A)[0]
            _, real_B_features = self.netD_B(real_B)[0]
            _, fake_A_features = self.netD_A(fake_A)[0]
            _, fake_B_features = self.netD_B(fake_B)[0]
        
        loss_perceptual = 0
        for real_feat, fake_feat in zip(real_A_features, fake_A_features):
            loss_perceptual += F.l1_loss(fake_feat, real_feat.detach())
        for real_feat, fake_feat in zip(real_B_features, fake_B_features):
            loss_perceptual += F.l1_loss(fake_feat, real_feat.detach())
        
        return loss_perceptual * self.lambda_perceptual