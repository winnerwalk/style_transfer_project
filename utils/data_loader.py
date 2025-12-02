import os
import paddle
from paddle.io import Dataset
from PIL import Image
import numpy as np
import random
from PIL import ImageEnhance

class UnpairedDataset(Dataset):
    """无配对图像数据集加载器"""
    
    def __init__(self, root_A, root_B, transform=None, max_size=None):
        """
        Args:
            root_A: A域图像路径
            root_B: B域图像路径
            transform: 图像变换
            max_size: 最大数据量（用于测试）
        """
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform
        
        # 获取所有图像文件
        self.images_A = self._get_images(root_A)
        self.images_B = self._get_images(root_B)
        
        if max_size is not None:
            self.images_A = self.images_A[:max_size]
            self.images_B = self.images_B[:max_size]
        
        self.length = max(len(self.images_A), len(self.images_B))
        
    def _get_images(self, root):
        """获取目录下所有图像文件"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        for ext in extensions:
            images.extend([
                os.path.join(root, f) for f in os.listdir(root)
                if f.lower().endswith(ext.lower())
            ])
        return sorted(images)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 随机选择A域图像
        img_A_path = self.images_A[idx % len(self.images_A)]
        # 随机选择B域图像
        img_B_path = self.images_B[random.randint(0, len(self.images_B) - 1)]
        
        # 加载图像
        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        
        return img_A, img_B


class ImageTransforms:
    """图像变换类"""
    
    def __init__(self, image_size=256, is_train=True):
        self.image_size = image_size
        self.is_train = is_train
        
    def __call__(self, img):
        # 调整大小
        img = img.resize((self.image_size + 30, self.image_size + 30), Image.BICUBIC)
        
        if self.is_train:
            # 随机裁剪
            x = np.random.randint(0, 30)
            y = np.random.randint(0, 30)
            img = img.crop((x, y, x + self.image_size, y + self.image_size))
            
            # 随机水平翻转
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 颜色抖动
            if np.random.random() > 0.5:
                img = self._color_jitter(img)
        else:
            # 中心裁剪
            img = img.crop((15, 15, 15 + self.image_size, 15 + self.image_size))
        
        # 转换为numpy数组
        img = np.array(img, dtype=np.float32)
        
        # 归一化到[-1, 1]
        img = (img / 255.0 - 0.5) / 0.5
        
        # 转换为CHW格式
        img = img.transpose((2, 0, 1))
        
        return paddle.to_tensor(img)
    
    def _color_jitter(self, img):
        """颜色抖动"""
        # 亮度
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        # 对比度
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = ImageEnhance.Contrast(img).enhance(factor)
        
        # 饱和度
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = ImageEnhance.Color(img).enhance(factor)
        
        # 色调
        if np.random.random() > 0.5:
            factor = np.random.uniform(-10, 10)
            img = ImageEnhance.Color(img).enhance(1.0)
            img = np.array(img)
            img = img.astype(np.float32)
            img[:, :, 0] = np.clip(img[:, :, 0] + factor, 0, 255)
            img = Image.fromarray(img.astype(np.uint8))
        
        return img


class ImagePool:
    """图像池，用于稳定训练"""
    
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []
        
    def query(self, images):
        """从池中查询图像"""
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if np.random.random() > 0.5:
                    idx = np.random.randint(0, self.pool_size)
                    return_images.append(self.images[idx].clone())
                    self.images[idx] = image
                else:
                    return_images.append(image)
        
        return paddle.concat(return_images, axis=0)


class DataAugmentation:
    """数据增强类"""
    
    @staticmethod
    def mixup(img1, img2, alpha=0.2):
        """MixUp数据增强"""
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        return mixed_img, lam
    
    @staticmethod
    def cutmix(img1, img2, alpha=1.0):
        """CutMix数据增强"""
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = DataAugmentation._rand_bbox(img1.shape, lam)
        mixed_img = img1.clone()
        mixed_img[:, :, bby1:bby2, bbx1:bbx2] = img2[:, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img1.shape[-1] * img1.shape[-2]))
        
        return mixed_img, lam
    
    @staticmethod
    def _rand_bbox(size, lam):
        """生成随机边界框"""
        W = size[-1]
        H = size[-2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def create_dataloaders(data_root, batch_size=4, image_size=256, max_size=None):
    """创建数据加载器"""
    
    # 训练数据
    train_transform = ImageTransforms(image_size, is_train=True)
    train_dataset = UnpairedDataset(
        os.path.join(data_root, 'trainA'),
        os.path.join(data_root, 'trainB'),
        transform=train_transform,
        max_size=max_size
    )
    
    # 测试数据
    test_transform = ImageTransforms(image_size, is_train=False)
    test_dataset = UnpairedDataset(
        os.path.join(data_root, 'testA'),
        os.path.join(data_root, 'testB'),
        transform=test_transform,
        max_size=max_size
    )
    
    # 创建数据加载器
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    return train_loader, test_loader


def download_dataset(dataset_name='horse2zebra', data_dir='./data'):
    """下载/解压数据集（支持本地文件和网络URL）"""
    import requests
    import zipfile
    import os

    # 数据集URL映射：horse2zebra用本地路径（原始字符串r''避免转义），其他用网络URL
    dataset_urls = {
        'horse2zebra': r'E:\\co-coursework\\style_transfer_project\\horse2zebra.zip',  # 修复转义问题
        'monet2photo': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip',
        'cezanne2photo': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/cezanne2photo.zip',
        'ukiyoe2photo': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ukiyoe2photo.zip',
        'vangogh2photo': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/vangogh2photo.zip'
    }

    if dataset_name not in dataset_urls:
        raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(dataset_urls.keys())}")

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    url = dataset_urls[dataset_name]
    zip_path = os.path.join(data_dir, f'{dataset_name}.zip')

    # 核心修改：判断是本地文件还是网络URL
    if os.path.exists(url):
        # 情况1：本地已存在ZIP文件，直接复制到data_dir（或直接使用原路径解压）
        print(f"找到本地ZIP文件：{url}")
        # 直接使用本地文件路径作为解压源，避免复制
        local_zip_path = url
    else:
        # 情况2：网络URL，正常下载
        print(f"Downloading {dataset_name} dataset from {url}...")
        response = requests.get(url, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\rProgress: {progress:.1f}%", end='')
        print()
        local_zip_path = zip_path

    # 解压ZIP文件
    print(f"Extracting dataset...")
    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"解压成功！")
    except zipfile.BadZipFile:
        raise Exception(f"错误：{local_zip_path} 不是有效的ZIP文件，请检查文件是否损坏")

    # 可选：删除解压后的ZIP文件（本地文件建议保留，避免误删）
    if not os.path.exists(url):  # 只有网络下载的文件才删除，本地文件不删
        os.remove(zip_path)
        print(f"已删除临时ZIP文件：{zip_path}")

    dataset_path = os.path.join(data_dir, dataset_name)
    print(f"Dataset {dataset_name} 已部署到：{dataset_path}")

    return dataset_path