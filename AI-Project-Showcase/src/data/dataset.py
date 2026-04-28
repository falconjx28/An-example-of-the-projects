"""
数据处理模块

提供数据加载、增强和预处理功能
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MedicalImageDataset(Dataset):
    """医学影像数据集类"""

    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        transform=None,
        image_size: Tuple[int, int] = (640, 640)
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform

        # 加载标注数据
        self.samples = self._load_annotations(annotation_file)

    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """加载标注文件"""
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    annotations.append({
                        'image_path': parts[0],
                        'label': int(parts[1]),
                        'bbox': [float(x) for x in parts[2:6]]
                    })
        return annotations

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 读取图像
        image_path = self.data_dir / sample['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 图像预处理
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = cv2.resize(image, self.image_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'bbox': torch.tensor(sample['bbox'], dtype=torch.float32)
        }


def get_train_transforms(image_size: int = 640) -> A.Compose:
    """获取训练数据增强策略"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=3, p=1),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 640) -> A.Compose:
    """获取验证数据增强策略（仅缩放和归一化）"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloader(
    data_dir: str,
    annotation_file: str,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    mode: str = 'train'
) -> DataLoader:
    """创建数据加载器"""

    transform = get_train_transforms() if mode == 'train' else get_val_transforms()

    dataset = MedicalImageDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )

    return dataloader


# 数据统计分析
class DataStatistics:
    """数据统计分析工具"""

    @staticmethod
    def compute_mean_std(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """计算数据集的均值和标准差"""
        image_files = list(Path(data_dir).rglob('*.jpg'))
        mean = np.zeros(3)
        std = np.zeros(3)
        n_samples = 0

        for img_path in image_files[:1000]:  # 使用前1000张图片估算
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean += img.mean(axis=(0, 1))
            std += img.std(axis=(0, 1))
            n_samples += 1

        mean /= n_samples
        std /= n_samples

        return mean, std

    @staticmethod
    def analyze_class_distribution(annotation_file: str) -> Dict[int, int]:
        """分析类别分布"""
        class_counts = {}
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    label = int(parts[1])
                    class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
