"""
模型定义模块

实现 YOLOv8 检测模型的核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class ConvBlock(nn.Module):
    """标准卷积块：Conv -> BN -> SiLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BottleneckBlock(nn.Module):
    """残差瓶颈块"""

    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBlock(hidden_channels, out_channels, 3, 1)

        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2fBlock(nn.Module):
    """C2f 特征融合块"""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1, shortcut: bool = False):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 1, 1)
        self.conv2 = ConvBlock(in_channels, out_channels, 1, 1)

        self.blocks = nn.ModuleList([
            BottleneckBlock(out_channels, out_channels, shortcut) for _ in range(num_blocks)
        ])

        self.conv_out = ConvBlock((num_blocks + 2) * out_channels, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        features = [x1, x2]
        for block in self.blocks:
            features.append(block(features[-1]))

        return self.conv_out(torch.cat(features, dim=1))


class SPPFBlock(nn.Module):
    """空间金字塔池化模块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLOv8Backbone(nn.Module):
    """YOLOv8 骨干网络"""

    def __init__(self, in_channels: int = 3, num_classes: int = 80):
        super().__init__()

        # Stem
        self.stem = ConvBlock(in_channels, 64, 3, 2)

        # Stage 1-4
        self.stage1 = nn.Sequential(
            ConvBlock(64, 128, 3, 2),
            C2fBlock(128, 128, num_blocks=3, shortcut=True)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(128, 256, 3, 2),
            C2fBlock(256, 256, num_blocks=6, shortcut=True)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(256, 512, 3, 2),
            C2fBlock(512, 512, num_blocks=6, shortcut=True)
        )

        self.stage4 = nn.Sequential(
            ConvBlock(512, 1024, 3, 2),
            C2fBlock(1024, 1024, num_blocks=3, shortcut=True),
            SPPFBlock(1024, 1024)
        )

        self.out_channels = [128, 256, 512, 1024]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return c2, c3, c4, c5


class DetectionHead(nn.Module):
    """检测头"""

    def __init__(self, num_classes: int = 80, in_channels: List[int] = [256, 512, 1024]):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 4  # 分类 + 边界框(x,y,w,h)

        # 三个检测层
        self.detect1 = nn.Conv2d(in_channels[0], self.num_outputs * 3, 1)
        self.detect2 = nn.Conv2d(in_channels[1], self.num_outputs * 3, 1)
        self.detect3 = nn.Conv2d(in_channels[2], self.num_outputs * 3, 1)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
        outputs = []

        for i, feat in enumerate(features):
            outputs.append(self.__getattr__(f'detect{i+1}')(feat))

        return outputs


class YOLOv8(nn.Module):
    """YOLOv8 完整模型"""

    def __init__(self, num_classes: int = 80, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # 骨干网络
        self.backbone = YOLOv8Backbone(in_channels, num_classes)

        # 检测头
        self.head = DetectionHead(num_classes, self.backbone.out_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs

    def load_pretrained(self, pretrained_path: str):
        """加载预训练权重"""
        state_dict = torch.load(pretrained_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")


class ModelLoss(nn.Module):
    """模型损失函数"""

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predictions: List[torch.Tensor], targets: Dict) -> Dict[str, torch.Tensor]:
        total_loss = 0
        loss_dict = {}

        for pred in predictions:
            # 分类损失
            cls_pred = pred[:, :self.num_classes]
            cls_target = targets['labels']
            cls_loss = self.bce(cls_pred, cls_target)

            # 边界框损失
            box_pred = pred[:, self.num_classes:]
            box_target = targets['boxes']
            box_loss = self.mse(box_pred, box_target)

            total_loss += cls_loss + box_loss

        loss_dict['total_loss'] = total_loss
        loss_dict['cls_loss'] = cls_loss
        loss_dict['box_loss'] = box_loss

        return loss_dict


def build_model(config: Dict) -> YOLOv8:
    """模型构建函数"""
    model = YOLOv8(
        num_classes=config['model']['num_classes'],
        in_channels=3
    )

    if config['model'].get('pretrained', False):
        # 加载预训练权重
        pretrained_path = config['model'].get('pretrained_path', '')
        if pretrained_path:
            model.load_pretrained(pretrained_path)

    return model
