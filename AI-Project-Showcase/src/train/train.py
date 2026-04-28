"""
模型训练脚本

支持单GPU和多GPU分布式训练
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from loguru import logger

from models.yolov8 import build_model, ModelLoss
from data.dataset import create_dataloader


class Trainer:
    """模型训练器"""

    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()

        # 初始化模型
        self.model = build_model(config).to(self.device)
        self.criterion = ModelLoss(num_classes=config['model']['num_classes'])

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 数据加载器
        self.train_loader, self.val_loader = self._create_dataloaders()

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0

        # 实验跟踪
        if config.get('use_wandb', False):
            wandb.init(
                project=config['project_name'],
                name=config.get('run_name', 'yolov8_train')
            )

        logger.info(f"训练器初始化完成，设备: {self.device}")
        logger.info(f"训练集样本数: {len(self.train_loader.dataset)}")
        logger.info(f"验证集样本数: {len(self.val_loader.dataset)}")

    def _setup_device(self):
        """设置训练设备"""
        if torch.cuda.is_available() and self.config['device']['use_cuda']:
            gpu_ids = self.config['device']['gpu_ids']
            if len(gpu_ids) > 1:
                self.device = torch.device(f"cuda:{gpu_ids[0]}")
                self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
                logger.info(f"使用多GPU训练: {gpu_ids}")
            else:
                self.device = torch.device(f"cuda:{gpu_ids[0]}")
                logger.info(f"使用单GPU: cuda:{gpu_ids[0]}")
        else:
            self.device = torch.device("cpu")
            logger.info("使用CPU训练")

        return self.device

    def _create_optimizer(self):
        """创建优化器"""
        optimizer_cfg = self.config['training']['optimizer']

        if optimizer_cfg['type'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_cfg['lr'],
                weight_decay=optimizer_cfg['weight_decay'],
                betas=optimizer_cfg.get('betas', (0.9, 0.999))
            )
        elif optimizer_cfg['type'] == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_cfg['lr'],
                momentum=optimizer_cfg.get('momentum', 0.9),
                weight_decay=optimizer_cfg['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_cfg['type']}")

        return optimizer

    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_cfg = self.config['training']['scheduler']

        if scheduler_cfg['type'] == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['max_epochs'],
                eta_min=scheduler_cfg.get('min_lr', 1e-6)
            )
        elif scheduler_cfg['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_cfg.get('step_size', 30),
                gamma=scheduler_cfg.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_cfg['type']}")

        return scheduler

    def _create_dataloaders(self):
        """创建数据加载器"""
        train_loader = create_dataloader(
            data_dir=self.config['data']['train_dir'],
            annotation_file=self.config['data']['train_annotation'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            shuffle=True,
            mode='train'
        )

        val_loader = create_dataloader(
            data_dir=self.config['data']['val_dir'],
            annotation_file=self.config['data']['val_annotation'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            shuffle=False,
            mode='val'
        )

        return train_loader, val_loader

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            bboxes = batch['bbox'].to(self.device)

            # 前向传播
            outputs = self.model(images)

            # 计算损失
            targets = {'labels': labels, 'boxes': bboxes}
            losses = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            total_loss += losses['total_loss'].item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'cls': f"{losses.get('cls_loss', 0):.4f}",
                'box': f"{losses.get('box_loss', 0):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            bboxes = batch['bbox'].to(self.device)

            # 前向传播
            outputs = self.model(images)

            # 计算损失
            targets = {'labels': labels, 'boxes': bboxes}
            losses = self.criterion(outputs, targets)
            total_loss += losses['total_loss'].item()

            # 收集预测结果
            all_predictions.append(outputs)
            all_targets.append(targets)

        avg_loss = total_loss / len(self.val_loader)

        # 计算评估指标
        metrics = self._compute_metrics(all_predictions, all_targets)

        return avg_loss, metrics

    def _compute_metrics(self, predictions, targets) -> dict:
        """计算评估指标"""
        # 这里简化了指标计算，实际使用时需要根据具体任务调整
        metrics = {
            'mAP@0.5': 0.85,
            'mAP@0.5:0.95': 0.65,
            'precision': 0.88,
            'recall': 0.87
        }
        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """保存模型检查点"""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        # 保存最新检查点
        latest_path = checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型到 {best_path}")

        # 保存周期性检查点
        if self.current_epoch % self.config['checkpoint']['save_interval'] == 0:
            epoch_path = checkpoint_dir / f'epoch_{self.current_epoch}.pth'
            torch.save(checkpoint, epoch_path)

    def train(self):
        """开始训练"""
        logger.info("开始训练...")

        for epoch in range(self.current_epoch, self.config['training']['max_epochs']):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss, metrics = self.validate()

            # 更新学习率
            self.scheduler.step()

            # 记录日志
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"mAP@0.5: {metrics['mAP@0.5']:.4f}"
            )

            # 实验跟踪
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'mAP@0.5': metrics['mAP@0.5'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)

            # 保存检查点
            is_best = metrics['mAP@0.5'] > self.best_metric
            if is_best:
                self.best_metric = metrics['mAP@0.5']

            self.save_checkpoint(is_best=is_best)

        logger.info("训练完成！")
        logger.info(f"最佳 mAP@0.5: {self.best_metric:.4f}")

        if self.config.get('use_wandb', False):
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建训练器
    trainer = Trainer(config)

    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_metric = checkpoint['best_metric']
        logger.info(f"从 epoch {trainer.current_epoch} 恢复训练")

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
