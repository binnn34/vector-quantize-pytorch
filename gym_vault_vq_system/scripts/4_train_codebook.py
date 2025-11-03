#!/usr/bin/env python3
"""
도마 포즈 VQ-VAE 모델 훈련
- 전처리된 포즈 데이터를 활용한 VQ-VAE 학습
- 코드북 활용도 모니터링
- 실시간 손실 함수 추적
- 모델 체크포인트 저장
"""

import os
import time
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from datetime import datetime

# 로컬 모듈 임포트
from gym_vault_vq_system.vq_model import GymVaultVQVAE


class GymVaultDataset(Dataset):
    """도마 포즈 데이터셋"""

    def __init__(self, poses, augment=True):
        """
        Args:
            poses: (N, 64, 17, 2/3) - 정규화된 포즈 시퀀스들
            augment: 데이터 증강 적용 여부
        """
        self.poses = torch.FloatTensor(poses)
        self.augment = augment

        # 포즈를 (N, 64, 51) 형태로 평면화
        self.poses = self.poses.reshape(self.poses.shape[0], self.poses.shape[1], -1)

        print(f"데이터셋 로드: {self.poses.shape}")

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose_seq = self.poses[idx]  # (64, 51)

        if self.augment and torch.rand(1) > 0.5:
            # 데이터 증강: 노이즈 추가
            noise = torch.randn_like(pose_seq) * 0.01
            pose_seq = pose_seq + noise

        return pose_seq


class VQVAETrainer:
    """VQ-VAE 훈련 관리자"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 초기화
        self.model = GymVaultVQVAE(
            input_dim=config['input_dim'],
            codebook_size=config['codebook_size'],
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)

        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] * 0.01
        )

        # 훈련 상태
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.codebook_usage = []

        # 출력 디렉토리
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"모델 초기화 완료 - 디바이스: {self.device}")
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_data(self, data_path):
        """전처리된 데이터 로드"""
        print(f"데이터 로드: {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # 학습/검증 데이터
        train_poses = data['train_poses']
        test_poses = data['test_poses']

        print(f"학습 데이터: {train_poses.shape}")
        print(f"검증 데이터: {test_poses.shape}")

        # 데이터셋 생성
        train_dataset = GymVaultDataset(train_poses, augment=True)
        val_dataset = GymVaultDataset(test_poses, augment=False)

        # 데이터로더
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return len(train_dataset), len(val_dataset)

    def train_epoch(self):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        recon_loss_sum = 0
        vq_loss_sum = 0
        commitment_loss_sum = 0
        pose_loss_sum = 0

        # 코드북 사용량 추적
        used_codes = set()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")

        for batch_idx, pose_seq in enumerate(pbar):
            pose_seq = pose_seq.to(self.device)  # (B, 64, 51)

            # 순전파
            self.optimizer.zero_grad()

            output = self.model(pose_seq)
            recon_poses = output['reconstructed']
            vq_loss = output['vq_loss']
            commitment_loss = output['commitment_loss']
            indices = output['indices']

            # 손실 계산
            recon_loss = nn.MSELoss()(recon_poses, pose_seq)
            pose_loss = self.model.pose_loss(recon_poses, pose_seq)

            total_batch_loss = (
                recon_loss +
                vq_loss +
                commitment_loss * 0.25 +
                pose_loss * 0.1
            )

            # 역전파
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 통계 업데이트
            total_loss += total_batch_loss.item()
            recon_loss_sum += recon_loss.item()
            vq_loss_sum += vq_loss.item()
            commitment_loss_sum += commitment_loss.item()
            pose_loss_sum += pose_loss.item()

            # 코드북 사용량 추적
            unique_codes = torch.unique(indices).cpu().numpy()
            used_codes.update(unique_codes)

            # 프로그레스 바 업데이트
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{total_batch_loss.item():.4f}",
                    'Recon': f"{recon_loss.item():.4f}",
                    'VQ': f"{vq_loss.item():.4f}",
                    'Codes': f"{len(used_codes)}/{self.config['codebook_size']}"
                })

        # 에포크 평균
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches

        # 코드북 사용률
        usage_rate = len(used_codes) / self.config['codebook_size']

        return {
            'total_loss': avg_loss,
            'recon_loss': recon_loss_sum / num_batches,
            'vq_loss': vq_loss_sum / num_batches,
            'commitment_loss': commitment_loss_sum / num_batches,
            'pose_loss': pose_loss_sum / num_batches,
            'codebook_usage': usage_rate,
            'used_codes': len(used_codes)
        }

    def validate(self):
        """검증"""
        self.model.eval()
        total_loss = 0
        recon_loss_sum = 0
        vq_loss_sum = 0
        commitment_loss_sum = 0
        pose_loss_sum = 0

        used_codes = set()

        with torch.no_grad():
            for pose_seq in tqdm(self.val_loader, desc="Validation"):
                pose_seq = pose_seq.to(self.device)

                output = self.model(pose_seq)
                recon_poses = output['reconstructed']
                vq_loss = output['vq_loss']
                commitment_loss = output['commitment_loss']
                indices = output['indices']

                # 손실 계산
                recon_loss = nn.MSELoss()(recon_poses, pose_seq)
                pose_loss = self.model.pose_loss(recon_poses, pose_seq)

                total_batch_loss = (
                    recon_loss +
                    vq_loss +
                    commitment_loss * 0.25 +
                    pose_loss * 0.1
                )

                total_loss += total_batch_loss.item()
                recon_loss_sum += recon_loss.item()
                vq_loss_sum += vq_loss.item()
                commitment_loss_sum += commitment_loss.item()
                pose_loss_sum += pose_loss.item()

                # 코드북 사용량 추적
                unique_codes = torch.unique(indices).cpu().numpy()
                used_codes.update(unique_codes)

        num_batches = len(self.val_loader)
        usage_rate = len(used_codes) / self.config['codebook_size']

        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': recon_loss_sum / num_batches,
            'vq_loss': vq_loss_sum / num_batches,
            'commitment_loss': commitment_loss_sum / num_batches,
            'pose_loss': pose_loss_sum / num_batches,
            'codebook_usage': usage_rate,
            'used_codes': len(used_codes)
        }

    def save_checkpoint(self, metrics, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'codebook_usage': self.codebook_usage,
            'config': self.config,
            'metrics': metrics
        }

        # 최신 체크포인트
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')

        # 최고 성능 모델
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            print(f"✅ 최고 성능 모델 저장 (검증 손실: {metrics['val']['total_loss']:.4f})")

        # 에포크별 저장 (10에포크마다)
        if (self.epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{self.epoch+1}.pth')

    def plot_training_progress(self):
        """훈련 진행 상황 시각화"""
        if len(self.train_losses) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 손실 곡선
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 코드북 사용률
        axes[0, 1].plot(self.codebook_usage, color='green', linewidth=2)
        axes[0, 1].set_title('Codebook Usage Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Usage Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)

        # 최근 손실 (확대)
        if len(self.train_losses) > 10:
            recent_start = max(0, len(self.train_losses) - 20)
            axes[1, 0].plot(
                range(recent_start, len(self.train_losses)),
                self.train_losses[recent_start:],
                label='Train Loss', color='blue'
            )
            axes[1, 0].plot(
                range(recent_start, len(self.val_losses)),
                self.val_losses[recent_start:],
                label='Val Loss', color='red'
            )
            axes[1, 0].set_title('Recent Training Progress')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # 학습률
        lrs = [self.optimizer.param_groups[0]['lr']]
        if hasattr(self, '_lr_history'):
            lrs = self._lr_history
        axes[1, 1].plot(lrs, color='orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, data_path):
        """전체 훈련 프로세스"""
        print("=== 도마 포즈 VQ-VAE 훈련 시작 ===")

        # 데이터 로드
        train_size, val_size = self.load_data(data_path)
        print(f"학습 샘플: {train_size}, 검증 샘플: {val_size}")

        # 훈련 루프
        start_time = time.time()

        for epoch in range(self.config['epochs']):
            self.epoch = epoch

            # 훈련
            train_metrics = self.train_epoch()

            # 검증
            val_metrics = self.validate()

            # 메트릭 저장
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            self.codebook_usage.append(val_metrics['codebook_usage'])

            # 학습률 기록
            if not hasattr(self, '_lr_history'):
                self._lr_history = []
            self._lr_history.append(self.optimizer.param_groups[0]['lr'])

            # 스케줄러 업데이트
            self.scheduler.step()

            # 최고 성능 확인
            is_best = val_metrics['total_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['total_loss']

            # 체크포인트 저장
            metrics = {'train': train_metrics, 'val': val_metrics}
            self.save_checkpoint(metrics, is_best)

            # 진행 상황 출력
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} "
                  f"({elapsed/60:.1f}분)")
            print(f"Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"코드북 사용률: {val_metrics['codebook_usage']:.1%} "
                  f"({val_metrics['used_codes']}/{self.config['codebook_size']})")
            print(f"학습률: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 시각화 업데이트 (10에포크마다)
            if (epoch + 1) % 10 == 0:
                self.plot_training_progress()

            # 조기 종료 검사
            if len(self.val_losses) > 20:
                recent_improvement = (
                    min(self.val_losses[-20:-10]) -
                    min(self.val_losses[-10:])
                )
                if recent_improvement < 0.0001:
                    print(f"조기 종료: 최근 10 에포크 동안 개선 없음")
                    break

        # 훈련 완료
        total_time = time.time() - start_time
        print(f"\n✅ 훈련 완료! 총 시간: {total_time/3600:.1f}시간")
        print(f"최고 검증 손실: {self.best_loss:.4f}")

        # 최종 시각화
        self.plot_training_progress()

        # 훈련 요약 저장
        summary = {
            'total_epochs': self.epoch + 1,
            'total_time_hours': total_time / 3600,
            'best_val_loss': self.best_loss,
            'final_codebook_usage': self.codebook_usage[-1],
            'config': self.config
        }

        with open(self.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def main():
    """메인 실행 함수"""

    # 훈련 설정
    config = {
        # 모델 파라미터
        'input_dim': 51,  # 17 관절 × 3 좌표
        'codebook_size': 512,
        'latent_dim': 128,
        'hidden_dim': 256,

        # 훈련 파라미터
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 200,

        # 경로
        'data_path': 'D:/vector-quantize-pytorch/gym_vault_vq_system/processed_data.pkl',
        'output_dir': 'D:/vector-quantize-pytorch/gym_vault_vq_system/checkpoints'
    }

    # 출력 디렉토리 생성
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    # 설정 저장
    with open(Path(config['output_dir']) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("=== 도마 포즈 VQ-VAE 훈련 설정 ===")
    print(f"코드북 크기: {config['codebook_size']}")
    print(f"잠재 차원: {config['latent_dim']}")
    print(f"배치 크기: {config['batch_size']}")
    print(f"학습률: {config['learning_rate']}")
    print(f"최대 에포크: {config['epochs']}")

    # 데이터 확인
    data_path = Path(config['data_path'])
    if not data_path.exists():
        print(f"❌ 데이터 파일이 없습니다: {data_path}")
        print("먼저 2_data_preprocessing.py를 실행하세요.")
        return

    # 트레이너 초기화 및 훈련
    trainer = VQVAETrainer(config)
    trainer.train(config['data_path'])

    print(f"\n✅ 훈련 완료!")
    print(f"체크포인트 저장 위치: {config['output_dir']}")
    print(f"최고 성능 모델: best_model.pth")


if __name__ == "__main__":
    main()