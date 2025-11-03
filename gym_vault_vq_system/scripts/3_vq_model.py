#!/usr/bin/env python3
"""
체조 도마 자세 평가를 위한 VQ-VAE 모델
- 포즈 시퀀스 인코더/디코더
- 체조 동작에 특화된 아키텍처
- 코드북 기반 자세 양자화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# vector_quantize_pytorch 모듈 경로 추가
sys.path.append('D:/vector-quantize-pytorch')
from vector_quantize_pytorch import VectorQuantize


class PoseEncoder(nn.Module):
    """포즈 시퀀스 인코더"""

    def __init__(self, input_dim=51, hidden_dim=256, latent_dim=128, num_layers=3):
        """
        Args:
            input_dim: 입력 차원 (17관절 × 3좌표 = 51)
            hidden_dim: 은닉층 차원
            latent_dim: 잠재 표현 차원
            num_layers: 인코더 레이어 수
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 포즈 전처리: 관절별 특징 추출
        self.pose_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 시간적 특징 추출 (1D Convolution)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # BiLSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # 최종 인코딩
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            encoded: (batch_size, sequence_length, latent_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 포즈 임베딩
        x = self.pose_embedding(x)  # (B, T, hidden_dim)

        # 2. 시간적 컨볼루션
        x_conv = x.transpose(1, 2)  # (B, hidden_dim, T)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, T, hidden_dim)

        # 3. 잔차 연결
        x = x + x_conv

        # 4. LSTM 인코딩
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim)

        # 5. 최종 투영
        encoded = self.final_projection(lstm_out)

        return encoded


class PoseDecoder(nn.Module):
    """포즈 시퀀스 디코더"""

    def __init__(self, latent_dim=128, hidden_dim=256, output_dim=51, num_layers=3):
        """
        Args:
            latent_dim: 잠재 표현 차원
            hidden_dim: 은닉층 차원
            output_dim: 출력 차원 (17관절 × 3좌표 = 51)
            num_layers: 디코더 레이어 수
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 잠재 표현 확장
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # BiLSTM for sequential decoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # 시간적 디컨볼루션
        self.temporal_deconv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 최종 포즈 복원
        self.pose_reconstruction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, latent_dim)
        Returns:
            reconstructed: (batch_size, sequence_length, output_dim)
        """
        # 1. 잠재 표현 확장
        x = self.latent_projection(x)  # (B, T, hidden_dim)

        # 2. LSTM 디코딩
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim)

        # 3. 시간적 디컨볼루션
        x_deconv = lstm_out.transpose(1, 2)  # (B, hidden_dim, T)
        x_deconv = self.temporal_deconv(x_deconv)
        x_deconv = x_deconv.transpose(1, 2)  # (B, T, hidden_dim)

        # 4. 잔차 연결
        x = lstm_out + x_deconv

        # 5. 포즈 복원
        reconstructed = self.pose_reconstruction(x)

        return reconstructed


class GymVaultVQVAE(nn.Module):
    """체조 도마 자세 평가용 VQ-VAE"""

    def __init__(
        self,
        input_dim=51,           # 17관절 × 3좌표
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        latent_dim=128,
        codebook_size=512,      # 코드북 크기
        codebook_dim=64,        # 코드북 차원
        commitment_weight=0.25,
        decay=0.99,
        encoder_layers=3,
        decoder_layers=3,
        use_ema=True,
        threshold_ema_dead_code=2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        # 인코더
        self.encoder = PoseEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers
        )

        # Vector Quantizer
        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            decay=decay,
            commitment_weight=commitment_weight,
            kmeans_init=True,              # K-means 초기화
            threshold_ema_dead_code=threshold_ema_dead_code,
            ema_update=use_ema,
            accept_image_fmap=False,        # 시퀀스 데이터
            channel_last=True              # (B, T, D) 형태
        )

        # 디코더
        self.decoder = PoseDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=input_dim,
            num_layers=decoder_layers
        )

        # 추가 손실 함수들
        self.commitment_weight = commitment_weight

    def encode(self, x):
        """포즈 시퀀스 인코딩"""
        return self.encoder(x)

    def quantize(self, z):
        """잠재 표현 양자화"""
        return self.vq(z)

    def decode(self, z_q):
        """양자화된 표현 디코딩"""
        return self.decoder(z_q)

    def forward(self, x, return_indices=False, return_loss_breakdown=False):
        """
        Args:
            x: (batch_size, sequence_length, input_dim) 포즈 시퀀스
        Returns:
            x_recon: 복원된 포즈 시퀀스
            vq_loss: VQ 손실
            indices: 코드북 인덱스 (optional)
            loss_breakdown: 상세 손실 (optional)
        """
        # 1. 인코딩
        z_e = self.encode(x)  # (B, T, latent_dim)

        # 2. 양자화
        z_q, indices, vq_loss = self.quantize(z_e)

        # 3. 디코딩
        x_recon = self.decode(z_q)

        # 4. 추가 손실 계산
        reconstruction_loss = F.mse_loss(x_recon, x)

        # 포즈 특화 손실들
        pose_losses = self.calculate_pose_specific_losses(x, x_recon)

        total_loss = reconstruction_loss + vq_loss + pose_losses['total']

        results = {
            'x_recon': x_recon,
            'total_loss': total_loss,
            'vq_loss': vq_loss,
            'reconstruction_loss': reconstruction_loss
        }

        if return_indices:
            results['indices'] = indices
            results['z_e'] = z_e
            results['z_q'] = z_q

        if return_loss_breakdown:
            results['loss_breakdown'] = {
                'reconstruction': reconstruction_loss,
                'vq': vq_loss,
                'pose_specific': pose_losses,
                'total': total_loss
            }

        return results

    def calculate_pose_specific_losses(self, x_true, x_recon, weight=0.1):
        """체조 포즈 특화 손실 함수들"""
        losses = {}

        # 1. 관절별 가중 손실 (중요한 관절에 더 큰 가중치)
        joint_weights = self.get_joint_importance_weights()
        weighted_joint_loss = self.weighted_joint_loss(x_true, x_recon, joint_weights)
        losses['weighted_joint'] = weighted_joint_loss * weight

        # 2. 본 길이 보존 손실
        bone_length_loss = self.bone_length_preservation_loss(x_true, x_recon)
        losses['bone_length'] = bone_length_loss * weight * 0.5

        # 3. 시간적 일관성 손실 (프레임 간 부드러움)
        temporal_consistency_loss = self.temporal_consistency_loss(x_true, x_recon)
        losses['temporal_consistency'] = temporal_consistency_loss * weight * 0.3

        # 4. 대칭성 손실 (좌우 대칭)
        symmetry_loss = self.symmetry_loss(x_true, x_recon)
        losses['symmetry'] = symmetry_loss * weight * 0.2

        # 총 포즈 특화 손실
        losses['total'] = sum(losses.values())

        return losses

    def get_joint_importance_weights(self):
        """체조에서 중요한 관절에 대한 가중치"""
        # COCO 17 키포인트 가중치 (체조 중심)
        weights = torch.ones(17)

        # 코어 관절 (높은 가중치)
        weights[5] = 2.0   # left_shoulder
        weights[6] = 2.0   # right_shoulder
        weights[11] = 2.0  # left_hip
        weights[12] = 2.0  # right_hip

        # 사지 관절 (중간 가중치)
        weights[7] = 1.5   # left_elbow
        weights[8] = 1.5   # right_elbow
        weights[9] = 1.5   # left_wrist
        weights[10] = 1.5  # right_wrist
        weights[13] = 1.5  # left_knee
        weights[14] = 1.5  # right_knee
        weights[15] = 1.5  # left_ankle
        weights[16] = 1.5  # right_ankle

        # 머리 관절 (낮은 가중치)
        weights[0] = 0.5   # nose
        weights[1] = 0.3   # left_eye
        weights[2] = 0.3   # right_eye
        weights[3] = 0.3   # left_ear
        weights[4] = 0.3   # right_ear

        return weights.to(next(self.parameters()).device)

    def weighted_joint_loss(self, x_true, x_recon, weights):
        """관절별 가중 손실"""
        # x_true, x_recon: (B, T, 51) -> (B, T, 17, 3)으로 reshape
        batch_size, seq_len, _ = x_true.shape

        x_true_joints = x_true.view(batch_size, seq_len, 17, 3)
        x_recon_joints = x_recon.view(batch_size, seq_len, 17, 3)

        # 관절별 MSE 계산
        joint_errors = torch.mean((x_true_joints - x_recon_joints) ** 2, dim=-1)  # (B, T, 17)

        # 가중치 적용
        weighted_errors = joint_errors * weights.unsqueeze(0).unsqueeze(0)  # (B, T, 17)

        return torch.mean(weighted_errors)

    def bone_length_preservation_loss(self, x_true, x_recon):
        """본 길이 보존 손실"""
        # 주요 본 연결 정의
        bone_connections = [
            (5, 7), (7, 9),    # 왼팔
            (6, 8), (8, 10),   # 오른팔
            (11, 13), (13, 15), # 왼다리
            (12, 14), (14, 16), # 오른다리
            (5, 6),            # 어깨
            (11, 12)           # 골반
        ]

        def calculate_bone_lengths(poses):
            # poses: (B, T, 51) -> (B, T, 17, 3)
            batch_size, seq_len, _ = poses.shape
            poses_joints = poses.view(batch_size, seq_len, 17, 3)

            bone_lengths = []
            for start_joint, end_joint in bone_connections:
                start_pos = poses_joints[:, :, start_joint]  # (B, T, 3)
                end_pos = poses_joints[:, :, end_joint]      # (B, T, 3)
                length = torch.norm(end_pos - start_pos, dim=-1)  # (B, T)
                bone_lengths.append(length)

            return torch.stack(bone_lengths, dim=-1)  # (B, T, num_bones)

        true_lengths = calculate_bone_lengths(x_true)
        recon_lengths = calculate_bone_lengths(x_recon)

        return F.mse_loss(recon_lengths, true_lengths)

    def temporal_consistency_loss(self, x_true, x_recon):
        """시간적 일관성 손실 (부드러운 움직임)"""
        if x_true.size(1) <= 1:  # 시퀀스 길이가 1 이하면 계산 불가
            return torch.tensor(0.0, device=x_true.device)

        # 프레임 간 차이 계산
        true_diff = x_true[:, 1:] - x_true[:, :-1]      # (B, T-1, D)
        recon_diff = x_recon[:, 1:] - x_recon[:, :-1]   # (B, T-1, D)

        return F.mse_loss(recon_diff, true_diff)

    def symmetry_loss(self, x_true, x_recon):
        """좌우 대칭성 손실"""
        # 좌우 대칭 관절 쌍
        symmetric_pairs = [
            (1, 2),   # left_eye, right_eye
            (3, 4),   # left_ear, right_ear
            (5, 6),   # left_shoulder, right_shoulder
            (7, 8),   # left_elbow, right_elbow
            (9, 10),  # left_wrist, right_wrist
            (11, 12), # left_hip, right_hip
            (13, 14), # left_knee, right_knee
            (15, 16)  # left_ankle, right_ankle
        ]

        batch_size, seq_len, _ = x_true.shape

        x_true_joints = x_true.view(batch_size, seq_len, 17, 3)
        x_recon_joints = x_recon.view(batch_size, seq_len, 17, 3)

        symmetry_losses = []

        for left_joint, right_joint in symmetric_pairs:
            # 원본에서 좌우 관절 간 상대적 위치
            true_diff = x_true_joints[:, :, left_joint] - x_true_joints[:, :, right_joint]
            recon_diff = x_recon_joints[:, :, left_joint] - x_recon_joints[:, :, right_joint]

            # X축 대칭성 (X좌표는 반대 부호, Y,Z는 동일해야 함)
            true_diff_symmetric = true_diff.clone()
            true_diff_symmetric[:, :, 0] *= -1  # X축 반전

            recon_diff_symmetric = recon_diff.clone()
            recon_diff_symmetric[:, :, 0] *= -1  # X축 반전

            symmetry_loss = F.mse_loss(recon_diff_symmetric, true_diff_symmetric)
            symmetry_losses.append(symmetry_loss)

        return torch.mean(torch.stack(symmetry_losses))

    def get_codebook_utilization(self):
        """코드북 사용률 계산"""
        if hasattr(self.vq, '_codebook'):
            cluster_size = self.vq._codebook.cluster_size
            total_codes = cluster_size.sum()
            used_codes = (cluster_size > 0).sum().float()
            utilization = used_codes / self.codebook_size
            return utilization.item()
        return 0.0

    def get_codebook_codes(self):
        """현재 코드북 벡터들 반환"""
        return self.vq.codebook

    def codes_to_poses(self, indices):
        """코드북 인덱스를 포즈로 변환"""
        codes = self.vq.get_codes_from_indices(indices)
        return self.decode(codes)


def create_model(config=None):
    """모델 생성 함수"""
    if config is None:
        config = {
            'input_dim': 51,
            'encoder_hidden_dim': 256,
            'decoder_hidden_dim': 256,
            'latent_dim': 128,
            'codebook_size': 512,
            'codebook_dim': 64,
            'commitment_weight': 0.25,
            'decay': 0.99,
            'encoder_layers': 3,
            'decoder_layers': 3,
            'use_ema': True,
            'threshold_ema_dead_code': 2
        }

    model = GymVaultVQVAE(**config)
    return model


def test_model():
    """모델 테스트"""
    print("=== VQ-VAE 모델 테스트 ===")

    # 모델 생성
    model = create_model()

    # 더미 데이터 (배치=4, 시퀀스=64, 관절=51)
    batch_size, seq_len, input_dim = 4, 64, 51
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"입력 크기: {x.shape}")

    # Forward pass
    with torch.no_grad():
        results = model(x, return_indices=True, return_loss_breakdown=True)

    print(f"복원 출력: {results['x_recon'].shape}")
    print(f"코드북 인덱스: {results['indices'].shape}")
    print(f"전체 손실: {results['total_loss'].item():.4f}")
    print(f"재구성 손실: {results['reconstruction_loss'].item():.4f}")
    print(f"VQ 손실: {results['vq_loss'].item():.4f}")

    # 코드북 사용률
    utilization = model.get_codebook_utilization()
    print(f"코드북 사용률: {utilization:.2%}")

    # 모델 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")

    print("✅ 모델 테스트 완료!")


if __name__ == "__main__":
    test_model()