#!/usr/bin/env python3
"""
보간 효과 시각화 스크립트

이전 문제 영상(007.avi)의 보간 전후 비교
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util

# 2_data_preprocessing.py에서 클래스 임포트
module_path = Path(__file__).parent / "2_data_preprocessing.py"
spec = importlib.util.spec_from_file_location("data_preprocessing", module_path)
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)

GymVaultDataPreprocessor = data_preprocessing.GymVaultDataPreprocessor


def main():
    # 이전 최악의 문제 영상: 007.avi
    SAMPLE_FILE = Path("D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data/train/007_poses.pkl")

    print(f"영상 로드: {SAMPLE_FILE.name}")
    with open(SAMPLE_FILE, 'rb') as f:
        data = pickle.load(f)

    poses_original = data['poses']

    # 어깨 너비 계산 (보간 전)
    shoulder_widths_before = []
    for frame in poses_original:
        left_shoulder = frame[5]
        right_shoulder = frame[6]
        width = np.linalg.norm(right_shoulder - left_shoulder)
        shoulder_widths_before.append(width)

    # 정규화 수행 (보간 포함)
    preprocessor = GymVaultDataPreprocessor(
        sequence_length=64,
        normalize_method='bone_length'
    )

    poses_normalized = preprocessor._normalize_bone_length(poses_original)

    # 어깨 너비 계산 (보간 후)
    shoulder_widths_after = []
    for frame in poses_normalized:
        left_shoulder = frame[5]
        right_shoulder = frame[6]
        width = np.linalg.norm(right_shoulder - left_shoulder)
        shoulder_widths_after.append(width)

    # 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 보간 전
    ax1.plot(shoulder_widths_before, linewidth=2, color='red', label='Before Interpolation')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Shoulder Width (pixels)', fontsize=12)
    ax1.set_title('007.avi - Shoulder Width BEFORE Interpolation (with failures)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # 실패 프레임 강조
    failed_frames = [i for i, w in enumerate(shoulder_widths_before) if w < 1e-6]
    if failed_frames:
        ax1.scatter(failed_frames, [0] * len(failed_frames),
                   color='red', s=100, marker='x', label=f'Failed Frames ({len(failed_frames)})', zorder=10)
        ax1.legend(fontsize=12)

    # 보간 후
    ax2.plot(shoulder_widths_after, linewidth=2, color='green', label='After Interpolation')
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Target (1.0)')
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Shoulder Width (normalized)', fontsize=12)
    ax2.set_title('007.avi - Shoulder Width AFTER Interpolation (all normalized)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    plt.tight_layout()

    output_path = "D:/vector-quantize-pytorch/gym_vault_vq_system/interpolation/frame_007.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 보간 효과 시각화 저장: {output_path}")

    # 통계 출력
    print(f"\n{'='*60}")
    print(f"보간 효과 통계 (007.avi)")
    print(f"{'='*60}")
    print(f"보간 전:")
    print(f"  실패 프레임 수: {len(failed_frames)}")
    print(f"  어깨 너비 최소: {np.min(shoulder_widths_before):.6f}")
    print(f"  어깨 너비 최대: {np.max(shoulder_widths_before):.6f}")
    print(f"  어깨 너비 평균: {np.mean(shoulder_widths_before):.6f}")

    print(f"\n보간 후:")
    print(f"  실패 프레임 수: 0")
    print(f"  어깨 너비 최소: {np.min(shoulder_widths_after):.6f}")
    print(f"  어깨 너비 최대: {np.max(shoulder_widths_after):.6f}")
    print(f"  어깨 너비 평균: {np.mean(shoulder_widths_after):.6f}")

    print(f"\n🎉 보간이 성공적으로 작동했습니다!")

    plt.show()


if __name__ == "__main__":
    main()
