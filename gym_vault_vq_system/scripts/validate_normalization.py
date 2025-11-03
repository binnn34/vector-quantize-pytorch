#!/usr/bin/env python3
"""
정규화 검증 스크립트

교수님 피드백 기반 정규화가 제대로 작동하는지 확인:
1. 어깨 중심이 원점(0, 0)에 있는지
2. 어깨가 수평(y축 동일)인지
3. 어깨 너비가 1.0인지
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 2_data_preprocessing.py에서 클래스 임포트
# Python import 시스템은 숫자로 시작하는 모듈을 직접 import할 수 없으므로 importlib 사용
import importlib.util

# 모듈 동적 로드
module_path = Path(__file__).parent / "2_data_preprocessing.py"
spec = importlib.util.spec_from_file_location("data_preprocessing", module_path)
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)

GymVaultDataPreprocessor = data_preprocessing.GymVaultDataPreprocessor


def validate_single_frame(frame_pose, title="Frame"):
    """
    단일 프레임의 정규화 검증

    검증 항목:
    1. 어깨 중심 = (0, 0)
    2. 어깨 수평 (left_shoulder[1] ≈ right_shoulder[1])
    3. 어깨 너비 = 1.0
    """
    left_shoulder = frame_pose[5]
    right_shoulder = frame_pose[6]

    # 검증 1: 어깨 중심
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    center_error = np.linalg.norm(shoulder_center)

    # 검증 2: 어깨 수평도
    y_diff = abs(right_shoulder[1] - left_shoulder[1])

    # 검증 3: 어깨 너비
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)

    print(f"\n{title} 검증 결과:")
    print(f"  ✓ 어깨 중심: ({shoulder_center[0]:.6f}, {shoulder_center[1]:.6f}) | 오차: {center_error:.6f}")
    print(f"  ✓ 어깨 수평도: {y_diff:.6f} (이상적으로 0에 가까워야 함)")
    print(f"  ✓ 어깨 너비: {shoulder_width:.6f} (이상적으로 1.0이어야 함)")

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 전체 포즈
    ax1.scatter(frame_pose[:, 0], frame_pose[:, 1], c='blue', s=50, alpha=0.7, label='Joints')
    ax1.scatter(left_shoulder[0], left_shoulder[1], c='red', s=100, label='Left Shoulder', marker='x')
    ax1.scatter(right_shoulder[0], right_shoulder[1], c='green', s=100, label='Right Shoulder', marker='x')
    ax1.scatter(0, 0, c='black', s=100, label='Origin', marker='+')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'{title} - Full Pose')
    ax1.legend()
    ax1.set_aspect('equal')

    # 어깨 확대
    ax2.scatter(left_shoulder[0], left_shoulder[1], c='red', s=200, label='Left Shoulder', marker='x')
    ax2.scatter(right_shoulder[0], right_shoulder[1], c='green', s=200, label='Right Shoulder', marker='x')
    ax2.scatter(shoulder_center[0], shoulder_center[1], c='orange', s=200, label='Shoulder Center', marker='o')
    ax2.scatter(0, 0, c='black', s=200, label='Origin', marker='+')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{title} - Shoulder Zoom')
    ax2.legend()
    ax2.set_aspect('equal')

    # 어깨 영역만 확대
    margin = 0.3
    ax2.set_xlim(-margin, 1 + margin)
    ax2.set_ylim(-margin, margin)

    plt.tight_layout()
    return center_error, y_diff, shoulder_width


def validate_sequence(poses_before, poses_after, video_name):
    """
    전체 시퀀스에 대해 정규화 전후 비교
    """
    num_frames = len(poses_before)

    # 통계 수집
    center_errors = []
    y_diffs = []
    shoulder_widths = []

    for frame_idx in range(num_frames):
        frame_after = poses_after[frame_idx]

        left_shoulder = frame_after[5]
        right_shoulder = frame_after[6]

        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        center_error = np.linalg.norm(shoulder_center)
        y_diff = abs(right_shoulder[1] - left_shoulder[1])
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)

        center_errors.append(center_error)
        y_diffs.append(y_diff)
        shoulder_widths.append(shoulder_width)

    center_errors = np.array(center_errors)
    y_diffs = np.array(y_diffs)
    shoulder_widths = np.array(shoulder_widths)

    print(f"\n{'='*60}")
    print(f"비디오: {video_name}")
    print(f"프레임 수: {num_frames}")
    print(f"{'='*60}")
    print(f"\n어깨 중심 오차 (원점으로부터 거리):")
    print(f"  평균: {center_errors.mean():.6f}")
    print(f"  표준편차: {center_errors.std():.6f}")
    print(f"  최대: {center_errors.max():.6f}")

    print(f"\n어깨 수평도 (y좌표 차이):")
    print(f"  평균: {y_diffs.mean():.6f}")
    print(f"  표준편차: {y_diffs.std():.6f}")
    print(f"  최대: {y_diffs.max():.6f}")

    print(f"\n어깨 너비 (1.0이 이상적):")
    print(f"  평균: {shoulder_widths.mean():.6f}")
    print(f"  표준편차: {shoulder_widths.std():.6f}")
    print(f"  최소/최대: {shoulder_widths.min():.6f} / {shoulder_widths.max():.6f}")

    # 시각화
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 어깨 중심 오차
    axes[0].plot(center_errors, linewidth=2)
    axes[0].axhline(0, color='red', linestyle='--', label='Target')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Center Error')
    axes[0].set_title(f'{video_name} - Shoulder Center Error (should be ~0)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 어깨 수평도
    axes[1].plot(y_diffs, linewidth=2, color='green')
    axes[1].axhline(0, color='red', linestyle='--', label='Target')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Y-coordinate Difference')
    axes[1].set_title(f'{video_name} - Shoulder Horizontality (should be ~0)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 어깨 너비
    axes[2].plot(shoulder_widths, linewidth=2, color='orange')
    axes[2].axhline(1.0, color='red', linestyle='--', label='Target')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Shoulder Width')
    axes[2].set_title(f'{video_name} - Shoulder Width (should be 1.0)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    return {
        'center_errors': center_errors,
        'y_diffs': y_diffs,
        'shoulder_widths': shoulder_widths
    }


def main():
    """메인 검증 프로세스"""

    # 테스트할 샘플 파일
    SAMPLE_FILE = Path("D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data/train/003_poses.pkl")

    if not SAMPLE_FILE.exists():
        print(f"❌ 샘플 파일이 없습니다: {SAMPLE_FILE}")
        return

    # 1. 원본 데이터 로드
    print(f"원본 데이터 로드: {SAMPLE_FILE}")
    with open(SAMPLE_FILE, 'rb') as f:
        data = pickle.load(f)

    poses_before = data['poses']
    video_name = data['video_name']

    print(f"\n비디오: {video_name}")
    print(f"형태: {poses_before.shape}")

    # 2. 정규화 수행
    print(f"\n정규화 수행 (bone_length 방식)...")
    preprocessor = GymVaultDataPreprocessor(
        sequence_length=64,
        normalize_method='bone_length'
    )

    poses_after = preprocessor._normalize_bone_length(poses_before)

    # 3. 첫 프레임 상세 검증
    print(f"\n{'='*60}")
    print(f"첫 프레임 상세 검증")
    print(f"{'='*60}")
    validate_single_frame(poses_after[0], title="Frame 0 (After Normalization)")
    plt.savefig("D:/vector-quantize-pytorch/gym_vault_vq_system/validation_frame0.png", dpi=150, bbox_inches='tight')
    print(f"✓ 프레임 0 시각화 저장: validation_frame0.png")

    # 4. 전체 시퀀스 통계 검증
    stats = validate_sequence(poses_before, poses_after, video_name)
    plt.savefig("D:/vector-quantize-pytorch/gym_vault_vq_system/validation_sequence.png", dpi=150, bbox_inches='tight')
    print(f"✓ 시퀀스 통계 시각화 저장: validation_sequence.png")

    # 5. 판정
    print(f"\n{'='*60}")
    print(f"종합 판정")
    print(f"{'='*60}")

    center_pass = stats['center_errors'].mean() < 1e-10
    horizontal_pass = stats['y_diffs'].mean() < 1e-10
    width_pass = abs(stats['shoulder_widths'].mean() - 1.0) < 1e-10

    print(f"어깨 중심 (원점): {'✅ PASS' if center_pass else '❌ FAIL'}")
    print(f"어깨 수평도: {'✅ PASS' if horizontal_pass else '❌ FAIL'}")
    print(f"어깨 너비 (1.0): {'✅ PASS' if width_pass else '❌ FAIL'}")

    if center_pass and horizontal_pass and width_pass:
        print(f"\n🎉 정규화가 올바르게 작동합니다!")
        print(f"이제 전체 데이터셋을 재처리해도 됩니다.")
    else:
        print(f"\n⚠️ 정규화에 문제가 있습니다. 코드를 다시 확인하세요.")

    plt.show()


if __name__ == "__main__":
    main()
