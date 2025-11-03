#!/usr/bin/env python3
"""
스켈레톤 오버레이 시각화 스크립트

원본 영상 프레임 위에 2D 포즈 스켈레톤을 그려서
보간 전후 비교 (특히 실패 프레임)
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import importlib.util

# 2_data_preprocessing.py에서 클래스 임포트
module_path = Path(__file__).parent / "2_data_preprocessing.py"
spec = importlib.util.spec_from_file_location("data_preprocessing", module_path)
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)

GymVaultDataPreprocessor = data_preprocessing.GymVaultDataPreprocessor


# COCO 17 키포인트 연결 구조
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),                    # 머리
    (1, 3), (2, 4),                    # 귀
    (5, 6),                            # 어깨
    (5, 7), (7, 9),                    # 왼팔
    (6, 8), (8, 10),                   # 오른팔
    (5, 11), (6, 12),                  # 몸통
    (11, 12),                          # 골반
    (11, 13), (13, 15),                # 왼다리
    (12, 14), (14, 16)                 # 오른다리
]

# 키포인트 이름
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def draw_skeleton(image, pose, color=(0, 255, 0), thickness=2, radius=4):
    """
    이미지에 스켈레톤 그리기

    Args:
        image: 원본 이미지
        pose: 2D 포즈 (17, 2) - (x, y) 좌표
        color: 스켈레톤 색상 (B, G, R)
        thickness: 선 두께
        radius: 키포인트 원 반지름
    """
    image = image.copy()

    # 1. 본(뼈) 그리기
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start_point = pose[start_idx]
        end_point = pose[end_idx]

        # 좌표가 유효한 경우만 그리기
        if not (np.isnan(start_point).any() or np.isnan(end_point).any()):
            start_pt = (int(start_point[0]), int(start_point[1]))
            end_pt = (int(end_point[0]), int(end_point[1]))
            cv2.line(image, start_pt, end_pt, color, thickness)

    # 2. 키포인트 그리기
    for i, point in enumerate(pose):
        if not np.isnan(point).any():
            pt = (int(point[0]), int(point[1]))

            # 어깨는 더 크게 강조
            if i == 5 or i == 6:  # left_shoulder, right_shoulder
                cv2.circle(image, pt, radius + 2, (0, 0, 255), -1)  # 빨간색
                cv2.circle(image, pt, radius + 2, color, 2)
            else:
                cv2.circle(image, pt, radius, color, -1)

    return image


def find_failed_frame(poses_original):
    """어깨 검출 실패 프레임 찾기"""
    failed_frames = []

    for frame_idx, frame in enumerate(poses_original):
        left_shoulder = frame[5]
        right_shoulder = frame[6]
        width = np.linalg.norm(right_shoulder - left_shoulder)

        if width < 1e-6:
            failed_frames.append(frame_idx)

    return failed_frames


def main():
    # 문제 영상: 007.avi
    VIDEO_PATH = Path("D:/vector-quantize-pytorch/AQA-7/Actions/gym_vault/train/007.avi")
    POSE_PATH = Path("D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data/train/007_poses.pkl")

    if not VIDEO_PATH.exists():
        print(f"❌ 영상 파일이 없습니다: {VIDEO_PATH}")
        return

    if not POSE_PATH.exists():
        print(f"❌ 포즈 파일이 없습니다: {POSE_PATH}")
        return

    # 1. 포즈 데이터 로드
    print(f"포즈 데이터 로드: {POSE_PATH.name}")
    with open(POSE_PATH, 'rb') as f:
        data = pickle.load(f)

    poses_original = data['poses']

    # 2. 실패 프레임 찾기
    failed_frames = find_failed_frame(poses_original)
    print(f"어깨 검출 실패 프레임: {failed_frames}")

    if len(failed_frames) == 0:
        print("✅ 모든 프레임에서 어깨 검출 성공!")
        return

    # 3. 보간 수행
    print(f"보간 수행 중...")
    preprocessor = GymVaultDataPreprocessor(
        sequence_length=64,
        normalize_method='bone_length'
    )

    # 보간만 수행 (정규화 없이)
    poses_interpolated = preprocessor._interpolate_failed_frames(poses_original.copy(), failed_frames)

    # 4. 영상 로드
    print(f"영상 로드: {VIDEO_PATH.name}")
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다!")
        return

    # 5. 실패 프레임 + 주변 프레임 시각화
    target_frame_idx = failed_frames[0]  # 첫 번째 실패 프레임
    frames_to_show = [target_frame_idx - 1, target_frame_idx, target_frame_idx + 1]

    print(f"\n시각화할 프레임: {frames_to_show}")
    print(f"  - 프레임 {target_frame_idx - 1}: 정상 (이전)")
    print(f"  - 프레임 {target_frame_idx}: 실패 (보간 대상)")
    print(f"  - 프레임 {target_frame_idx + 1}: 정상 (이후)")

    images = []
    for frame_idx in frames_to_show:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            images.append(frame)
        else:
            print(f"⚠️ 프레임 {frame_idx} 읽기 실패")
            images.append(None)

    cap.release()

    # 6. 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col, frame_idx in enumerate(frames_to_show):
        if images[col] is None:
            continue

        img = images[col].copy()

        # 상단: 보간 전
        img_before = draw_skeleton(
            img,
            poses_original[frame_idx],
            color=(0, 255, 0) if frame_idx != target_frame_idx else (0, 0, 255),  # 실패 프레임은 빨간색
            thickness=3,
            radius=5
        )

        axes[0, col].imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        axes[0, col].axis('off')

        if frame_idx == target_frame_idx:
            axes[0, col].set_title(f'Frame {frame_idx}\n❌ BEFORE (Failed)',
                                  fontsize=14, fontweight='bold', color='red')
        else:
            axes[0, col].set_title(f'Frame {frame_idx}\nBEFORE (Normal)',
                                  fontsize=14, fontweight='bold')

        # 하단: 보간 후
        img_after = draw_skeleton(
            img,
            poses_interpolated[frame_idx],
            color=(0, 255, 0),  # 모두 초록색
            thickness=3,
            radius=5
        )

        axes[1, col].imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
        axes[1, col].axis('off')

        if frame_idx == target_frame_idx:
            axes[1, col].set_title(f'Frame {frame_idx}\n✅ AFTER (Interpolated)',
                                  fontsize=14, fontweight='bold', color='green')
        else:
            axes[1, col].set_title(f'Frame {frame_idx}\nAFTER (Normal)',
                                  fontsize=14, fontweight='bold')

    plt.suptitle('007.avi - Skeleton Interpolation Comparison\n(Red skeleton = failed detection, Green = valid)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = "D:/vector-quantize-pytorch/gym_vault_vq_system/interpolation/skeleton_overlay_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 스켈레톤 오버레이 시각화 저장: {output_path}")

    # 7. 어깨 너비 비교
    print(f"\n{'='*60}")
    print(f"어깨 너비 비교 (프레임 {target_frame_idx})")
    print(f"{'='*60}")

    for i, frame_idx in enumerate(frames_to_show):
        ls_before = poses_original[frame_idx][5]
        rs_before = poses_original[frame_idx][6]
        width_before = np.linalg.norm(rs_before - ls_before)

        ls_after = poses_interpolated[frame_idx][5]
        rs_after = poses_interpolated[frame_idx][6]
        width_after = np.linalg.norm(rs_after - ls_after)

        print(f"프레임 {frame_idx}:")
        print(f"  보간 전: {width_before:.2f} pixels")
        print(f"  보간 후: {width_after:.2f} pixels")

        if frame_idx == target_frame_idx:
            print(f"  → 보간으로 복구됨! ✅")
        print()

    plt.show()


if __name__ == "__main__":
    main()
