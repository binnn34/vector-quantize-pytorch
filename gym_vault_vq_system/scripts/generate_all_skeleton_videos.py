#!/usr/bin/env python3
"""
전체 train 데이터셋 스켈레톤 오버레이 영상 생성

68개 train 영상 전체에 대해:
- 원본 영상 위에 2D 포즈 스켈레톤 오버레이
- 보간 전후 비교 (side-by-side)
- 영상 파일로 저장
"""

import numpy as np
import pickle
import cv2
from pathlib import Path
from tqdm import tqdm
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


def draw_skeleton(image, pose, color=(0, 255, 0), thickness=2, radius=4, show_shoulder_width=False):
    """
    이미지에 스켈레톤 그리기

    Args:
        image: 원본 이미지
        pose: 2D 포즈 (17, 2) - (x, y) 좌표
        color: 스켈레톤 색상 (B, G, R)
        thickness: 선 두께
        radius: 키포인트 원 반지름
        show_shoulder_width: 어깨 너비 표시 여부
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
                cv2.circle(image, pt, radius + 3, (0, 0, 255), -1)  # 빨간색
                cv2.circle(image, pt, radius + 3, (255, 255, 255), 2)  # 흰색 테두리
            else:
                cv2.circle(image, pt, radius, color, -1)
                cv2.circle(image, pt, radius, (255, 255, 255), 1)  # 흰색 테두리

    # 3. 어깨 너비 표시 (선택)
    if show_shoulder_width:
        left_shoulder = pose[5]
        right_shoulder = pose[6]

        if not (np.isnan(left_shoulder).any() or np.isnan(right_shoulder).any()):
            width = np.linalg.norm(right_shoulder - left_shoulder)

            # 어깨 중심에 너비 텍스트 표시
            center = ((left_shoulder + right_shoulder) / 2).astype(int)
            cv2.putText(image, f'{width:.1f}px', tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return image


def process_video(video_path, pose_path, output_path, preprocessor):
    """
    단일 영상 처리: 원본 영상 + 스켈레톤 오버레이

    Args:
        video_path: 원본 영상 경로
        pose_path: 포즈 데이터 경로
        output_path: 출력 영상 경로
        preprocessor: 정규화 전처리기

    Returns:
        success: 성공 여부
        info: 처리 정보
    """
    try:
        # 1. 포즈 데이터 로드
        with open(pose_path, 'rb') as f:
            data = pickle.load(f)

        poses_original = data['poses']
        video_name = data['video_name']

        # 2. 실패 프레임 찾기
        failed_frames = []
        for frame_idx, frame in enumerate(poses_original):
            left_shoulder = frame[5]
            right_shoulder = frame[6]
            width = np.linalg.norm(right_shoulder - left_shoulder)
            if width < 1e-6:
                failed_frames.append(frame_idx)

        # 3. 보간 수행
        if len(failed_frames) > 0:
            poses_interpolated = preprocessor._interpolate_failed_frames(poses_original.copy(), failed_frames)
        else:
            poses_interpolated = poses_original.copy()

        # 4. 영상 로드
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, f"영상을 열 수 없음: {video_path}"

        # 영상 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 5. 출력 영상 설정 (side-by-side이므로 너비 2배)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))

        # 6. 프레임별 처리
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= len(poses_original):
                break

            # 왼쪽: 보간 전
            if frame_idx in failed_frames:
                frame_before = draw_skeleton(frame, poses_original[frame_idx],
                                            color=(0, 0, 255), thickness=3, radius=5)
                # "FAILED" 텍스트 표시
                cv2.putText(frame_before, 'FAILED DETECTION', (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                frame_before = draw_skeleton(frame, poses_original[frame_idx],
                                            color=(0, 255, 0), thickness=3, radius=5)

            cv2.putText(frame_before, 'BEFORE Interpolation', (20, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 오른쪽: 보간 후
            frame_after = draw_skeleton(frame, poses_interpolated[frame_idx],
                                       color=(0, 255, 0), thickness=3, radius=5)

            if frame_idx in failed_frames:
                cv2.putText(frame_after, 'INTERPOLATED', (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.putText(frame_after, 'AFTER Interpolation', (20, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Side-by-side 합치기
            combined = np.hstack([frame_before, frame_after])

            # 프레임 번호 표시
            cv2.putText(combined, f'Frame: {frame_idx}/{total_frames-1}', (width - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            out.write(combined)
            frame_idx += 1

        cap.release()
        out.release()

        return True, {
            'video_name': video_name,
            'total_frames': total_frames,
            'failed_frames': len(failed_frames),
            'failed_frame_ids': failed_frames
        }

    except Exception as e:
        return False, f"처리 실패: {str(e)}"


def main():
    """전체 train 데이터셋 처리"""

    # 경로 설정
    VIDEO_DIR = Path("D:/vector-quantize-pytorch/AQA-7/Actions/gym_vault/train")
    POSE_DIR = Path("D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data/train")
    OUTPUT_DIR = Path("D:/vector-quantize-pytorch/gym_vault_vq_system/skeleton_videos")

    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 정규화기 초기화
    preprocessor = GymVaultDataPreprocessor(
        sequence_length=64,
        normalize_method='bone_length'
    )

    # 포즈 파일 리스트
    pose_files = sorted(list(POSE_DIR.glob('*_poses.pkl')))

    print("="*80)
    print(f"전체 Train 데이터셋 스켈레톤 오버레이 영상 생성")
    print("="*80)
    print(f"총 영상 수: {len(pose_files)}개")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print("="*80)

    success_count = 0
    failed_count = 0
    total_failed_frames = 0

    results = []

    for pose_file in tqdm(pose_files, desc="영상 처리 중"):
        # 영상 파일명 추출 (003_poses.pkl → 003.avi)
        video_name = pose_file.stem.replace('_poses', '.avi')
        video_path = VIDEO_DIR / video_name
        output_path = OUTPUT_DIR / f"{video_name.replace('.avi', '_skeleton.mp4')}"

        # 영상 파일 존재 확인
        if not video_path.exists():
            print(f"\n⚠️ 영상 파일 없음: {video_name}")
            failed_count += 1
            continue

        # 처리
        success, info = process_video(video_path, pose_file, output_path, preprocessor)

        if success:
            success_count += 1
            total_failed_frames += info['failed_frames']
            results.append(info)
        else:
            failed_count += 1
            print(f"\n❌ {video_name}: {info}")

    # 결과 요약
    print("\n" + "="*80)
    print("처리 완료 요약")
    print("="*80)
    print(f"성공: {success_count}개")
    print(f"실패: {failed_count}개")
    print(f"총 보간된 프레임 수: {total_failed_frames}개")

    # 실패 프레임이 많은 영상 Top 10
    if results:
        results_sorted = sorted(results, key=lambda x: x['failed_frames'], reverse=True)

        print("\n보간된 프레임이 많은 영상 Top 10:")
        print("-"*80)
        for i, result in enumerate(results_sorted[:10], 1):
            print(f"{i}. {result['video_name']}: {result['failed_frames']}개 프레임 보간")
            if result['failed_frames'] > 0:
                print(f"   프레임 번호: {result['failed_frame_ids'][:5]}{'...' if len(result['failed_frame_ids']) > 5 else ''}")

    print(f"\n✅ 모든 영상이 저장되었습니다: {OUTPUT_DIR}")
    print(f"\n💡 영상 확인 방법:")
    print(f"   - 왼쪽: 보간 전 (빨간색 = 검출 실패, 초록색 = 정상)")
    print(f"   - 오른쪽: 보간 후 (모두 초록색)")


if __name__ == "__main__":
    main()
