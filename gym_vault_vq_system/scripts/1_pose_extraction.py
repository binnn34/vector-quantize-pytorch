#!/usr/bin/env python3
"""
체조 도마 영상에서 포즈 추출 및 품질 기반 자동 분류
MMPose (HRNet 2D + PoseLift 3D) → 품질 평가 → train/test 자동 분류
"""

import os
import sys
import cv2
import numpy as np
import torch
import json
import pickle
from pathlib import Path
from tqdm import tqdm

# MMPose 경로 추가
sys.path.append('D:/mmpose')

from mmpose.apis import MMPoseInferencer
# from pose_quality_filter import VaultPoseQualityFilter  # 자동 분류 비활성화

# ==================== 설정 ====================
MMPOSE_ONLY = True  # MMPose만 사용 (MediaPipe 폴백 비활성화)
USE_VIDEO_MODE_FOR_3D = True  # 비디오 전체를 inferencer에 넣어 3D 안정화 (권장)

# MMPose 모델 지정
POSE2D_MODEL = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192'  # HRNet COCO
POSE3D_MODEL = 'video-pose-lift_tcn-243frm_8xb128-120e_h36m'  # TCN 3D 리프팅


class GymVaultPoseExtractor:
    """도마 영상에서 MMPose로 포즈 추출"""

    def __init__(self, use_3d=True, confidence_threshold=0.3):
        self.use_3d = use_3d
        self.confidence_threshold = confidence_threshold
        self.setup_pose_estimator()

        # COCO 17 키포인트 정의
        self.coco_keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def setup_pose_estimator(self):
        """MMPose Inferencer 초기화"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self.inferencer = MMPoseInferencer(
                pose2d=POSE2D_MODEL,
                pose3d=POSE3D_MODEL if self.use_3d else None,
                device=device
            )
            mode = "3D (HRNet + PoseLift)" if self.use_3d else "2D (HRNet)"
            print(f"✅ MMPose 초기화 완료 [{mode}, {device.upper()}]")
        except Exception as e:
            print(f"❌ MMPose 초기화 실패: {e}")
            raise

    def extract_video_poses_mmpose_videomode(self, video_path, vis_output_dir=None):
        """
        비디오 모드: 영상 전체를 MMPose에 투입 (3D에 최적)
        Temporal context를 활용하여 3D 포즈 안정성 확보

        Args:
            video_path: 입력 영상 경로
            vis_output_dir: 시각화 결과 저장 경로 (None이면 저장 안 함)
        """
        poses = []

        try:
            # Generator 방식으로 프레임별 결과 받기
            for result in self.inferencer(
                inputs=str(video_path),
                show=False,
                return_vis=False,
                pred_out_dir=str(vis_output_dir) if vis_output_dir else None
            ):
                predictions = result.get('predictions', [])

                if not predictions or len(predictions) == 0:
                    poses.append(None)
                    continue

                # 첫 번째 사람의 키포인트 (단일 인물 가정)
                person = predictions[0]

                # 3D 우선 → 2D 폴백
                if self.use_3d and 'keypoints_3d' in person and person['keypoints_3d'] is not None:
                    k3d = np.asarray(person['keypoints_3d'])[:, :3]  # (17, 3): x, y, z
                    poses.append(k3d)
                elif 'keypoints' in person and person['keypoints'] is not None:
                    k2d = np.asarray(person['keypoints'])[:, :2]  # (17, 2): x, y
                    # 2D를 3D 형식으로 패딩 (z=0)
                    k3d = np.hstack([k2d, np.zeros((17, 1))])
                    poses.append(k3d)
                else:
                    poses.append(None)

        except Exception as e:
            print(f"비디오 모드 추출 오류: {e}")
            return None

        # None 제거
        poses = [p for p in poses if p is not None]

        if len(poses) == 0:
            return None

        return np.asarray(poses)  # (frames, 17, 3)

    def extract_pose_mmpose_frame(self, frame):
        """
        프레임 모드: 개별 프레임 처리 (2D 중심, 빠름)
        3D는 temporal context 부족으로 덜 안정적
        """
        try:
            gen = self.inferencer(frame, show=False, return_vis=False)
            result = next(gen, None)

            if result is None:
                return None

            predictions = result.get('predictions', [])
            if not predictions or len(predictions) == 0:
                return None

            person = predictions[0]

            # 3D 우선 → 2D 폴백
            if self.use_3d and 'keypoints_3d' in person and person['keypoints_3d'] is not None:
                k3d = np.asarray(person['keypoints_3d'])[:, :3]
                return k3d
            elif 'keypoints' in person and person['keypoints'] is not None:
                k2d = np.asarray(person['keypoints'])[:, :2]
                # 2D를 3D로 패딩
                k3d = np.hstack([k2d, np.zeros((17, 1))])
                return k3d

        except Exception as e:
            pass

        return None

    def extract_video_poses(self, video_path, vis_output_dir=None):
        """
        영상에서 포즈 시퀀스 추출
        3D 사용 시 비디오 모드 권장

        Args:
            video_path: 입력 영상 경로
            vis_output_dir: 시각화 결과 저장 경로 (None이면 저장 안 함)
        """
        # 3D 사용 시 비디오 모드 (temporal context 활용)
        if USE_VIDEO_MODE_FOR_3D and self.use_3d:
            poses = self.extract_video_poses_mmpose_videomode(video_path, vis_output_dir)
            if poses is not None:
                print(f"  추출 완료: {poses.shape} [{os.path.basename(video_path)}]")
            return poses

        # 2D 또는 프레임 모드 (빠름)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        poses = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 크기 조정 (속도 향상)
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640.0 / width
                frame = cv2.resize(frame, (640, int(height * scale)))

            pose = self.extract_pose_mmpose_frame(frame)
            if pose is not None:
                poses.append(pose)

        cap.release()

        if len(poses) == 0:
            return None

        poses_array = np.asarray(poses)
        print(f"  추출 완료: {poses_array.shape} [{os.path.basename(video_path)}]")
        return poses_array

    def process_all_videos(self, video_dir, output_dir, max_videos=None):
        """
        모든 도마 영상 처리 (포즈 추출만)
        train/validation/test(bad/good) 폴더 구조 유지하며 처리

        Args:
            video_dir: 영상 디렉토리 (AQA-7/Actions/gym_vault)
            output_dir: 출력 디렉토리
            max_videos: 최대 처리 영상 개수 (각 split별로 적용)
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)

        print(f"\n{'='*70}")
        print(f"  도마 영상 포즈 추출 시스템 (MMPose)")
        print(f"{'='*70}")
        print(f"  입력:      {video_dir}")
        print(f"  출력:      {output_dir}")
        print(f"  모드:      {'3D (Video Mode)' if USE_VIDEO_MODE_FOR_3D and self.use_3d else '2D/프레임 모드'}")
        print(f"{'='*70}\n")

        # 폴더 구조 정의
        splits = {
            'train': [video_dir / 'train'],
            'validation': [video_dir / 'validation'],
            'test': [
                video_dir / 'test' / 'bad',
                video_dir / 'test' / 'good'
            ]
        }

        total_successful = 0
        total_failed = []
        all_metadata = {}

        # 각 split별로 처리
        for split_name, split_paths in splits.items():
            for split_path in split_paths:
                if not split_path.exists():
                    print(f"⚠️  경로 없음: {split_path}")
                    continue

                # 출력 폴더 구조 생성
                if split_name == 'test':
                    # test/bad 또는 test/good
                    relative_path = split_path.relative_to(video_dir)
                    split_output_dir = output_dir / relative_path
                else:
                    # train 또는 validation
                    split_output_dir = output_dir / split_name

                split_output_dir.mkdir(parents=True, exist_ok=True)

                # 시각화 폴더 생성
                vis_dir = split_output_dir / 'visualizations'
                vis_dir.mkdir(parents=True, exist_ok=True)

                # 비디오 파일 검색
                video_files = sorted(split_path.glob('*.avi'))
                if max_videos and max_videos > 0:
                    video_files = video_files[:max_videos]

                if len(video_files) == 0:
                    print(f"ℹ️  비디오 없음: {split_path}")
                    continue

                print(f"\n{'='*70}")
                print(f"📂 처리 중: {split_path.relative_to(video_dir)}")
                print(f"   총 {len(video_files)}개 영상")
                print(f"{'='*70}")

                successful = 0
                failed = []

                # 포즈 추출
                for video_file in tqdm(video_files, desc=f"{split_path.name} 포즈 추출"):
                    try:
                        poses = self.extract_video_poses(str(video_file), vis_output_dir=vis_dir)

                        if poses is not None and len(poses) >= 30:  # 최소 30프레임
                            # 결과 저장
                            output_file = split_output_dir / f"{video_file.stem}_poses.pkl"

                            pose_data = {
                                'video_name': video_file.name,
                                'split': str(split_path.relative_to(video_dir)),
                                'poses': poses,
                                'keypoint_names': self.coco_keypoints,
                                'shape': poses.shape,
                                'extractor': f"MMPose (HRNet + {'PoseLift 3D' if self.use_3d else '2D Only'})",
                                'mode': 'video_mode' if USE_VIDEO_MODE_FOR_3D and self.use_3d else 'frame_mode'
                            }

                            with open(output_file, 'wb') as f:
                                pickle.dump(pose_data, f)

                            successful += 1
                        else:
                            failed.append(video_file.name)

                    except Exception as e:
                        failed.append(video_file.name)
                        print(f"\n  ⚠️ {video_file.name}: {str(e)[:50]}")

                # 분할별 결과 요약
                print(f"\n  ✅ 성공: {successful}개")
                print(f"  ❌ 실패: {len(failed)}개")

                if failed and len(failed) <= 5:
                    print(f"  실패한 파일: {', '.join(failed)}")

                # 분할별 메타데이터 저장
                split_metadata = {
                    'split': str(split_path.relative_to(video_dir)),
                    'total_videos': len(video_files),
                    'successful': successful,
                    'failed': len(failed),
                    'failed_videos': failed,
                    'extractor': f"MMPose (HRNet + {'PoseLift 3D' if self.use_3d else '2D Only'})",
                    'mode': 'video_mode' if USE_VIDEO_MODE_FOR_3D and self.use_3d else 'frame_mode',
                    'pose_format': 'COCO-17',
                    'pose_dim': '(T, 17, 3)' if self.use_3d else '(T, 17, 2)'
                }

                metadata_file = split_output_dir / 'extraction_metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(split_metadata, f, indent=2, ensure_ascii=False)

                # 전체 통계 누적
                total_successful += successful
                total_failed.extend(failed)
                all_metadata[str(split_path.relative_to(video_dir))] = split_metadata

        # 전체 결과 요약
        print(f"\n{'='*70}")
        print(f"전체 포즈 추출 완료")
        print(f"{'='*70}")
        print(f"  ✅ 총 성공: {total_successful}개")
        print(f"  ❌ 총 실패: {len(total_failed)}개")
        print(f"  💾 저장 위치: {output_dir}")
        print(f"{'='*70}\n")

        # 전체 메타데이터 저장
        overall_metadata = {
            'total_successful': total_successful,
            'total_failed': len(total_failed),
            'splits': all_metadata
        }

        with open(output_dir / 'overall_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(overall_metadata, f, indent=2, ensure_ascii=False)

        return total_successful, total_failed


def main():
    """메인 실행 함수"""

    # 경로 설정
    VIDEO_DIR = "D:/vector-quantize-pytorch/AQA-7/Actions/gym_vault"
    OUTPUT_DIR = "D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data"

    # 포즈 추출기 초기화
    extractor = GymVaultPoseExtractor(
        use_3d=True,  # 3D 포즈 사용 (HRNet + PoseLift)
        confidence_threshold=0.3
    )

    # 모드 선택
    print("\n" + "="*70)
    print("  도마 영상 포즈 추출 시스템 (MMPose)")
    print("="*70)
    print("\n모드 선택:")
    print("  1. 테스트 (처음 10개 영상 - 약 2-3분)")
    print("  2. 전체 실행 (176개 영상 - 약 15-20분)")
    print("  3. 커스텀 (영상 개수 직접 입력)")

    choice = input("\n선택 (1/2/3): ").strip()

    if choice == '1':
        print("\n>>> 테스트 모드: 처음 10개 영상 처리\n")
        max_videos = 10
    elif choice == '2':
        print("\n>>> 전체 모드: 176개 영상 전체 처리\n")
        max_videos = None
    elif choice == '3':
        max_videos = int(input("처리할 영상 개수: "))
    else:
        print("잘못된 선택입니다.")
        return

    # 실행
    successful, failed = extractor.process_all_videos(
        VIDEO_DIR,
        OUTPUT_DIR,
        max_videos=max_videos
    )

    if successful > 0:
        print(f"✅ 포즈 추출 완료!")
        print(f"  성공: {successful}개 영상")
        if failed:
            print(f"  실패: {len(failed)}개 영상\n")
    else:
        print(f"❌ 포즈 추출 실패. 설정을 확인해주세요.\n")


if __name__ == "__main__":
    main()
