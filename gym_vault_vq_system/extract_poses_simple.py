#!/usr/bin/env python3
"""
간단한 MMPose 포즈 추출 스크립트
inferencer_demo.py 기반으로 작성
"""

import os
import sys
import pickle
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# MMPose 경로 추가
sys.path.append('D:/mmpose')

from mmpose.apis import MMPoseInferencer


def extract_poses_from_videos(video_dir, output_dir, max_videos=None):
    """
    비디오 폴더에서 포즈 추출

    Args:
        video_dir: 비디오가 있는 폴더 경로
        output_dir: 출력 폴더 경로
        max_videos: 최대 처리 영상 개수
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*70}")
    print(f"  MMPose 포즈 추출 (HRNet 2D)")
    print(f"{'='*70}")
    print(f"  입력: {video_dir}")
    print(f"  출력: {output_dir}")
    print(f"{'='*70}\n")

    # MMPose Inferencer 초기화 (2D만 사용)
    print("MMPose 초기화 중...")
    inferencer = MMPoseInferencer(
        pose2d='td-hm_hrnet-w48_8xb32-210e_coco-256x192',
        device='cuda'
    )
    print("✅ MMPose 초기화 완료\n")

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

    # 각 split별로 처리
    for split_name, split_paths in splits.items():
        for split_path in split_paths:
            if not split_path.exists():
                print(f"⚠️  경로 없음: {split_path}")
                continue

            # 출력 폴더 구조 생성
            if split_name == 'test':
                relative_path = split_path.relative_to(video_dir)
                split_output_dir = output_dir / relative_path
            else:
                split_output_dir = output_dir / split_name

            split_output_dir.mkdir(parents=True, exist_ok=True)

            # 시각화 폴더
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
                    # MMPose Inferencer 실행
                    poses_list = []

                    for result in inferencer(
                        inputs=str(video_file),
                        show=False,
                        return_vis=False,
                        pred_out_dir=str(vis_dir)
                    ):
                        predictions = result.get('predictions', [])

                        if not predictions or len(predictions) == 0:
                            poses_list.append(None)
                            continue

                        # 첫 번째 사람의 키포인트
                        person = predictions[0]

                        if 'keypoints' in person and person['keypoints'] is not None:
                            keypoints = np.asarray(person['keypoints'])  # (17, 3): x, y, score
                            poses_list.append(keypoints[:, :2])  # x, y만 저장
                        else:
                            poses_list.append(None)

                    # None 제거
                    poses_list = [p for p in poses_list if p is not None]

                    if len(poses_list) >= 30:  # 최소 30프레임
                        poses_array = np.asarray(poses_list)  # (T, 17, 2)

                        # .pkl 파일로 저장
                        output_file = split_output_dir / f"{video_file.stem}_poses.pkl"

                        pose_data = {
                            'video_name': video_file.name,
                            'split': str(split_path.relative_to(video_dir)),
                            'poses': poses_array,
                            'keypoint_names': [
                                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                            ],
                            'shape': poses_array.shape,
                            'extractor': 'MMPose (HRNet 2D)'
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
                'extractor': 'MMPose (HRNet 2D)',
                'pose_format': 'COCO-17',
                'pose_dim': '(T, 17, 2)'
            }

            metadata_file = split_output_dir / 'extraction_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(split_metadata, f, indent=2, ensure_ascii=False)

            total_successful += successful
            total_failed.extend(failed)

    # 전체 결과 요약
    print(f"\n{'='*70}")
    print(f"전체 포즈 추출 완료")
    print(f"{'='*70}")
    print(f"  ✅ 총 성공: {total_successful}개")
    print(f"  ❌ 총 실패: {len(total_failed)}개")
    print(f"  💾 저장 위치: {output_dir}")
    print(f"{'='*70}\n")

    return total_successful, total_failed


def main():
    """메인 실행 함수"""

    VIDEO_DIR = "D:/vector-quantize-pytorch/AQA-7/Actions/gym_vault"
    OUTPUT_DIR = "D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data"

    print("\n" + "="*70)
    print("  도마 영상 포즈 추출 (MMPose HRNet)")
    print("="*70)
    print("\n모드 선택:")
    print("  1. 테스트 (처음 10개 영상)")
    print("  2. 전체 실행 (176개 영상)")
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
    successful, failed = extract_poses_from_videos(
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
