#!/usr/bin/env python3
"""
전체 데이터셋 정규화 검증 스크립트

train/validation/test 모든 영상에 대해 정규화 검증
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import importlib.util

# 2_data_preprocessing.py에서 클래스 임포트
module_path = Path(__file__).parent / "2_data_preprocessing.py"
spec = importlib.util.spec_from_file_location("data_preprocessing", module_path)
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)

GymVaultDataPreprocessor = data_preprocessing.GymVaultDataPreprocessor


def validate_video(poses_after, video_name):
    """
    단일 비디오의 정규화 검증
    """
    num_frames = len(poses_after)

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

    return {
        'video_name': video_name,
        'num_frames': num_frames,
        'center_error_mean': center_errors.mean(),
        'center_error_max': center_errors.max(),
        'y_diff_mean': y_diffs.mean(),
        'y_diff_max': y_diffs.max(),
        'shoulder_width_mean': shoulder_widths.mean(),
        'shoulder_width_std': shoulder_widths.std(),
        'shoulder_width_min': shoulder_widths.min(),
        'shoulder_width_max': shoulder_widths.max(),
    }


def main():
    """전체 데이터셋 검증"""

    # 검증할 데이터셋 디렉토리
    DATASETS = {
        'train': Path("D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data/train"),
        'validation': Path("D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data/validation"),
        'test': Path("D:/vector-quantize-pytorch/gym_vault_vq_system/2d_pose_data/test"),
    }

    # 정규화기 초기화
    preprocessor = GymVaultDataPreprocessor(
        sequence_length=64,
        normalize_method='bone_length'
    )

    all_results = []
    failed_videos = []

    print("="*80)
    print("전체 데이터셋 정규화 검증 시작")
    print("="*80)

    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            print(f"\n⚠️ {dataset_name} 디렉토리가 없습니다: {dataset_path}")
            continue

        pose_files = list(dataset_path.glob('*_poses.pkl'))
        print(f"\n{dataset_name.upper()}: {len(pose_files)}개 영상 검증 중...")

        for pose_file in tqdm(pose_files, desc=f"{dataset_name}"):
            try:
                # 1. 원본 데이터 로드
                with open(pose_file, 'rb') as f:
                    data = pickle.load(f)

                poses_before = data['poses']
                video_name = data['video_name']

                # 2. 정규화 수행
                poses_after = preprocessor._normalize_bone_length(poses_before)

                # 3. 검증
                result = validate_video(poses_after, video_name)
                result['dataset'] = dataset_name
                result['file_path'] = str(pose_file)
                all_results.append(result)

            except Exception as e:
                failed_videos.append({
                    'file': pose_file.name,
                    'dataset': dataset_name,
                    'error': str(e)
                })
                print(f"\n❌ 실패: {pose_file.name} - {e}")

    # 결과 분석
    print("\n" + "="*80)
    print("검증 결과 요약")
    print("="*80)

    if len(all_results) == 0:
        print("❌ 검증된 영상이 없습니다!")
        return

    # 데이터셋별 통계
    for dataset_name in ['train', 'validation', 'test']:
        dataset_results = [r for r in all_results if r['dataset'] == dataset_name]

        if len(dataset_results) == 0:
            continue

        print(f"\n[{dataset_name.upper()}] {len(dataset_results)}개 영상")

        # 어깨 중심 오차
        center_errors = [r['center_error_mean'] for r in dataset_results]
        center_error_max_list = [r['center_error_max'] for r in dataset_results]
        print(f"  어깨 중심 오차 (평균):")
        print(f"    전체 평균: {np.mean(center_errors):.10f}")
        print(f"    최대값: {np.max(center_error_max_list):.10f}")

        # 어깨 수평도
        y_diffs = [r['y_diff_mean'] for r in dataset_results]
        y_diff_max_list = [r['y_diff_max'] for r in dataset_results]
        print(f"  어깨 수평도 (평균):")
        print(f"    전체 평균: {np.mean(y_diffs):.10f}")
        print(f"    최대값: {np.max(y_diff_max_list):.10f}")

        # 어깨 너비
        shoulder_widths = [r['shoulder_width_mean'] for r in dataset_results]
        shoulder_width_stds = [r['shoulder_width_std'] for r in dataset_results]
        print(f"  어깨 너비:")
        print(f"    전체 평균: {np.mean(shoulder_widths):.10f}")
        print(f"    표준편차 평균: {np.mean(shoulder_width_stds):.10f}")
        print(f"    최소: {np.min([r['shoulder_width_min'] for r in dataset_results]):.10f}")
        print(f"    최대: {np.max([r['shoulder_width_max'] for r in dataset_results]):.10f}")

    # 문제가 있는 영상 찾기
    print("\n" + "="*80)
    print("이상치 검사")
    print("="*80)

    problematic_videos = []

    for result in all_results:
        issues = []

        # 어깨 중심이 원점에서 너무 멀리 떨어진 경우
        if result['center_error_mean'] > 1e-6:
            issues.append(f"중심 오차 큼: {result['center_error_mean']:.10f}")

        # 어깨가 수평이 아닌 경우
        if result['y_diff_mean'] > 1e-6:
            issues.append(f"수평도 나쁨: {result['y_diff_mean']:.10f}")

        # 어깨 너비가 1.0이 아닌 경우
        if abs(result['shoulder_width_mean'] - 1.0) > 1e-6:
            issues.append(f"너비 오차: {result['shoulder_width_mean']:.10f}")

        if issues:
            problematic_videos.append({
                'video': result['video_name'],
                'dataset': result['dataset'],
                'issues': issues
            })

    if len(problematic_videos) > 0:
        print(f"\n⚠️ 문제 발견: {len(problematic_videos)}개 영상")
        for pv in problematic_videos[:10]:  # 최대 10개만 출력
            print(f"  - [{pv['dataset']}] {pv['video']}")
            for issue in pv['issues']:
                print(f"      {issue}")
    else:
        print(f"\n✅ 모든 영상 정상!")

    # 실패한 영상
    if len(failed_videos) > 0:
        print("\n" + "="*80)
        print(f"처리 실패: {len(failed_videos)}개 영상")
        print("="*80)
        for fv in failed_videos:
            print(f"  [{fv['dataset']}] {fv['file']}: {fv['error']}")

    # 최종 판정
    print("\n" + "="*80)
    print("최종 판정")
    print("="*80)

    total_videos = len(all_results)
    success_videos = total_videos - len(problematic_videos)
    success_rate = (success_videos / total_videos * 100) if total_videos > 0 else 0

    print(f"총 검증 영상: {total_videos}개")
    print(f"성공: {success_videos}개")
    print(f"문제 있음: {len(problematic_videos)}개")
    print(f"처리 실패: {len(failed_videos)}개")
    print(f"성공률: {success_rate:.2f}%")

    if success_rate >= 95.0 and len(failed_videos) == 0:
        print(f"\n🎉 정규화가 전체 데이터셋에서 올바르게 작동합니다!")
        print(f"전체 데이터셋 재처리를 진행해도 됩니다.")
    elif success_rate >= 90.0:
        print(f"\n⚠️ 대부분 정상이지만 일부 영상에 문제가 있습니다.")
        print(f"문제 영상을 확인하고 재처리를 진행하세요.")
    else:
        print(f"\n❌ 정규화에 심각한 문제가 있습니다!")
        print(f"코드를 다시 점검해야 합니다.")

    # 결과를 CSV로 저장
    import csv
    output_csv = Path("D:/vector-quantize-pytorch/gym_vault_vq_system/validation_all_results.csv")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        if len(all_results) > 0:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n✓ 상세 결과 저장: {output_csv}")


if __name__ == "__main__":
    main()
