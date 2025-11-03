#!/usr/bin/env python3
"""
도마 포즈 데이터 전처리
- 포즈 정규화 (루트 관절 기준, 스케일 정규화)
- 시퀀스 길이 통일
- 학습/검증 데이터 분할
- VQ 모델 입력 형태로 변환
"""

import os
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch


class GymVaultDataPreprocessor:
    """도마 포즈 데이터 전처리기"""

    def __init__(self, sequence_length=64, normalize_method='root_relative'):
        """
        Args:
            sequence_length: 통일할 시퀀스 길이 (프레임 수)
            normalize_method: 정규화 방법 ('root_relative', 'global', 'bone_length')
        """
        self.sequence_length = sequence_length
        self.normalize_method = normalize_method

        # COCO 17 키포인트 연결 (본 구조)
        self.skeleton_connections = [
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

        # 주요 관절 그룹
        self.joint_groups = {
            'head': [0, 1, 2, 3, 4],           # 머리
            'torso': [5, 6, 11, 12],           # 몸통
            'left_arm': [5, 7, 9],             # 왼팔
            'right_arm': [6, 8, 10],           # 오른팔
            'left_leg': [11, 13, 15],          # 왼다리
            'right_leg': [12, 14, 16]          # 오른다리
        }

    def load_pose_data(self, pose_dir):
        """추출된 포즈 데이터 로드"""
        pose_dir = Path(pose_dir)
        pose_files = list(pose_dir.glob('*_poses.pkl'))

        print(f"포즈 파일 {len(pose_files)}개 발견")

        all_poses = []
        metadata_list = []

        for pose_file in tqdm(pose_files, desc="포즈 데이터 로딩"):
            try:
                with open(pose_file, 'rb') as f:
                    data = pickle.load(f)

                poses = data['poses']  # (frames, keypoints, coords)

                if len(poses) > 0:
                    all_poses.append(poses)
                    metadata_list.append({
                        'file_name': pose_file.name,
                        'video_name': data['video_name'],
                        'original_length': len(poses),
                        'shape': poses.shape
                    })

            except Exception as e:
                print(f"파일 로드 실패 {pose_file}: {e}")

        print(f"성공적으로 로드된 포즈 데이터: {len(all_poses)}개")
        return all_poses, metadata_list

    def normalize_pose_sequence(self, poses):
        """포즈 시퀀스 정규화"""
        # poses: (frames, 17, 2 or 3)

        if self.normalize_method == 'root_relative':
            return self._normalize_root_relative(poses)
        elif self.normalize_method == 'global':
            return self._normalize_global(poses)
        elif self.normalize_method == 'bone_length':
            return self._normalize_bone_length(poses)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")

    def _normalize_root_relative(self, poses):
        """루트 관절(골반 중심) 기준 상대 좌표로 정규화"""
        normalized_poses = poses.copy()

        for frame_idx in range(len(poses)):
            frame_pose = poses[frame_idx]  # (17, coords)

            # 골반 중심점 계산 (left_hip + right_hip) / 2
            if frame_pose.shape[1] >= 2:  # 최소 x, y 좌표
                left_hip = frame_pose[11]   # left_hip
                right_hip = frame_pose[12]  # right_hip

                # 골반 중심 계산
                root_joint = (left_hip + right_hip) / 2.0

                # 모든 관절을 루트 기준으로 상대화
                normalized_poses[frame_idx] = frame_pose - root_joint

                # 스케일 정규화 (전체 포즈 크기로 나누기)
                pose_scale = np.linalg.norm(normalized_poses[frame_idx])
                if pose_scale > 0:
                    normalized_poses[frame_idx] = normalized_poses[frame_idx] / pose_scale

        return normalized_poses

    def _normalize_global(self, poses):
        """전역 정규화 (전체 시퀀스 기준)"""
        # 전체 시퀀스에서 최소/최대값으로 정규화
        poses_flat = poses.reshape(-1, poses.shape[-1])

        # 각 좌표 차원별로 정규화
        for dim in range(poses.shape[-1]):
            coord_data = poses_flat[:, dim]
            min_val, max_val = np.min(coord_data), np.max(coord_data)

            if max_val > min_val:
                poses[:, :, dim] = (poses[:, :, dim] - min_val) / (max_val - min_val)

        return poses

    def _normalize_bone_length(self, poses):
        """
        프레임별 어깨 기준 정규화 (교수님 피드백 반영)

        각 프레임마다 독립적으로:
        1. Translation: 어깨 중심을 원점으로
        2. Rotation: 어깨를 수평으로
        3. Scaling: 어깨 너비를 1.0으로

        이 방법으로 카메라 위치(Translation), 각도(Rotation), 줌(Scaling) 불변성 확보
        """
        normalized_poses = poses.copy()

        for frame_idx in range(len(poses)):
            frame_pose = poses[frame_idx].copy()

            # 어깨 키포인트 (COCO format: 5=left_shoulder, 6=right_shoulder)
            left_shoulder = frame_pose[5]
            right_shoulder = frame_pose[6]

            # Step 1: 어깨 중심 및 너비 계산
            shoulder_center = (left_shoulder + right_shoulder) / 2.0
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)

            # 어깨 검출 오류 처리 (너무 작은 값 방지)
            if shoulder_width < 1e-6:
                normalized_poses[frame_idx] = frame_pose
                continue

            # Step 2: Translation - 어깨 중심을 원점으로
            centered_pose = frame_pose - shoulder_center

            # Step 3: Rotation - 어깨를 수평으로 (y축 기준)
            # 회전 전 어깨 벡터 (centered 좌표계 기준)
            shoulder_vector = (right_shoulder - shoulder_center) - (left_shoulder - shoulder_center)
            shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])

            # 회전 행렬 (2D)
            cos_angle = np.cos(-shoulder_angle)
            sin_angle = np.sin(-shoulder_angle)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle],
                [sin_angle,  cos_angle]
            ])

            # 모든 키포인트 회전
            rotated_pose = np.zeros_like(centered_pose)
            for joint_idx in range(len(centered_pose)):
                rotated_pose[joint_idx] = rotation_matrix @ centered_pose[joint_idx]

            # Step 4: Scaling - 어깨 너비를 1.0으로
            normalized_poses[frame_idx] = rotated_pose / shoulder_width

        return normalized_poses

    def resize_sequence(self, poses, target_length=None):
        """시퀀스 길이 통일"""
        if target_length is None:
            target_length = self.sequence_length

        current_length = len(poses)

        if current_length == target_length:
            return poses

        elif current_length > target_length:
            # 다운샘플링: 균등 간격으로 선택
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return poses[indices]

        else:
            # 업샘플링: 선형 보간
            indices = np.linspace(0, current_length - 1, target_length)
            interpolated_poses = []

            for i in indices:
                if i == int(i):
                    # 정확한 인덱스
                    interpolated_poses.append(poses[int(i)])
                else:
                    # 선형 보간
                    lower_idx = int(np.floor(i))
                    upper_idx = int(np.ceil(i))
                    weight = i - lower_idx

                    if upper_idx < current_length:
                        interpolated_pose = (1 - weight) * poses[lower_idx] + weight * poses[upper_idx]
                        interpolated_poses.append(interpolated_pose)
                    else:
                        interpolated_poses.append(poses[lower_idx])

            return np.array(interpolated_poses)

    def extract_features(self, poses):
        """포즈에서 추가 특징 추출"""
        features = {}

        # 1. 관절별 속도 (프레임 간 차이)
        if len(poses) > 1:
            velocities = np.diff(poses, axis=0)  # (frames-1, 17, coords)
            features['velocities'] = velocities

        # 2. 관절별 가속도
        if len(poses) > 2:
            accelerations = np.diff(velocities, axis=0)  # (frames-2, 17, coords)
            features['accelerations'] = accelerations

        # 3. 본 각도 (주요 관절 연결 각도)
        bone_angles = []
        for frame_pose in poses:
            angles = self._calculate_bone_angles(frame_pose)
            bone_angles.append(angles)
        features['bone_angles'] = np.array(bone_angles)

        # 4. 관절별 거리 (루트 관절로부터)
        root_distances = []
        for frame_pose in poses:
            # 골반 중심
            root = (frame_pose[11] + frame_pose[12]) / 2.0
            distances = [np.linalg.norm(joint - root) for joint in frame_pose]
            root_distances.append(distances)
        features['root_distances'] = np.array(root_distances)

        return features

    def _calculate_bone_angles(self, pose):
        """주요 본 각도 계산"""
        angles = []

        # 주요 본 벡터들
        bone_vectors = []
        for start_joint, end_joint in self.skeleton_connections:
            if start_joint < len(pose) and end_joint < len(pose):
                vector = pose[end_joint] - pose[start_joint]
                bone_vectors.append(vector)

        # 인접한 본들 간의 각도 계산
        for i in range(len(bone_vectors) - 1):
            v1 = bone_vectors[i]
            v2 = bone_vectors[i + 1]

            # 벡터 정규화
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 수치 안정성
                angle = np.arccos(cos_angle)
                angles.append(angle)
            else:
                angles.append(0.0)

        return angles

    def create_training_data(self, all_poses, test_size=0.33, random_state=42):
        """학습/검증 데이터 생성"""
        print(f"총 {len(all_poses)}개 포즈 시퀀스 전처리 시작")

        processed_sequences = []
        sequence_features = []

        for i, poses in enumerate(tqdm(all_poses, desc="포즈 전처리")):
            try:
                # 1. 정규화
                normalized_poses = self.normalize_pose_sequence(poses)

                # 2. 시퀀스 길이 통일
                resized_poses = self.resize_sequence(normalized_poses)

                # 3. 특징 추출
                features = self.extract_features(resized_poses)

                # 4. 저장
                processed_sequences.append(resized_poses)
                sequence_features.append(features)

            except Exception as e:
                print(f"시퀀스 {i} 처리 실패: {e}")

        print(f"전처리 완료: {len(processed_sequences)}개 시퀀스")

        # 넘파이 배열로 변환
        processed_sequences = np.array(processed_sequences)  # (N, frames, 17, coords)
        print(f"전처리된 데이터 형태: {processed_sequences.shape}")

        # 학습/검증 분할
        train_indices, test_indices = train_test_split(
            range(len(processed_sequences)),
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        train_data = processed_sequences[train_indices]
        test_data = processed_sequences[test_indices]

        train_features = [sequence_features[i] for i in train_indices]
        test_features = [sequence_features[i] for i in test_indices]

        print(f"학습 데이터: {len(train_data)}개")
        print(f"검증 데이터: {len(test_data)}개")

        return {
            'train_poses': train_data,
            'test_poses': test_data,
            'train_features': train_features,
            'test_features': test_features,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'preprocessing_config': {
                'sequence_length': self.sequence_length,
                'normalize_method': self.normalize_method,
                'total_sequences': len(processed_sequences),
                'train_ratio': 1 - test_size,
                'test_ratio': test_size
            }
        }

    def save_processed_data(self, processed_data, output_path):
        """전처리된 데이터 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"전처리 데이터 저장: {output_path}")

        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)

        # 메타데이터도 JSON으로 별도 저장
        metadata = processed_data['preprocessing_config']
        metadata['train_shape'] = processed_data['train_poses'].shape
        metadata['test_shape'] = processed_data['test_poses'].shape

        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"메타데이터 저장: {metadata_path}")

    def load_processed_data(self, data_path):
        """전처리된 데이터 로드"""
        print(f"전처리 데이터 로드: {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        print(f"학습 데이터: {data['train_poses'].shape}")
        print(f"검증 데이터: {data['test_poses'].shape}")

        return data


def main():
    """메인 실행 함수"""

    # 경로 설정
    POSE_DIR = "D:/vector-quantize-pytorch/gym_vault_vq_system/pose_data"
    OUTPUT_PATH = "D:/vector-quantize-pytorch/gym_vault_vq_system/processed_data.pkl"

    # 전처리기 초기화
    preprocessor = GymVaultDataPreprocessor(
        sequence_length=64,  # 64프레임으로 통일
        normalize_method='root_relative'  # 루트 관절 기준 정규화
    )

    # 1. 포즈 데이터 로드
    all_poses, metadata = preprocessor.load_pose_data(POSE_DIR)

    if len(all_poses) == 0:
        print("❌ 포즈 데이터가 없습니다. 먼저 1_pose_extraction.py를 실행하세요.")
        return

    # 2. 전처리 및 학습/검증 분할
    processed_data = preprocessor.create_training_data(
        all_poses,
        test_size=0.33,  # 33% 검증용
        random_state=42
    )

    # 3. 저장
    preprocessor.save_processed_data(processed_data, OUTPUT_PATH)

    print(f"\n✅ 전처리 완료!")
    print(f"학습 데이터: {processed_data['train_poses'].shape}")
    print(f"검증 데이터: {processed_data['test_poses'].shape}")
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()