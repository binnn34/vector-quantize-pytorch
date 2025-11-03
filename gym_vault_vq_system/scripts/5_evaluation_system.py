#!/usr/bin/env python3
"""
도마 포즈 자동 평가 시스템
- 훈련된 VQ-VAE 모델을 활용한 무인 채점
- 코드북 패턴 분석을 통한 자세 품질 평가
- 상세한 분석 리포트 및 개선 제안
- 실시간 포즈 점수 생성
"""

import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import cv2

# 로컬 모듈 임포트
from gym_vault_vq_system.vq_model import GymVaultVQVAE
from gym_vault_vq_system.data_preprocessing import GymVaultDataPreprocessor
from gym_vault_vq_system.pose_extraction import GymVaultPoseExtractor


class VaultPoseEvaluator:
    """도마 포즈 자동 평가기"""

    def __init__(self, model_path, config_path=None):
        """
        Args:
            model_path: 훈련된 VQ-VAE 모델 경로
            config_path: 설정 파일 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 로드
        self.load_model(model_path, config_path)

        # 평가 기준 설정
        self.setup_evaluation_criteria()

        # 포즈 추출기 (필요시)
        self.pose_extractor = None

        print(f"도마 포즈 평가기 초기화 완료 - 디바이스: {self.device}")

    def load_model(self, model_path, config_path):
        """훈련된 모델 로드"""
        print(f"모델 로드: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']

        # 모델 초기화
        self.model = GymVaultVQVAE(
            input_dim=config['input_dim'],
            codebook_size=config['codebook_size'],
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)

        # 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 설정 저장
        self.config = config
        self.training_metrics = checkpoint.get('metrics', {})

        print(f"✅ 모델 로드 완료")
        print(f"코드북 크기: {config['codebook_size']}")
        print(f"훈련 에포크: {checkpoint['epoch']}")

    def setup_evaluation_criteria(self):
        """평가 기준 설정"""
        # 체조 도마 평가 요소들
        self.evaluation_criteria = {
            'technical_execution': {
                'weight': 0.40,
                'description': '기술 실행의 정확성과 완성도',
                'components': {
                    'body_alignment': 0.25,    # 몸체 정렬
                    'landing_stability': 0.25, # 착지 안정성
                    'flight_height': 0.20,     # 비행 높이
                    'rotation_control': 0.30    # 회전 제어
                }
            },
            'artistry': {
                'weight': 0.25,
                'description': '예술성과 표현력',
                'components': {
                    'fluidity': 0.40,          # 동작의 유연성
                    'rhythm': 0.30,            # 리듬감
                    'extension': 0.30          # 동작의 확장성
                }
            },
            'difficulty': {
                'weight': 0.20,
                'description': '기술의 난이도',
                'components': {
                    'complexity': 0.60,        # 복잡성
                    'innovation': 0.40         # 창의성
                }
            },
            'consistency': {
                'weight': 0.15,
                'description': '동작의 일관성',
                'components': {
                    'temporal_consistency': 0.50,  # 시간적 일관성
                    'spatial_consistency': 0.50    # 공간적 일관성
                }
            }
        }

        # 점수 범위 설정
        self.score_ranges = {
            'excellent': (9.5, 10.0),    # 뛰어남
            'good': (8.5, 9.5),          # 좋음
            'average': (7.0, 8.5),       # 보통
            'poor': (5.0, 7.0),          # 부족
            'very_poor': (0.0, 5.0)      # 매우 부족
        }

    def extract_pose_from_video(self, video_path):
        """비디오에서 포즈 추출"""
        if self.pose_extractor is None:
            self.pose_extractor = GymVaultPoseExtractor(use_3d=True)

        poses = self.pose_extractor.extract_video_poses(video_path)
        if poses is None:
            raise ValueError(f"포즈 추출 실패: {video_path}")

        return poses

    def preprocess_poses(self, poses):
        """포즈 전처리"""
        preprocessor = GymVaultDataPreprocessor(
            sequence_length=64,
            normalize_method='root_relative'
        )

        # 정규화
        normalized_poses = preprocessor.normalize_pose_sequence(poses)

        # 시퀀스 길이 통일
        resized_poses = preprocessor.resize_sequence(normalized_poses)

        # 평면화 (64, 17, 3) → (64, 51)
        flattened_poses = resized_poses.reshape(resized_poses.shape[0], -1)

        return torch.FloatTensor(flattened_poses).unsqueeze(0)  # (1, 64, 51)

    def encode_pose_sequence(self, pose_tensor):
        """포즈 시퀀스를 코드북으로 인코딩"""
        with torch.no_grad():
            pose_tensor = pose_tensor.to(self.device)

            # VQ-VAE 인코딩
            output = self.model(pose_tensor)

            return {
                'indices': output['indices'].cpu().numpy(),          # 코드북 인덱스
                'quantized': output['quantized'].cpu().numpy(),      # 양자화된 벡터
                'reconstructed': output['reconstructed'].cpu().numpy(), # 재구성된 포즈
                'vq_loss': output['vq_loss'].item(),                 # VQ 손실
                'commitment_loss': output['commitment_loss'].item()   # 커밋먼트 손실
            }

    def calculate_reconstruction_quality(self, original, reconstructed):
        """재구성 품질 평가"""
        # MSE 기반 재구성 오차
        mse = np.mean((original - reconstructed) ** 2)

        # 관절별 오차 분석
        original_reshaped = original.reshape(-1, 17, 3)
        reconstructed_reshaped = reconstructed.reshape(-1, 17, 3)

        joint_errors = np.mean((original_reshaped - reconstructed_reshaped) ** 2, axis=(0, 2))

        # 프레임별 오차
        frame_errors = np.mean((original - reconstructed) ** 2, axis=1)

        return {
            'overall_mse': mse,
            'joint_errors': joint_errors,
            'frame_errors': frame_errors,
            'quality_score': max(0, 10 - mse * 1000)  # MSE를 10점 스케일로 변환
        }

    def analyze_code_patterns(self, indices):
        """코드북 패턴 분석"""
        # 코드 사용 빈도
        unique_codes, counts = np.unique(indices, return_counts=True)
        code_frequency = dict(zip(unique_codes, counts))

        # 시간적 일관성 (연속된 프레임의 코드 변화)
        code_changes = np.sum(np.diff(indices.flatten()) != 0)
        temporal_consistency = 1.0 - (code_changes / len(indices.flatten()))

        # 코드 다양성
        diversity = len(unique_codes) / self.config['codebook_size']

        # 패턴 복잡도
        complexity = len(unique_codes) / len(indices.flatten())

        return {
            'code_frequency': code_frequency,
            'unique_codes': len(unique_codes),
            'temporal_consistency': temporal_consistency,
            'diversity': diversity,
            'complexity': complexity,
            'total_transitions': code_changes
        }

    def evaluate_technical_execution(self, pose_data, code_analysis):
        """기술 실행 평가"""
        scores = {}

        # 몸체 정렬 (재구성 품질 기반)
        alignment_score = pose_data['reconstruction_quality']['quality_score']
        scores['body_alignment'] = min(10, max(0, alignment_score))

        # 착지 안정성 (마지막 프레임들의 일관성)
        landing_frames = pose_data['reconstructed'][0, -10:]  # 마지막 10프레임
        landing_variance = np.var(landing_frames, axis=0).mean()
        scores['landing_stability'] = min(10, max(0, 10 - landing_variance * 100))

        # 비행 높이 (중간 프레임들의 y좌표 분석)
        middle_frames = pose_data['reconstructed'][0, 20:40]
        avg_height = np.mean(middle_frames.reshape(-1, 17, 3)[:, :, 1])  # y좌표
        scores['flight_height'] = min(10, max(0, (avg_height + 1) * 5))  # 정규화된 높이

        # 회전 제어 (시간적 일관성)
        scores['rotation_control'] = code_analysis['temporal_consistency'] * 10

        # 가중 평균
        weights = self.evaluation_criteria['technical_execution']['components']
        technical_score = sum(scores[k] * weights[k] for k in scores.keys())

        return technical_score, scores

    def evaluate_artistry(self, pose_data, code_analysis):
        """예술성 평가"""
        scores = {}

        # 유연성 (동작의 부드러움)
        pose_sequence = pose_data['reconstructed'][0]
        velocity = np.diff(pose_sequence, axis=0)
        smoothness = 1.0 / (1.0 + np.std(velocity))
        scores['fluidity'] = min(10, smoothness * 10)

        # 리듬감 (코드 변화의 주기성)
        code_transitions = np.diff(pose_data['indices'].flatten())
        rhythm_score = 1.0 / (1.0 + np.std(code_transitions))
        scores['rhythm'] = min(10, rhythm_score * 10)

        # 확장성 (동작의 크기와 범위)
        pose_range = np.ptp(pose_sequence, axis=0)  # 각 차원의 범위
        extension_score = np.mean(pose_range) * 5
        scores['extension'] = min(10, max(0, extension_score))

        # 가중 평균
        weights = self.evaluation_criteria['artistry']['components']
        artistry_score = sum(scores[k] * weights[k] for k in scores.keys())

        return artistry_score, scores

    def evaluate_difficulty(self, pose_data, code_analysis):
        """난이도 평가"""
        scores = {}

        # 복잡성 (사용된 코드의 다양성)
        complexity_score = code_analysis['diversity'] * 10
        scores['complexity'] = min(10, complexity_score)

        # 창의성 (희귀한 코드 패턴 사용)
        rare_codes = sum(1 for count in code_analysis['code_frequency'].values()
                        if count <= 2)
        innovation_score = (rare_codes / max(1, len(code_analysis['code_frequency']))) * 10
        scores['innovation'] = min(10, innovation_score)

        # 가중 평균
        weights = self.evaluation_criteria['difficulty']['components']
        difficulty_score = sum(scores[k] * weights[k] for k in scores.keys())

        return difficulty_score, scores

    def evaluate_consistency(self, pose_data, code_analysis):
        """일관성 평가"""
        scores = {}

        # 시간적 일관성
        scores['temporal_consistency'] = code_analysis['temporal_consistency'] * 10

        # 공간적 일관성 (좌우 대칭성)
        pose_sequence = pose_data['reconstructed'][0].reshape(-1, 17, 3)
        left_joints = [1, 3, 5, 7, 9, 11, 13, 15]  # 왼쪽 관절
        right_joints = [2, 4, 6, 8, 10, 12, 14, 16]  # 오른쪽 관절

        left_poses = pose_sequence[:, left_joints]
        right_poses = pose_sequence[:, right_joints]
        right_poses[:, :, 0] *= -1  # x좌표 반전

        symmetry_error = np.mean(np.abs(left_poses - right_poses))
        scores['spatial_consistency'] = min(10, max(0, 10 - symmetry_error * 20))

        # 가중 평균
        weights = self.evaluation_criteria['consistency']['components']
        consistency_score = sum(scores[k] * weights[k] for k in scores.keys())

        return consistency_score, scores

    def generate_overall_score(self, category_scores):
        """전체 점수 생성"""
        weights = {k: v['weight'] for k, v in self.evaluation_criteria.items()}

        overall_score = sum(category_scores[k] * weights[k] for k in category_scores.keys())

        # 점수 등급 결정
        grade = 'very_poor'
        for grade_name, (min_score, max_score) in self.score_ranges.items():
            if min_score <= overall_score <= max_score:
                grade = grade_name
                break

        return overall_score, grade

    def evaluate_pose_sequence(self, pose_input, input_type='tensor'):
        """포즈 시퀀스 종합 평가"""
        print("포즈 시퀀스 평가 시작...")

        # 입력 처리
        if input_type == 'video':
            poses = self.extract_pose_from_video(pose_input)
            pose_tensor = self.preprocess_poses(poses)
        elif input_type == 'poses':
            pose_tensor = self.preprocess_poses(pose_input)
        else:  # tensor
            pose_tensor = pose_input

        # 인코딩
        encoded_data = self.encode_pose_sequence(pose_tensor)

        # 재구성 품질 평가
        original = pose_tensor.numpy()
        reconstructed = encoded_data['reconstructed']
        reconstruction_quality = self.calculate_reconstruction_quality(original, reconstructed)

        # 코드 패턴 분석
        code_analysis = self.analyze_code_patterns(encoded_data['indices'])

        # 포즈 데이터 통합
        pose_data = {
            'original': original,
            'reconstructed': reconstructed,
            'indices': encoded_data['indices'],
            'vq_loss': encoded_data['vq_loss'],
            'commitment_loss': encoded_data['commitment_loss'],
            'reconstruction_quality': reconstruction_quality
        }

        # 각 범주별 평가
        print("기술 실행 평가...")
        technical_score, technical_details = self.evaluate_technical_execution(pose_data, code_analysis)

        print("예술성 평가...")
        artistry_score, artistry_details = self.evaluate_artistry(pose_data, code_analysis)

        print("난이도 평가...")
        difficulty_score, difficulty_details = self.evaluate_difficulty(pose_data, code_analysis)

        print("일관성 평가...")
        consistency_score, consistency_details = self.evaluate_consistency(pose_data, code_analysis)

        # 범주별 점수
        category_scores = {
            'technical_execution': technical_score,
            'artistry': artistry_score,
            'difficulty': difficulty_score,
            'consistency': consistency_score
        }

        # 전체 점수
        overall_score, grade = self.generate_overall_score(category_scores)

        # 결과 통합
        evaluation_result = {
            'overall_score': overall_score,
            'grade': grade,
            'category_scores': category_scores,
            'detailed_scores': {
                'technical_execution': technical_details,
                'artistry': artistry_details,
                'difficulty': difficulty_details,
                'consistency': consistency_details
            },
            'pose_data': pose_data,
            'code_analysis': code_analysis,
            'evaluation_timestamp': datetime.now().isoformat()
        }

        print(f"✅ 평가 완료! 전체 점수: {overall_score:.2f} (등급: {grade})")

        return evaluation_result

    def generate_detailed_report(self, evaluation_result, output_path=None):
        """상세 평가 리포트 생성"""
        if output_path is None:
            output_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 리포트 데이터 준비
        report = {
            'summary': {
                'overall_score': evaluation_result['overall_score'],
                'grade': evaluation_result['grade'],
                'evaluation_date': evaluation_result['evaluation_timestamp']
            },
            'category_analysis': {},
            'technical_analysis': {
                'reconstruction_quality': evaluation_result['pose_data']['reconstruction_quality'],
                'code_analysis': evaluation_result['code_analysis'],
                'vq_metrics': {
                    'vq_loss': evaluation_result['pose_data']['vq_loss'],
                    'commitment_loss': evaluation_result['pose_data']['commitment_loss']
                }
            },
            'recommendations': self.generate_recommendations(evaluation_result)
        }

        # 범주별 분석
        for category, score in evaluation_result['category_scores'].items():
            criteria = self.evaluation_criteria[category]
            report['category_analysis'][category] = {
                'score': score,
                'weight': criteria['weight'],
                'description': criteria['description'],
                'component_scores': evaluation_result['detailed_scores'][category],
                'contribution_to_total': score * criteria['weight']
            }

        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"상세 리포트 저장: {output_path}")
        return report

    def generate_recommendations(self, evaluation_result):
        """개선 제안 생성"""
        recommendations = []

        category_scores = evaluation_result['category_scores']
        detailed_scores = evaluation_result['detailed_scores']

        # 가장 낮은 점수 범주 식별
        lowest_category = min(category_scores.items(), key=lambda x: x[1])

        if lowest_category[1] < 7.0:
            recommendations.append({
                'priority': 'high',
                'category': lowest_category[0],
                'issue': f"{self.evaluation_criteria[lowest_category[0]]['description']} 점수가 낮음 ({lowest_category[1]:.2f})",
                'suggestion': self.get_category_suggestions(lowest_category[0], detailed_scores[lowest_category[0]])
            })

        # 기술 실행 세부 분석
        tech_details = detailed_scores['technical_execution']
        if tech_details['landing_stability'] < 6.0:
            recommendations.append({
                'priority': 'high',
                'category': 'technical_execution',
                'issue': '착지 안정성 부족',
                'suggestion': '착지 시 균형감과 자세 유지에 더 집중하세요.'
            })

        if tech_details['body_alignment'] < 6.0:
            recommendations.append({
                'priority': 'medium',
                'category': 'technical_execution',
                'issue': '몸체 정렬 부족',
                'suggestion': '전체 동작에서 몸의 선을 일직선으로 유지하도록 연습하세요.'
            })

        # 코드 분석 기반 제안
        code_analysis = evaluation_result['code_analysis']
        if code_analysis['temporal_consistency'] < 0.7:
            recommendations.append({
                'priority': 'medium',
                'category': 'consistency',
                'issue': '동작의 일관성 부족',
                'suggestion': '연속된 동작들 간의 매끄러운 연결에 집중하세요.'
            })

        return recommendations

    def get_category_suggestions(self, category, detailed_scores):
        """범주별 구체적 제안"""
        suggestions = {
            'technical_execution': "기본기 강화와 정확한 자세 연습에 집중하세요.",
            'artistry': "동작의 표현력과 연결성을 향상시키세요.",
            'difficulty': "더 복잡하고 창의적인 기술에 도전해보세요.",
            'consistency': "반복 연습을 통해 동작의 안정성을 높이세요."
        }
        return suggestions.get(category, "해당 영역의 기본기를 강화하세요.")

    def visualize_evaluation(self, evaluation_result, output_path=None):
        """평가 결과 시각화"""
        if output_path is None:
            output_path = f"evaluation_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 전체 점수 게이지
        overall_score = evaluation_result['overall_score']
        ax = axes[0, 0]
        ax.pie([overall_score, 10-overall_score],
               labels=[f'{overall_score:.1f}', ''],
               colors=['#2ecc71', '#ecf0f1'],
               startangle=90)
        ax.set_title(f'전체 점수\n등급: {evaluation_result["grade"].upper()}', fontsize=14, fontweight='bold')

        # 2. 범주별 점수 바차트
        categories = list(evaluation_result['category_scores'].keys())
        scores = list(evaluation_result['category_scores'].values())

        ax = axes[0, 1]
        bars = ax.bar(range(len(categories)), scores, color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=0, fontsize=10)
        ax.set_ylim(0, 10)
        ax.set_title('범주별 점수', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{scores[i]:.1f}', ha='center', va='bottom', fontweight='bold')

        # 3. 코드북 사용 패턴
        code_freq = evaluation_result['code_analysis']['code_frequency']
        ax = axes[0, 2]
        if code_freq:
            codes, frequencies = zip(*sorted(code_freq.items()))
            ax.bar(range(len(codes)), frequencies, color='#1abc9c')
            ax.set_title('코드북 사용 패턴', fontsize=14, fontweight='bold')
            ax.set_xlabel('코드 인덱스')
            ax.set_ylabel('사용 빈도')

        # 4. 재구성 오차 (관절별)
        joint_errors = evaluation_result['pose_data']['reconstruction_quality']['joint_errors']
        ax = axes[1, 0]
        joint_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder',
                      'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
                      'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

        bars = ax.bar(range(len(joint_errors)), joint_errors, color='#e67e22')
        ax.set_xticks(range(len(joint_errors)))
        ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
        ax.set_title('관절별 재구성 오차', fontsize=14, fontweight='bold')
        ax.set_ylabel('MSE')

        # 5. 시간별 프레임 오차
        frame_errors = evaluation_result['pose_data']['reconstruction_quality']['frame_errors']
        ax = axes[1, 1]
        ax.plot(frame_errors, color='#c0392b', linewidth=2)
        ax.set_title('프레임별 재구성 오차', fontsize=14, fontweight='bold')
        ax.set_xlabel('프레임')
        ax.set_ylabel('MSE')
        ax.grid(True, alpha=0.3)

        # 6. 평가 요약 텍스트
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = f"""
평가 요약

전체 점수: {overall_score:.2f}/10
등급: {evaluation_result['grade'].upper()}

범주별 점수:
• 기술 실행: {scores[0]:.1f}/10
• 예술성: {scores[1]:.1f}/10
• 난이도: {scores[2]:.1f}/10
• 일관성: {scores[3]:.1f}/10

기술적 분석:
• 사용된 코드: {evaluation_result['code_analysis']['unique_codes']}개
• 시간적 일관성: {evaluation_result['code_analysis']['temporal_consistency']:.1%}
• 재구성 품질: {evaluation_result['pose_data']['reconstruction_quality']['quality_score']:.1f}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"시각화 결과 저장: {output_path}")


def main():
    """메인 실행 함수"""

    # 설정
    MODEL_PATH = "D:/vector-quantize-pytorch/gym_vault_vq_system/checkpoints/best_model.pth"
    TEST_DATA_PATH = "D:/vector-quantize-pytorch/gym_vault_vq_system/processed_data.pkl"
    OUTPUT_DIR = "D:/vector-quantize-pytorch/gym_vault_vq_system/evaluations"

    # 출력 디렉토리 생성
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=== 도마 포즈 자동 평가 시스템 ===")

    # 모델 확인
    if not Path(MODEL_PATH).exists():
        print(f"❌ 훈련된 모델이 없습니다: {MODEL_PATH}")
        print("먼저 4_train_codebook.py를 실행하여 모델을 훈련하세요.")
        return

    # 평가기 초기화
    evaluator = VaultPoseEvaluator(MODEL_PATH)

    # 테스트 데이터 로드
    if Path(TEST_DATA_PATH).exists():
        print(f"테스트 데이터 로드: {TEST_DATA_PATH}")

        with open(TEST_DATA_PATH, 'rb') as f:
            data = pickle.load(f)

        test_poses = data['test_poses']
        print(f"테스트 샘플 수: {len(test_poses)}")

        # 샘플 평가 (처음 3개)
        sample_count = min(3, len(test_poses))

        for i in range(sample_count):
            print(f"\n--- 샘플 {i+1} 평가 ---")

            # 포즈 텐서 준비 (이미 전처리됨)
            pose_tensor = torch.FloatTensor(test_poses[i:i+1])  # (1, 64, 17, 3)
            pose_tensor = pose_tensor.reshape(1, 64, -1)  # (1, 64, 51)

            # 평가 실행
            result = evaluator.evaluate_pose_sequence(pose_tensor, input_type='tensor')

            # 리포트 생성
            report_path = f"{OUTPUT_DIR}/sample_{i+1}_report.json"
            evaluator.generate_detailed_report(result, report_path)

            # 시각화
            viz_path = f"{OUTPUT_DIR}/sample_{i+1}_visualization.png"
            evaluator.visualize_evaluation(result, viz_path)

            print(f"✅ 샘플 {i+1} 평가 완료")
            print(f"   점수: {result['overall_score']:.2f} (등급: {result['grade']})")
            print(f"   리포트: {report_path}")
            print(f"   시각화: {viz_path}")

    else:
        print(f"❌ 테스트 데이터가 없습니다: {TEST_DATA_PATH}")
        print("먼저 2_data_preprocessing.py를 실행하세요.")

    print(f"\n✅ 평가 시스템 테스트 완료!")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("\n=== 사용법 ===")
    print("1. 새로운 비디오 평가: evaluator.evaluate_pose_sequence('video.avi', 'video')")
    print("2. 포즈 데이터 평가: evaluator.evaluate_pose_sequence(poses, 'poses')")
    print("3. 상세 리포트 생성: evaluator.generate_detailed_report(result)")


if __name__ == "__main__":
    main()