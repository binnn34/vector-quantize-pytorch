# Project Memory: Gym Vault VQ System

## 프로젝트 개요
- **목표**: 도마 경기 영상에서 2D pose 기반 점수 예측 시스템 구축
- **핵심 기술**: VQ-VAE (Vector Quantized Variational AutoEncoder) + Score Prediction Model
- **데이터셋**: AQA-7 (Vault class 2)

---

## 📁 프로젝트 구조

```
D:\vector-quantize-pytorch\
├── AQA-7/                          # 원본 데이터셋
│   ├── train_labels.csv            # Train 68개 영상 라벨 (점수 포함)
│   ├── validation_labels.csv       # Validation 라벨
│   └── test_labels.csv             # Test 라벨 (bad/good 분류 포함)
│
├── gym_vault_vq_system/
│   ├── 2d_pose_data/              # 원본 2D pose 데이터 (정규화 전)
│   │   ├── train/                 # 68개 *_poses.pkl 파일
│   │   ├── validation/            # 아직 비어있음
│   │   └── test/                  # 아직 비어있음
│   │
│   ├── normalized_poses/          # 정규화된 2D pose 데이터
│   │   ├── train/                 # 68개 *_normalized.pkl 파일
│   │   │   └── 003_normalized.pkl, 007_normalized.pkl, ...
│   │   ├── validation/            # 아직 비어있음
│   │   ├── test/                  # 아직 비어있음
│   │   └── train_normalization_stats.csv
│   │
│   └── models/
│       └── vqvae_pose.py          # VQ-VAE 모델 구현 완료
│
└── vector_quantize_pytorch/       # VQ 라이브러리 (설치 완료)
```

---

## ✅ 완료된 작업

### 1. 데이터 준비
- **Train 68개 영상 2D pose 추출 완료**
  - MMPose (HRNet + RTMDet) 사용
  - 17 keypoints (COCO format)
  - 저장 형식: `{video_name}_poses.pkl`

- **CSV 라벨링 완료**
  - train_labels.csv: 68개 영상, 점수 포함
  - validation_labels.csv: 라벨 준비됨
  - test_labels.csv: bad/good 분류 포함

- **정규화 완료** (어깨 너비 기준)
  - normalized_poses/train/ 에 저장
  - 파일 구조:
    ```python
    {
        'video_name': '003.avi',
        'score': 16.37,
        'poses': np.array(103, 17, 2),  # 정규화된 좌표
        'normalization_method': 'shoulder_width',
        'keypoint_names': [...],
        ...
    }
    ```

### 2. VQ-VAE 모델 구현 완료
- **파일**: `models/vqvae_pose.py`
- **구성**:
  - `PoseEncoder`: (B, T, 17, 2) → (B, T, 128) LSTM 기반
  - `VectorQuantize`: 512 codebook, EMA decay=0.8
  - `PoseDecoder`: (B, T, 128) → (B, T, 17, 2) LSTM 기반
  - `PoseVQVAE`: 전체 모델 + encode/decode 메서드

- **패키지 설치 완료**: `vector_quantize_pytorch` (pip install -e .)

---

## 🔄 현재 진행 단계

### Phase 1: VQ-VAE Codebook 학습
**목표**: Train 68개 영상으로 512개 "자세 사전(Codebook)" 생성

**다음 작업**:
1. ✅ VQ-VAE 모델 구현 완료
2. ⏳ 데이터 로더 작성 (PoseDataset + DataLoader)
3. ⏳ 학습 스크립트 작성 (train_vqvae.py)
4. ⏳ 학습 실행 (100 epochs, 예상 2-3시간)
5. ⏳ Codebook 검증 (사용률, perplexity, 재구성 품질)

---

## 📋 전체 파이프라인

```
원본 비디오 (*.avi)
    ↓ (완료)
2D Poses (T, 17, 2) + Score Label
    ↓ (완료) 정규화 (어깨 너비 기준)
Normalized Poses
    ↓ (진행 예정) VQ-VAE Encoder
Continuous Latent (T, 128)
    ↓ (진행 예정) Vector Quantization
Discrete Codes (T,) → [3, 45, 67, ...]
    ↓ (진행 예정) VQ-VAE Decoder
Reconstructed Poses (검증용)
    ↓ (향후) Score Prediction Model (Transformer/LSTM)
Predicted Score (0-100)
```

---

## 🎯 점수 기반 학습 플랜

### VQ-VAE 학습 단계
```python
# Loss = Reconstruction Loss + VQ Loss

for epoch in range(100):
    for batch in train_loader:
        poses = batch['poses']  # (B, T, 17, 2)

        # Forward
        recon, vq_loss, perplexity, codes = vqvae(poses)

        # Loss
        recon_loss = MSE(poses, recon)
        total_loss = recon_loss + vq_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### 점수 예측 모델 (VQ-VAE 학습 후)
```python
# Input: Discrete Codes (T,)
# Output: Score (float)

codes = vqvae.encode(poses)  # (B, T)
embeddings = vqvae.vq.codebook[codes]  # (B, T, 128)

# Transformer/LSTM
score_pred = score_model(embeddings)  # (B, 1)

# Loss
loss = MSE(score_pred, score_label)
```

---

## ⚠️ 중요 사항

### 데이터 상태
- ✅ Train: 68개 pose 추출 + 정규화 완료
- ❌ Validation: pose 추출 아직 안됨
- ❌ Test: pose 추출 아직 안됨
- **전략**: Train으로 전체 파이프라인 검증 후 validation/test 진행

### 정규화 이슈
- normalized_poses 파일에서 좌표 범위 확인 필요
- 어깨 너비가 여전히 52.47 (1.0이 되어야 정상)
- → 정규화 스크립트 재확인 필요할 수 있음

### Codebook 학습 목표
- **Codebook size**: 512개
- **사용률 목표**: 80% 이상 (dead codes 최소화)
- **Perplexity 목표**: 높을수록 좋음 (다양한 코드 사용)
- **Reconstruction quality**: MSE < 0.1 목표

---

## 📝 교수님 보고 내용

### 현재 진행 상황
- Train 68개 영상 2D pose 추출 완료
- 정규화 완료 (어깨 너비 기준)
- VQ-VAE 모델 구현 완료

### 다음 단계
- VQ-VAE Codebook 학습 (2-3시간 예상)
- 점수 예측 모델 학습 (1-2시간 예상)
- 전체 완료 목표: 2-3일 내

### 기대 효과
1. 데이터 압축: 프레임당 34차원 → 1개 discrete code
2. 의미있는 표현: 512개 코드가 "대표 자세" 학습
3. 강건한 학습: Raw 좌표 대신 discrete codes로 점수 예측

---

## 🔗 관련 파일 위치

- **모델**: `D:\vector-quantize-pytorch\gym_vault_vq_system\models\vqvae_pose.py`
- **Train 데이터**: `D:\vector-quantize-pytorch\gym_vault_vq_system\normalized_poses\train\`
- **라벨**: `D:\vector-quantize-pytorch\AQA-7\train_labels.csv`
- **정규화 통계**: `D:\vector-quantize-pytorch\gym_vault_vq_system\normalized_poses\train_normalization_stats.csv`

---

## 💡 다음 세션에서 할 일

1. 데이터 로더 구현 (PoseDataset 클래스)
2. 학습 스크립트 작성 (train_vqvae.py)
3. 학습 실행 및 모니터링
4. Codebook 검증 및 시각화
5. (선택) 정규화 재확인 및 수정

---

*Last Updated: 2025-10-28*
*Status: VQ-VAE 학습 준비 단계*
