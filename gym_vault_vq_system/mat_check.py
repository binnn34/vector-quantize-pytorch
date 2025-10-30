# 파일: export_vault_class2.py
import scipy.io as sio
import numpy as np
import pandas as pd

# .mat 파일 경로
TRAIN = r"D:\vector-quantize-pytorch\AQA-7\Split_4\split_4_train_list.mat"
TEST  = r"D:\vector-quantize-pytorch\AQA-7\Split_4\split_4_test_list.mat"

# gym_vault(도마)의 class id
VAULT_CLASS_ID = 2

def load_three_col(path):
    """3열짜리 numpy 배열 읽기"""
    d = sio.loadmat(path)
    for k, v in d.items():
        if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
            return v.astype(float)
    raise RuntimeError(f"3열 배열을 찾지 못했습니다: {path}")

def filter_class(df_array, class_id):
    """특정 class id만 필터링"""
    mask = df_array[:, 0].astype(int) == class_id
    filtered = df_array[mask]
    return filtered

def save_csv(data, out_path, name):
    """CSV 저장"""
    df = pd.DataFrame(data, columns=["class_id", "video_idx", "score"])
    df[["class_id", "video_idx"]] = df[["class_id", "video_idx"]].astype(int)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] {name}: {len(df)}행  →  {out_path}")

# 데이터 불러오기
train = load_three_col(TRAIN)
test  = load_three_col(TEST)

# 도마(class_id=2)만 추출
train_vault = filter_class(train, VAULT_CLASS_ID)
test_vault  = filter_class(test,  VAULT_CLASS_ID)

# CSV로 저장
save_csv(train_vault, r"D:\vector-quantize-pytorch\AQA-7\vault_train_class2.csv", "train")
save_csv(test_vault,  r"D:\vector-quantize-pytorch\AQA-7\vault_test_class2.csv", "test")

print("\n🎯 도마(gym_vault, class_id=2) 데이터만 필터링 완료!")
