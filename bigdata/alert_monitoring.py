import pandas as pd
import numpy as np
import joblib
import os

# 1. 모델 및 피처 목록 로드
model = joblib.load("models/best_nox_model.pkl")

# 피처 리스트는 학습 시점의 X.columns 저장값
with open("models/feature_names.txt", "r", encoding="utf-8") as f:
    feature_names = [line.strip() for line in f.readlines()]

# 2. 데이터 로딩 및 동일한 전처리 적용
data = pd.read_csv("data/병합된_NOX_데이터셋.csv")
data_encoded = pd.get_dummies(data, columns=["사업소", "호기"])

# 누락된 피처 보정 (get_dummies 결과에 따라 일부 컬럼이 없을 수 있음)
for col in feature_names:
    if col not in data_encoded.columns:
        data_encoded[col] = 0

X = data_encoded[feature_names]  # 피처 순서 맞추기
y_true = data_encoded["질소산화물(ppm)평균"]

# 3. 예측 수행
y_pred = model.predict(X)

# 4. 규제 초과 감지
limit = 50
exceed_mask = y_pred > limit
exceed_cases = data_encoded[exceed_mask].copy()
exceed_cases["예측값"] = y_pred[exceed_mask]

# 5. 저장 또는 알림
os.makedirs("alerts", exist_ok=True)
if not exceed_cases.empty:
    exceed_cases.to_csv("alerts/nox_limit_exceeded.csv", index=False)
    print(f"🚨 {len(exceed_cases)}건의 초과 사례가 alerts 폴더에 저장되었습니다.")
else:
    print("✅ NOx 예측값이 모두 규제 기준 이하입니다.")
