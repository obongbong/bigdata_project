# shap_analysis.py

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv("data/병합된_NOX_데이터셋.csv")
data_encoded = pd.get_dummies(df, columns=["사업소", "호기"])

# 특성 추출 (타겟 제외)
X = data_encoded.drop(columns=["질소산화물(ppm)평균"])

# 2. 학습된 모델 불러오기
model = joblib.load("models/best_nox_model.pkl")

# 3. SHAP Explainer 구성
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 4. 요약 플롯 (전체 feature 영향도)
shap.summary_plot(shap_values, X, plot_type="bar")

# 5. 개별 예측에 대한 설명 (임의 샘플 선택)
sample_idx = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X.iloc[sample_idx],
    matplotlib=True
)
