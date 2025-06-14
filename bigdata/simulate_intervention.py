import pandas as pd
import joblib

# 1. 모델 및 feature 리스트 불러오기
model = joblib.load("models/best_nox_model.pkl")
with open("models/feature_names.txt", "r", encoding="utf-8") as f:
    feature_names = [line.strip() for line in f.readlines()]

# 2. 데이터 로드 및 인코딩
df = pd.read_csv("data/병합된_NOX_데이터셋.csv")
df_encoded = pd.get_dummies(df, columns=["사업소", "호기"])

# 누락된 컬럼 보정
for col in feature_names:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# 순서 정렬
df_encoded = df_encoded[feature_names]

# 3. 시뮬레이션 대상 샘플 선택
base_sample = df_encoded.sample(1, random_state=42)
target_variable = "유량"  # 조작 변수
variation_range = range(-20, 25, 5)  # -20% ~ +20%

# 4. 시뮬레이션 수행
results = []
for delta in variation_range:
    modified_sample = base_sample.copy()
    modified_sample[target_variable] *= (1 + delta / 100)
    pred = model.predict(modified_sample)[0]
    results.append({"변화율(%)": delta, "예측 NOx": pred})

# 5. 시각화
import matplotlib.pyplot as plt

df_result = pd.DataFrame(results)
plt.plot(df_result["변화율(%)"], df_result["예측 NOx"], marker='o')
plt.axhline(y=50, color='r', linestyle='--', label='규제기준 50ppm')
plt.title("유량 변화에 따른 NOx 예측 반응 시뮬레이션")
plt.xlabel("유량 변화율 (%)")
plt.ylabel("예측 NOx")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
