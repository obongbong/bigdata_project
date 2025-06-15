import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

# 📌 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 모델 및 시퀀스 불러오기
model = load_model("models/nox_lstm_model.h5", compile=False)
scaler = joblib.load("models/nox_scaler.pkl")
last_seq = np.load("models/last_sequence.npy").reshape(1, -1, 1)  # (1, 14, 1)

# 2. 미래 예측 (30일)
FUTURE_DAYS = 30
future_preds = []

for _ in range(FUTURE_DAYS):
    pred = model.predict(last_seq, verbose=0)  # shape: (1, 1)
    future_preds.append(pred[0, 0])
    last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

# 3. 역변환
future_preds_nox = np.expm1(scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)))

# 4. 날짜 생성 및 시각화
last_date = pd.to_datetime("2024-12-31")
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)

plt.figure(figsize=(12, 5))
plt.plot(future_dates, future_preds_nox, marker='o', label="Predicted NOx (2025-01)")
plt.title("2025년 1월 NOx 예측 (LSTM 기반, 30일)")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. 저장
pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_NOx": future_preds_nox.flatten()
})
pred_df.to_csv("results/future_nox_2025_january.csv", index=False)
