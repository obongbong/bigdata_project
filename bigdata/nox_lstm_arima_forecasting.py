import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. 데이터 로딩
file_path = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
df = pd.read_excel(file_path)
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")

# 2. 일자별 NOx 평균 시계열 구성
daily_nox = df.groupby("일자")["NOX"].mean().reset_index()
daily_nox["NOX_log"] = np.log1p(daily_nox["NOX"])

# 3. 정규화
scaler = MinMaxScaler()
scaled_nox = scaler.fit_transform(daily_nox[["NOX_log"]])

# 4. 시퀀스 생성 함수 (Direct 방식)
SEQ_LEN = 14
FUTURE_DAYS = 30

def create_multistep_sequences(data, seq_len, future_len):
    X, y = [], []
    for i in range(len(data) - seq_len - future_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + future_len].flatten())
    return np.array(X), np.array(y)

X, y = create_multistep_sequences(scaled_nox, SEQ_LEN, FUTURE_DAYS)

# 5. Train/Test 분할
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. LSTM 모델 구성
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(SEQ_LEN, 1)))
model.add(Dense(FUTURE_DAYS))
model.compile(optimizer='adam', loss='mse')

# 7. 학습
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

# 8. 테스트셋 전체 예측
y_pred_scaled = model.predict(X_test)
y_pred_log = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, FUTURE_DAYS)
y_test_log = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, FUTURE_DAYS)

y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test_log)

# 9. 평가 지표 (전체 30일 * 샘플 수 기준 평균)
mae = mean_absolute_error(y_test_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
r2 = r2_score(y_test_real.flatten(), y_pred.flatten())

print("📊 Direct Multi-step 예측 평가")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# 10. 마지막 입력 기준으로 미래 예측
last_input = X_test[-1].reshape(1, SEQ_LEN, 1)
future_scaled = model.predict(last_input).flatten()
future_log = scaler.inverse_transform(future_scaled.reshape(-1, 1)).flatten()
future_nox = np.expm1(future_log)

# 11. 시각화
last_date = daily_nox["일자"].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)
past_dates = daily_nox["일자"].iloc[-30:]
past_values = daily_nox["NOX"].iloc[-30:]

plt.figure(figsize=(10, 4))
plt.plot(past_dates, past_values, label="Actual NOx (Last 30 days)", color='blue')
plt.plot(future_dates, future_nox, label="Predicted NOx (Next 30 days)", color='orange', marker='o', linestyle='--')
plt.title("Direct Multi-step Forecasting (NOx)")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
