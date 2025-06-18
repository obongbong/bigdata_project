import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 두 데이터셋 불러오기
df_daily = pd.read_excel("data/한국남동발전_대기오염물질배출농도(일평균).xls")
df_hourly = pd.read_excel("data/한국남동발전_대기오염물질배출농도.xls")

# 2. 날짜 처리
df_daily["일자"] = pd.to_datetime(df_daily["일자"].astype(str), format="%Y%m%d")
df_hourly["일자"] = pd.to_datetime(df_hourly["일자"].astype(str), format="%Y%m")

# 날짜 통일 (월 단위)
df_daily["일자"] = df_daily["일자"].dt.to_period("M").dt.to_timestamp()
df_hourly["일자"] = df_hourly["일자"].dt.to_period("M").dt.to_timestamp()

# 3. 병합 (내부 조인)
df_merged = pd.merge(df_daily, df_hourly, on="일자", how="inner")

# 4. 예측에 사용할 변수 정의
features = [
    "NOX",  # 타깃
    "황산화물(ppm)평균", 
    "질소산화물(ppm)평균", 
    "먼지(㎎/S㎥)평균"
]

# 5. 이상치 제거 (NOX 기준)
Q1 = df_merged["NOX"].quantile(0.25)
Q3 = df_merged["NOX"].quantile(0.75)
IQR = Q3 - Q1
df_merged = df_merged[(df_merged["NOX"] >= Q1 - 1.5 * IQR) & (df_merged["NOX"] <= Q3 + 1.5 * IQR)]

# 6. 정규화
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_merged[features])

# 7. 시퀀스 생성
SEQ_LEN = 14
FUTURE_DAYS = 90

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len - FUTURE_DAYS):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+FUTURE_DAYS, 0])  # NOX만 예측
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

# 8. Train/Test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 9. 모델 정의
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(features))))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(FUTURE_DAYS))
model.compile(optimizer='adam', loss='mse')

# 10. 학습
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# 11. 예측
y_pred = model.predict(X_test)

# 12. 역변환 함수 (NOX만)
def inverse_transform_nox(scaled_seq):
    padded = np.zeros((len(scaled_seq), len(features)))
    padded[:, 0] = scaled_seq
    return scaler.inverse_transform(padded)[:, 0]

y_test_inv = np.array([inverse_transform_nox(seq) for seq in y_test])
y_pred_inv = np.array([inverse_transform_nox(seq) for seq in y_pred])

# 13. 평가 (90일 예측 중 마지막 날짜 기준)
print("\n📊 [Multivariate LSTM - NOx 3개월 예측 성능]")
print("MAE  :", mean_absolute_error(y_test_inv[:, -1], y_pred_inv[:, -1]))
print("RMSE :", np.sqrt(mean_squared_error(y_test_inv[:, -1], y_pred_inv[:, -1])))
print("R²   :", r2_score(y_test_inv[:, -1], y_pred_inv[:, -1]))

# 14. 시각화 (마지막 테스트 시퀀스)
last_dates = df_merged["일자"].iloc[-FUTURE_DAYS:]
plt.figure(figsize=(12, 5))
plt.plot(last_dates, y_test_inv[-1], label="실제값 (NOx)")
plt.plot(last_dates, y_pred_inv[-1], label="예측값 (NOx)")
plt.title("📈 Multivariate LSTM 기반 NOx 예측 결과 (3개월)")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()