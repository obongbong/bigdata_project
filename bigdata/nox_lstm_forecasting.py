import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense


# 📌 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기 및 전처리
file_path = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
df = pd.read_excel(file_path)
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")
# df = df[(df["일자"] >= "2023-01-01") & (df["일자"] <= "2024-12-31")]

# 2. 일자별 NOx 평균 시계열 구성
daily_nox = df.groupby("일자")["NOX"].mean().reset_index()
daily_nox["NOX_log"] = np.log1p(daily_nox["NOX"])  # 로그 변환

# 3. 정규화
scaler = MinMaxScaler()
scaled_nox = scaler.fit_transform(daily_nox[["NOX_log"]])

# 4. 시퀀스 데이터 생성 함수
def create_sequences(data, seq_len=14):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 14
X, y = create_sequences(scaled_nox, SEQ_LEN)

# 5. Train/Test 분할
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. LSTM 모델 정의
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(SEQ_LEN, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 7. 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# 8. 예측 수행 (테스트셋)
y_pred = model.predict(X_test)

# 9. 역변환 (log1p → expm1)
y_test_inv = np.expm1(scaler.inverse_transform(y_test.reshape(-1, 1)))
y_pred_inv = np.expm1(scaler.inverse_transform(y_pred))

# 10. 성능 평가
print("📊 [LSTM NOx 예측 결과]")
print("MAE :", mean_absolute_error(y_test_inv, y_pred_inv))
print("RMSE:", np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
print("R²  :", r2_score(y_test_inv, y_pred_inv))

# 11. 예측 결과 시각화
test_dates = daily_nox["일자"].iloc[train_size + SEQ_LEN:]  # 마지막 날짜부터 테스트셋 길이만큼

plt.figure(figsize=(10, 4))
plt.plot(test_dates, y_test_inv, label="실제값")
plt.plot(test_dates, y_pred_inv, label="예측값")
plt.title("NOx 예측 결과 (LSTM)")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("전체 날짜:", daily_nox["일자"].min(), "~", daily_nox["일자"].max())
print("예측 시작 날짜:", test_dates.iloc[0])
print("예측 종료 날짜:", test_dates.iloc[-1])

# 12. 미래 예측 (2025년 이후 7일치 예측)
# FUTURE_DAYS = 30
# last_seq = scaled_nox[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
# future_preds = []

# for _ in range(FUTURE_DAYS):
#     pred = model.predict(last_seq)  # shape: (1, 1)
#     future_preds.append(pred[0, 0])

#     # pred → shape: (1, 1, 1) 로 reshape
#     pred_reshaped = pred.reshape(1, 1, 1)

#     # 마지막 시퀀스 업데이트: 뒤에 pred 붙이기
#     last_seq = np.append(last_seq[:, 1:, :], pred_reshaped, axis=1)


# # 역변환
# future_preds_nox = np.expm1(scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)))

# # 예측 날짜 생성
# last_date = daily_nox["일자"].iloc[-1]
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)

# # 시각화 (실제값 + 미래예측)
# plt.figure(figsize=(10, 4))
# plt.plot(daily_nox["일자"], daily_nox["NOX"], label="Actual NOx (2023~2024)")
# plt.plot(future_dates, future_preds_nox, label="Predicted Future NOx (2025)", marker='o', linestyle='--')
# plt.title("NOx 실제값 + 1달 미래 예측 (2025)")
# plt.xlabel("Date")
# plt.ylabel("NOx")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# 12. 미래 예측 (30일)
# FUTURE_DAYS = 30
# last_seq = scaled_nox[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
# future_preds = []

# for _ in range(FUTURE_DAYS):
#     pred = model.predict(last_seq, verbose=0)
#     future_preds.append(np.clip(pred[0, 0], 0, 1))  # clip으로 발산 방지
#     last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

# # 역변환
# future_preds_nox = np.expm1(scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)))

# # 예측 날짜 생성
# last_date = daily_nox["일자"].iloc[-1]
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)

# # 비교용 과거 30일치 실제값
# past_dates = daily_nox["일자"].iloc[-30:]
# past_values = daily_nox["NOX"].iloc[-30:]

# # 시각화: 과거 30일 + 미래 30일
# plt.figure(figsize=(10, 4))
# plt.plot(past_dates, past_values, label="Actual NOx (Last 30 days)", color='blue')
# plt.plot(future_dates, future_preds_nox, label="Predicted NOx (Next 30 days)", color='orange', marker='o', linestyle='--')
# plt.title("NOx 실제값 (최근 1달) + 예측값 (미래 1달)")
# plt.xlabel("Date")
# plt.ylabel("NOx")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


FUTURE_DAYS = 30
NUM_WINDOWS = 10
SEQ_LEN = 14

all_preds = []  # shape = (NUM_WINDOWS, FUTURE_DAYS)

# 여러 입력 시퀀스를 sliding window로 생성
for i in range(NUM_WINDOWS):
    start_idx = -(SEQ_LEN + FUTURE_DAYS + i)
    end_idx = -(FUTURE_DAYS + i)
    input_seq = scaled_nox[start_idx:end_idx].reshape(1, SEQ_LEN, 1)

    preds = []
    last_seq = input_seq.copy()
    for _ in range(FUTURE_DAYS):
        pred = model.predict(last_seq, verbose=0)
        preds.append(pred[0, 0])
        last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    all_preds.append(preds)

# 평균화
avg_preds = np.mean(all_preds, axis=0)

# 역변환
future_preds_nox = np.expm1(scaler.inverse_transform(np.array(avg_preds).reshape(-1, 1)))

# 날짜 생성
last_date = daily_nox["일자"].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)

# 과거 30일 데이터
past_dates = daily_nox["일자"].iloc[-30:]
past_values = daily_nox["NOX"].iloc[-30:]

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(past_dates, past_values, label="Actual NOx (Last 30 days)", color='blue')
plt.plot(future_dates, future_preds_nox, label="Predicted NOx (Next 30 days)", color='orange', marker='o', linestyle='--')
plt.title("NOx Sliding Window Averaged 예측 (미래 30일)")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
