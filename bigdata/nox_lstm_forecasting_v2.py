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

# 2. 일자별 NOx 평균 시계열 구성
daily_nox = df.groupby("일자")["NOX"].mean().reset_index()

# ✅ 이상치 제거
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)].copy()

daily_nox = remove_outliers_iqr(daily_nox, "NOX").reset_index(drop=True)
daily_nox["NOX_log"] = np.log1p(daily_nox["NOX"])

# 3. 정규화
scaler = MinMaxScaler()
scaled_nox = scaler.fit_transform(daily_nox[["NOX_log"]])

# 4. 시퀀스 생성
SEQ_LEN = 14
def create_sequences(data, seq_len=14):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_nox, SEQ_LEN)

# 5. 학습 데이터 분할
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. 모델 정의
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(SEQ_LEN, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 7. 학습
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# 8. 테스트셋 예측
y_pred = model.predict(X_test)
y_test_inv = np.expm1(scaler.inverse_transform(y_test.reshape(-1, 1)))
y_pred_inv = np.expm1(scaler.inverse_transform(y_pred))

# 9. 평가
print("📊 [LSTM NOx 예측 결과]")
print("MAE :", mean_absolute_error(y_test_inv, y_pred_inv))
print("RMSE:", np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
print("R²  :", r2_score(y_test_inv, y_pred_inv))

# 10. 시각화
test_dates = daily_nox["일자"].iloc[train_size + SEQ_LEN:]
plt.figure(figsize=(10, 4))
plt.plot(test_dates, y_test_inv, label="실제값")
plt.plot(test_dates, y_pred_inv, label="예측값")
plt.title("NOx 예측 결과 (LSTM + 이상치 제거)")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
