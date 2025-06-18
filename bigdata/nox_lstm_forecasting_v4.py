import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로딩
file_path = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
df = pd.read_excel(file_path)
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")

# 2. 피처 정의 및 로그/EMA 적용
features = ["NOX", "SOX", "먼지", "유량", "산소"]
df = df[["일자"] + features].dropna()
df["NOX_ema"] = df["NOX"].ewm(span=5).mean()
df["NOX_log"] = np.log1p(df["NOX_ema"])

# 3. 이상치 제거 (모든 피처에 대해 반복 적용)
def remove_outliers_all(df, cols):
    for col in cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df.reset_index(drop=True)

df = remove_outliers_all(df, ["NOX_log", "SOX", "먼지", "유량", "산소"])


# 4. 정규화
use_features = ["NOX_log", "SOX", "먼지", "유량", "산소"]
scaler = RobustScaler()
scaled = scaler.fit_transform(df[use_features])

# 5. 시퀀스 생성
SEQ_LEN = 14
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

# 6. Train/Test 분할
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 7. 모델 구성
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQ_LEN, X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 8. 학습
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)

# 9. 예측 및 평가
y_pred = model.predict(X_test)

# 10. 역변환
def inverse_transform(y_scaled):
    padded = np.zeros((len(y_scaled), len(use_features)))
    padded[:, 0] = y_scaled.flatten()
    return np.expm1(scaler.inverse_transform(padded)[:, 0])

y_test_inv = inverse_transform(y_test)
y_pred_inv = inverse_transform(y_pred)

print("📊 [최종 개선 모델 성능]")
print("MAE :", mean_absolute_error(y_test_inv, y_pred_inv))
print("RMSE:", np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
print("R²  :", r2_score(y_test_inv, y_pred_inv))

# 11. 시각화
test_dates = df["일자"].iloc[len(df) - len(y_test):]
plt.figure(figsize=(10, 4))
plt.plot(test_dates, y_test_inv, label="실제값")
plt.plot(test_dates, y_pred_inv, label="예측값")
plt.title("정확도 개선 모델 예측 결과 (NOx)")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
