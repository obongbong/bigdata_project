# train_nox_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import random
import os

# ✅ 시드 고정 (재현성 확보)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 1. 데이터 불러오기
df = pd.read_excel("data/한국남동발전_대기오염물질배출농도(일평균).xls")
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")
df = df[(df["일자"] >= "2023-01-01") & (df["일자"] <= "2024-12-31")]

# 2. 일자별 NOx 평균
daily_nox = df.groupby("일자")["NOX"].mean().reset_index()
daily_nox["NOX_log"] = np.log1p(daily_nox["NOX"])

# 3. 정규화
scaler = MinMaxScaler()
scaled_nox = scaler.fit_transform(daily_nox[["NOX_log"]])

# 4. 시퀀스 생성 함수
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

# 6. 모델 정의 및 학습
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(SEQ_LEN, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# 7. 모델 저장
model.save("models/nox_lstm_model.h5")

# 8. 스케일러 저장
import joblib
joblib.dump(scaler, "models/nox_scaler.pkl")

# 9. 시퀀스 저장
np.save("models/last_sequence.npy", scaled_nox[-SEQ_LEN:])
