import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# — 설정값
file_path   = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
SEQ_LEN     = 14
TEST_RATIO  = 0.2
RF_EST      = 100
RF_DEPTH    = None
XGB_EST     = 100
XGB_DEPTH   = 5
THRESHOLD   = 70   # 이상치 제거 기준
N_FUTURE    = 14   # 예측할 미래 일수

# — 1. 데이터 로딩 및 이상치 제거
df = pd.read_excel(file_path)
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")
df = df[df["NOX"] <= THRESHOLD].reset_index(drop=True)

# — 2. 전처리: EMA + 로그 변환
df["NOX_ema"] = df["NOX"].ewm(span=5).mean()
df["NOX_log"] = np.log1p(df["NOX_ema"])

# — 3. 스케일링
features = ["NOX_log", "SOX", "먼지", "유량", "산소"]
scaler   = RobustScaler().fit(df[features])
scaled   = scaler.transform(df[features])

# — 4. 시퀀스(flatten) 생성
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN].flatten())
    y.append(scaled[i+SEQ_LEN, 0])
X = np.array(X)
y = np.array(y)

# — 5. 학습/테스트 분할
split    = int(len(X) * (1 - TEST_RATIO))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
dates_te   = df["일자"].iloc[SEQ_LEN + split : SEQ_LEN + split + len(y_te)]

# — 6. 모델 정의 및 학습
rf   = RandomForestRegressor(n_estimators=RF_EST, max_depth=RF_DEPTH,
                             random_state=42, n_jobs=-1)
xgbm = XGBRegressor(n_estimators=XGB_EST, max_depth=XGB_DEPTH,
                    objective='reg:squarederror', random_state=42)
voting = VotingRegressor([("rf", rf), ("xgb", xgbm)])
voting.fit(X_tr, y_tr)

# — 7. 14일 미래 예측 (rolling forecast)
last_seq = scaled[-SEQ_LEN:].copy()
future_scaled = []
current = last_seq.copy()

for _ in range(N_FUTURE):
    inp = current.flatten().reshape(1, -1)
    p = voting.predict(inp)[0]
    future_scaled.append(p)
    nxt = np.zeros(len(features))
    nxt[0] = p
    current = np.vstack([current[1:], nxt])

# — 8. 역변환
pad_f = np.zeros((N_FUTURE, len(features)))
pad_f[:,0] = future_scaled
future_nox = np.expm1(scaler.inverse_transform(pad_f)[:,0])

# — 9. 미래 날짜 생성: 2025-07-01 ~ 2025-07-14
future_dates = pd.date_range(start="2025-07-01", periods=N_FUTURE, freq="D")

# — 10. 미래 14일 예측만 시각화
plt.figure(figsize=(10,4))
plt.plot(future_dates, future_nox, '--o', label="2025-07-01~14 예측")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.title("Voting Regressor: 2025년 7월 1일~14일 미래 예측")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# — 11. 결과 DataFrame
df_future = pd.DataFrame({
    "Date": future_dates,
    "예측 NOx": future_nox
})
print("\n📋 [2025-07-01 ~ 07-14 미래 14일치 예측]")
print(df_future)
