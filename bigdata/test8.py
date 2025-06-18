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

# — 1. 데이터 로딩 및 이상치 제거
df = pd.read_excel(file_path)
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")
df = df[df["NOX"] <= THRESHOLD].reset_index(drop=True)

# — 2. 전처리: EMA만 적용
df["NOX_ema"] = df["NOX"].ewm(span=5).mean()

# — 3. 스케일링
features = ["NOX_ema", "SOX", "먼지", "유량", "산소"]
scaler   = RobustScaler().fit(df[features])
scaled   = scaler.transform(df[features])

# — 4. 시퀀스(flatten) 생성
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN].flatten())
    y.append(scaled[i+SEQ_LEN, 0])  # NOX_ema 스케일된 값
X = np.array(X)
y = np.array(y)

# — 5. 학습/테스트 분할
split     = int(len(X) * (1 - TEST_RATIO))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
dates_te   = df["일자"].iloc[SEQ_LEN + split : SEQ_LEN + split + len(y_te)]

# — 6. 모델 정의 및 학습
rf   = RandomForestRegressor(
    n_estimators=RF_EST,
    max_depth=RF_DEPTH,
    random_state=42,
    n_jobs=-1
)
xgbm = XGBRegressor(
    n_estimators=XGB_EST,
    max_depth=XGB_DEPTH,
    objective='reg:squarederror',
    random_state=42
)
voting = VotingRegressor([("rf", rf), ("xgb", xgbm)])
voting.fit(X_tr, y_tr)

# — 7. 과거 테스트 예측
y_pred_scaled = voting.predict(X_te)
pad = np.zeros((len(y_te), len(features)))
inv_true = scaler.inverse_transform(np.hstack([y_te.reshape(-1,1), pad[:,1:]]))
inv_pred = scaler.inverse_transform(np.hstack([y_pred_scaled.reshape(-1,1), pad[:,1:]]))
y_true = inv_true[:,0]
y_pred = inv_pred[:,0]

# — 8. 성능 평가
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
print("📊 [Voting Regressor (RF + XGB) 성능]")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# — 9. 시계열 플롯
plt.figure(figsize=(10,4))
plt.plot(dates_te, y_true, label="실제 NOx")
plt.plot(dates_te, y_pred, label="앙상블 예측값")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.title("Voting Regressor 예측 결과 (NOx)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# — 10. 샘플 결과 DataFrame
df_res = pd.DataFrame({
    "Date": dates_te,
    "실제값": y_true,
    "앙상블 예측값": y_pred
})
print("\n📋 [실제·예측값 샘플]")
print(df_res.head())
