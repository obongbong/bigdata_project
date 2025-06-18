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
RF_EST       = 100
RF_DEPTH     = None
XGB_EST      = 100
XGB_DEPTH    = 5
THRESHOLD    = 70  # 이상치 제거 기준

# — 1. 데이터 로딩 및 임계값 기반 이상치 제거
df = pd.read_excel(file_path)
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")
df = df[df["NOX"] <= THRESHOLD].reset_index(drop=True)

# — 2. 전처리: NOx 지수이동평균 → 로그 변환
df["NOX_ema"] = df["NOX"].ewm(span=5).mean()
df["NOX_log"] = np.log1p(df["NOX_ema"])

# — 3. 피처 및 스케일링
features = ["NOX_log", "SOX", "먼지", "유량", "산소"]
scaler  = RobustScaler().fit(df[features])
scaled  = scaler.transform(df[features])

# — 4. 시퀀스 생성 (flatten 형태)
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    seq = scaled[i:i+SEQ_LEN].flatten()
    X.append(seq)
    y.append(scaled[i+SEQ_LEN, 0])  # NOX_log 스케일된 값
X = np.array(X)
y = np.array(y)

# — 5. 학습/테스트 분할
split    = int(len(X) * (1 - TEST_RATIO))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
dates_te   = df["일자"].iloc[SEQ_LEN + split : SEQ_LEN + split + len(y_te)]

# — 6. 개별 모델 정의
rf   = RandomForestRegressor(n_estimators=RF_EST, max_depth=RF_DEPTH,
                             random_state=42, n_jobs=-1)
xgbm = XGBRegressor(n_estimators=XGB_EST, max_depth=XGB_DEPTH,
                    objective='reg:squarederror', random_state=42)

# — 7. Voting 앙상블
voting = VotingRegressor([("rf", rf), ("xgb", xgbm)])
voting.fit(X_tr, y_tr)

# — 8. 예측 및 역변환
y_pred_scaled = voting.predict(X_te)

def inverse_transform(y_scaled):
    pad = np.zeros((len(y_scaled), len(features)))
    pad[:, 0] = y_scaled
    return np.expm1(scaler.inverse_transform(pad)[:, 0])

y_true = inverse_transform(y_te)
y_pred = inverse_transform(y_pred_scaled)

# — 9. 성능 평가
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print("📊 [Voting Regressor (RF + XGB) 성능]")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# — 10. 시계열 플롯
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

# — 11. 샘플 결과 DataFrame
df_res = pd.DataFrame({
    "Date": dates_te,
    "실제값": y_true,
    "앙상블 예측값": y_pred
})
print("\n📋 [실제·예측값 샘플]")
print(df_res.head())


# — 12. 14일치 미래 예측
N_FUTURE = 14
# 1) 마지막 SEQ_LEN일 시퀀스 가져오기
last_seq = scaled[-SEQ_LEN:].copy()

future_preds_scaled = []
current_seq = last_seq.copy()

for _ in range(N_FUTURE):
    # 2) flat → 모델 입력
    inp = current_seq.flatten().reshape(1, -1)
    # 3) 예측 (스케일된 NOX_log)
    pred_scaled = voting.predict(inp)[0]
    future_preds_scaled.append(pred_scaled)
    # 4) 다음 시퀀스 구성: 맨 앞 버리고 예측값 추가
    next_feat = np.zeros(len(features))
    next_feat[0] = pred_scaled
    current_seq = np.vstack([current_seq[1:], next_feat])

# — 13. 역변환
pad = np.zeros((N_FUTURE, len(features)))
pad[:, 0] = future_preds_scaled
future_nox = np.expm1(scaler.inverse_transform(pad)[:, 0])

# — 14. 미래 날짜 생성
last_date = df["일자"].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(N_FUTURE)]

# — 15. 플롯에 덧붙여서 시각화
plt.figure(figsize=(10,4))
plt.plot(dates_te, y_true,      label="실제 NOx")
plt.plot(dates_te, y_pred,      label="앙상블 예측값")
plt.plot(future_dates, future_nox, '--', linewidth=2, label="14일 미래 예측")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.title("Voting Regressor 과거 & 14일 미래 예측")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# — 16. 14일치 예측 결과 확인
df_future = pd.DataFrame({
    "Date": future_dates,
    "예측 NOx": future_nox
})
print("\n📋 [미래 14일치 예측]")
print(df_future)
