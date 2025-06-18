import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# — 설정값
daily_path   = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
monthly_path = "data/한국남동발전_대기오염물질배출농도.xls"
SEQ_LEN      = 14
TEST_RATIO   = 0.2
RF_EST       = 100
RF_DEPTH     = None
XGB_EST      = 100
XGB_DEPTH    = 5
THRESHOLD    = 70

# 1) 두 파일 읽기
df_daily   = pd.read_excel(daily_path)
df_monthly = pd.read_excel(monthly_path)

# 2) 날짜 처리
df_daily["일자"]   = pd.to_datetime(df_daily["일자"].astype(str), format="%Y%m%d")
df_monthly["연월"] = pd.to_datetime(df_monthly["일자"].astype(str),
                                  format="%Y%m").dt.to_period("M")

# 3) 월별 수치형 컬럼에 '_m' suffix
num_cols = df_monthly.select_dtypes(include=[np.number]).columns.tolist()
monthly_feats = {c: f"{c}_m" for c in num_cols}
df_monthly = df_monthly[num_cols + ["연월"]].rename(columns=monthly_feats)

# 4) 병합용 '년월' 키
df_daily["년월"] = df_daily["일자"].dt.to_period("M")

# 5) 일별·월별 병합
df = pd.merge(df_daily, df_monthly,
              left_on="년월", right_on="연월",
              how="left").drop(columns=["년월", "연월"])

# 6) 결측·이상치 처리
required = ["NOX", "SOX", "먼지", "유량", "산소"]
df = df.dropna(subset=required)
df = df[df["NOX"] <= THRESHOLD].reset_index(drop=True)

# 7) NOx EMA
df["NOX_ema"] = df["NOX"].ewm(span=5).mean()

# 8) 스케일링
features = ["NOX_ema", "SOX", "먼지", "유량", "산소"] + list(monthly_feats.values())
scaler   = RobustScaler().fit(df[features])
scaled   = scaler.transform(df[features])

# 9) 시퀀스 생성 및 날짜 매핑
X, y, dates = [], [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN].flatten())
    y.append(scaled[i+SEQ_LEN, 0])               # NOX_ema 스케일된 값
    dates.append(df["일자"].iloc[i+SEQ_LEN])     # 대응 날짜
X, y, dates = map(np.array, (X, y, dates))

# 10) Train/Test 분할
split     = int(len(X) * (1 - TEST_RATIO))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
dates_te   = dates[split:]

# 11) 모델 정의 및 학습
rf   = RandomForestRegressor(n_estimators=RF_EST,
                             max_depth=RF_DEPTH,
                             random_state=42,
                             n_jobs=-1)
xgbm = XGBRegressor(n_estimators=XGB_EST,
                    max_depth=XGB_DEPTH,
                    objective='reg:squarederror',
                    random_state=42)
voting = VotingRegressor([("rf", rf), ("xgb", xgbm)])
voting.fit(X_tr, y_tr)

# 12) 예측 및 역변환 함수
def inv_nox(arr_scaled):
    pad = np.zeros((len(arr_scaled), len(features)))
    pad[:, 0] = arr_scaled
    return scaler.inverse_transform(pad)[:, 0]

y_pred_scaled = voting.predict(X_te)
y_true = inv_nox(y_te)
y_pred = inv_nox(y_pred_scaled)

# 13) 성능 평가
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
print("📊 [Voting Regressor 성능]")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# 14) 시계열 플롯
plt.figure(figsize=(10,4))
plt.plot(dates_te, y_true, label="실제 NOx_EMA")
plt.plot(dates_te, y_pred, label="예측 NOx_EMA")
plt.xlabel("Date")
plt.ylabel("NOx (EMA)")
plt.title("Voting Regressor 과거 예측 결과")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 15) 샘플 결과 출력
df_res = pd.DataFrame({
    "Date": dates_te,
    "실제 NOx_EMA": y_true,
    "예측 NOx_EMA": y_pred
})
print("\n📋 [실제·예측값 샘플]")
print(df_res.head())
