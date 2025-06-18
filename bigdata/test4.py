import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# — 설정값
file_path   = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
SEQ_LEN     = 14
TEST_RATIO  = 0.2
RF_ESTIMATORS = 100
RF_MAX_DEPTH  = None   # None이면 나무를 완전히 성장시킵니다.
RANDOM_STATE  = 42

# — 1. 데이터 로딩 및 전처리
df = pd.read_excel(file_path)
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")

features = ["NOX", "SOX", "먼지", "유량", "산소"]
df = df[["일자"] + features].dropna()
df["NOX_ema"] = df["NOX"].ewm(span=5).mean()
df["NOX_log"] = np.log1p(df["NOX_ema"])

use_feats = ["NOX_log", "SOX", "먼지", "유량", "산소"]
scaler    = RobustScaler().fit(df[use_feats])
scaled    = scaler.transform(df[use_feats])

# — 2. 시퀀스 생성 함수
def create_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

# — 3. 학습/테스트 분할
split     = int(len(X) * (1 - TEST_RATIO))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
dates_te   = df["일자"].iloc[len(df) - len(y_te):]

# — 4. flatten
X_tr_flat = X_tr.reshape(len(X_tr), -1)
X_te_flat = X_te.reshape(len(X_te), -1)

# — 5. Random Forest 학습
rf = RandomForestRegressor(
    n_estimators=RF_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_tr_flat, y_tr)

# — 6. 예측 및 역변환 함수
def inverse_transform(y_scaled):
    pad = np.zeros((len(y_scaled), len(use_feats)))
    pad[:,0] = y_scaled
    return np.expm1(scaler.inverse_transform(pad)[:,0])

y_pred     = rf.predict(X_te_flat)
y_te_inv   = inverse_transform(y_te)
y_pred_inv = inverse_transform(y_pred)

# — 7. 성능 평가
mae  = mean_absolute_error(y_te_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_te_inv, y_pred_inv))
r2   = r2_score(y_te_inv, y_pred_inv)

print("📊 [Random Forest Regressor 성능]")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# — 8. 시계열 플롯
plt.figure(figsize=(10,4))
plt.plot(dates_te, y_te_inv,   label="실제값")
plt.plot(dates_te, y_pred_inv, label="RF 예측값")
plt.xlabel("Date")
plt.ylabel("NOx")
plt.title("Random Forest 예측 결과 (NOx)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# — 9. 결과 DataFrame 샘플
df_res = pd.DataFrame({
    'Date': dates_te,
    '실제값': y_te_inv,
    'RF 예측값': y_pred_inv
})
print("\n📋 [실제·예측값 샘플]")
print(df_res.head())
