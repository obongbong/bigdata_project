import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# — 설정값
file_path     = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
SEQ_LEN       = 14
TEST_RATIO    = 0.2
RF_ESTIMATORS = 100
RF_MAX_DEPTH  = None
RANDOM_STATE  = 42

# — 원본 데이터 로드
df_orig = pd.read_excel(file_path)
df_orig["일자"] = pd.to_datetime(df_orig["일자"].astype(str), format="%Y%m%d")

# — 시퀀스 생성 함수
def create_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len, 0])
    return np.array(X), np.array(y)

# — 역변환 함수
def inverse_transform(y_scaled, scaler, n_feats):
    pad = np.zeros((len(y_scaled), n_feats))
    pad[:, 0] = y_scaled
    return np.expm1(scaler.inverse_transform(pad)[:, 0])

# — 이상치 임계값 리스트
thresholds = [30, 40, 50, 60, 70, 80, 90, 100]

results = []
for th in thresholds:
    # 1) 이상치 제거 기준 적용
    df = df_orig[df_orig["NOX"] <= th].reset_index(drop=True)
    
    # 2) 기본 전처리
    features = ["NOX", "SOX", "먼지", "유량", "산소"]
    df = df[["일자"] + features].dropna()
    df["NOX_ema"] = df["NOX"].ewm(span=5).mean()
    df["NOX_log"] = np.log1p(df["NOX_ema"])
    
    use_feats = ["NOX_log", "SOX", "먼지", "유량", "산소"]
    if len(df) < SEQ_LEN + 1:
        continue  # 데이터 부족 시 건너뜀
    
    # 3) 스케일링 및 시퀀스 생성
    scaler = RobustScaler().fit(df[use_feats])
    scaled = scaler.transform(df[use_feats])
    X, y   = create_sequences(scaled, SEQ_LEN)
    
    # 4) 학습/테스트 분할
    split    = int(len(X) * (1 - TEST_RATIO))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    if len(y_te) == 0:
        continue
    
    # 5) 플랫(flatten)
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)
    
    # 6) Random Forest 학습 & 예측
    rf = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_tr_flat, y_tr)
    y_pred = rf.predict(X_te_flat)
    
    # 7) 역변환 및 성능 계산
    y_te_inv   = inverse_transform(y_te, scaler, len(use_feats))
    y_pred_inv = inverse_transform(y_pred, scaler, len(use_feats))
    mae  = mean_absolute_error(y_te_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_te_inv, y_pred_inv))
    r2   = r2_score(y_te_inv, y_pred_inv)
    
    results.append({
        'Threshold': th,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })

# — 결과 DataFrame
df_results = pd.DataFrame(results)
print("\n📊 [Threshold별 Random Forest 성능]")
print(df_results.to_string(index=False))

# — 시각화
plt.figure(figsize=(8,4))
plt.plot(df_results['Threshold'], df_results['MAE'],   '-o', label='MAE')
plt.plot(df_results['Threshold'], df_results['RMSE'],  '-s', label='RMSE')
plt.plot(df_results['Threshold'], df_results['R²'],    '-^', label='R²')
plt.xlabel('NOx Threshold')
plt.title('Threshold별 Random Forest 성능')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
