import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 한글 폰트 설정 (Windows용: 'Malgun Gothic')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기
df = pd.read_csv("data/병합된_NOX_데이터셋.csv")

# 2. 범주형 인코딩
df = pd.get_dummies(df, columns=["사업소", "호기"])

# 3. 특성과 타겟 설정
X = df[["NOX_일평균", "SOX_일평균", "먼지", "산소", "유량", "온도"] +
       [col for col in df.columns if col.startswith("사업소_") or col.startswith("호기_")]]
y = df["질소산화물(ppm)평균"]

# 4. 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. GridSearchCV 기반 최적 모델 탐색
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\n📌 Best Params:", grid_search.best_params_)

# 6. 예측 및 평가
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 RMSE: {rmse:.3f}")
print(f"📈 R²: {r2:.3f}")

# 7. 변수 중요도 시각화
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)
print("\n📌 NOx에 영향 큰 상위 5개 인자:")
print(importance_df.head(5))

# 8. 예측값 vs 실제값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual NOx")
plt.ylabel("Predicted NOx")
plt.title("Actual vs Predicted NOx")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. 규제 초과 예측 분석
regulation_limit = 50
exceed_pred = y_pred > regulation_limit
exceed_rate = np.mean(exceed_pred)
print(f"\n🚨 규제 기준 {regulation_limit}ppm 초과 예측 비율: {exceed_rate*100:.2f}%")

exceed_cases = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "초과여부": exceed_pred
})
print("\n⚠️ 예측 기준 초과 샘플:")
print(exceed_cases[exceed_cases["초과여부"]].head())

# 10. GridSearch 결과 저장 및 시각화
cv_results = pd.DataFrame(grid_search.cv_results_)
result_summary = cv_results[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
result_summary = result_summary.sort_values(by="mean_test_score", ascending=False)
result_summary.to_csv("gridsearch_nox_results.csv", index=False)

print("\n📊 GridSearch 상위 10개 조합:")
print(result_summary.head(10))

cv_results["param_summary"] = cv_results["params"].astype(str)
plt.figure(figsize=(12, 6))
sns.barplot(x="mean_test_score", y="param_summary", data=cv_results.sort_values(by="mean_test_score", ascending=False))
plt.title("하이퍼파라미터 조합별 교차검증 평균 R²")
plt.xlabel("Mean Test R² Score")
plt.ylabel("Parameter Combination")
plt.tight_layout()
plt.show()

# 11. 최적 모델 및 feature 목록 저장
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_nox_model.pkl")

# 🔐 학습에 사용된 feature 이름 저장
with open("models/feature_names.txt", "w", encoding="utf-8") as f:
    for col in X.columns:
        f.write(col + "\n")

print("\n💾 모델 및 feature 목록이 models 폴더에 저장되었습니다.")
