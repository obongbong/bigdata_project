import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기
file_path = "data/한국남동발전_대기오염물질배출농도(일평균).xls"
df = pd.read_excel(file_path)

# 2. 날짜 변환 및 필터링
df["일자"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d")
df = df[(df["일자"] >= "2023-01-01") & (df["일자"] <= "2024-12-31")]

# 3. 사업소_호기 컬럼 생성
df["사업소_호기"] = df["사업소"].str.strip() + "_" + df["호기"].str.strip()

# ─────────────────────────────────────
# ✅ ① 기술통계 분석: NOx에 대한 요약
print("📊 [기술통계 분석 - NOx]")
print(df["NOX"].describe())
print()

# ─────────────────────────────────────
# ✅ ② 그룹별 분석: 사업소_호기별 NOx 평균, 표준편차
print("📊 [사업소_호기별 NOx 평균 및 표준편차]")
group_stats = df.groupby("사업소_호기")["NOX"].agg(["mean", "std", "max", "min", "count"]).sort_values(by="mean", ascending=False)
print(group_stats)
print()

# ─────────────────────────────────────
# ✅ ③ 상관관계 분석: NOx와 다른 수치형 변수들과의 관계
print("📊 [NOx와 다른 변수의 상관관계]")
numeric_cols = df.select_dtypes(include='number')  # 수치형 변수만 추출
corr = numeric_cols.corr()
nox_corr = corr["NOX"].sort_values(ascending=False)
print(nox_corr)
print()

# Heatmap 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("🔍 Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────
# ✅ ④ 이상치 탐지: NOx > 200
print("📊 [이상치 탐지 - NOx > 200]")
outliers = df[df["NOX"] > 200]
print(outliers[["일자", "사업소", "호기", "NOX", "SOX", "유량", "온도"]])
print()

# 히스토그램 (0~100 범위로 제한)
plt.figure(figsize=(10, 4))
sns.histplot(df[df["NOX"] <= 100]["NOX"], bins=50, kde=True)
plt.title("NOx(질소산화물 농도)의 분포 (0~100 범위)")
plt.xlabel("NOx")
plt.ylabel("Count")
plt.xlim(0, 100)
plt.tight_layout()
plt.show()
