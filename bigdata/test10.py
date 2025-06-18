import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
file_path = "data/한국남동발전_대기오염물질배출농도.xls"
df = pd.read_excel(file_path)

# 2. 컬럼 이름 공백 제거
df.columns = df.columns.str.strip()

# 3. 분석에 사용할 열 지정 (실제 컬럼명 사용)
cols = ["황산화물(ppm)평균", "질소산화물(ppm)평균", "먼지(㎎/S㎥)평균"]

# 4. 존재하는 컬럼만 선택
df_selected = df[cols].copy()

# 5. 결측치 제거
df_selected.dropna(inplace=True)

# 6. 상관계수 계산
corr_matrix = df_selected.corr(method='pearson')

# 7. 히트맵 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("📊 대기오염물질 간 상관관계")
plt.tight_layout()
plt.show()
