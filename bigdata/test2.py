import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
file_path = "data/전국_대기오염물질_배출량.xlsx"
df = pd.read_excel(file_path)

# 2. 첫 번째 열 이름을 '오염물질'로 바꾸고 필요없는 열 제거
df.rename(columns={df.columns[0]: "오염물질"}, inplace=True)
df = df.dropna(subset=["오염물질"])  # 오염물질 이름이 없는 행 제거

# 3. 오염물질별 연도별 배출량 정리 (Tidy format으로 변환)
df_melted = df.melt(id_vars="오염물질", var_name="연도", value_name="배출량")

# 4. 피벗 테이블로 재구성 (행: 연도, 열: 오염물질)
df_pivot = df_melted.pivot_table(index="연도", columns="오염물질", values="배출량")

# 5. 수치형 변환 및 결측치 제거
df_pivot = df_pivot.apply(pd.to_numeric, errors="coerce").dropna()

# 6. 상관계수 계산 및 히트맵 시각화
corr = df_pivot.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("오염물질 간 상관관계")
plt.tight_layout()
plt.show()
