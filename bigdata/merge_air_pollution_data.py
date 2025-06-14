import pandas as pd
import re

# 1. 파일 불러오기
monthly_df = pd.read_excel("data/한국남동발전_대기오염물질배출농도.xls")
daily_df = pd.read_excel("data/한국남동발전_대기오염물질배출농도(일평균).xls")

# 2. 날짜 처리
monthly_df["월"] = monthly_df["일자"].astype(str)  # 월은 문자열 처리
daily_df["일자"] = pd.to_datetime(daily_df["일자"], format="%Y%m%d")
daily_df["월"] = daily_df["일자"].dt.strftime("%Y%m")

# 3. 날짜 필터링
daily_df = daily_df[(daily_df["일자"] >= "2023-01-01") & (daily_df["일자"] <= "2024-12-31")]
monthly_df = monthly_df[(monthly_df["월"] >= "202301") & (monthly_df["월"] <= "202412")]

# 4. 일평균 데이터를 월 단위로 집계
daily_grouped = daily_df.groupby(["사업소", "호기", "월"]).agg({
    "NOX": "mean",
    "SOX": "mean",
    "먼지": "mean",
    "산소": "mean",
    "유량": "mean",
    "온도": "mean"
}).reset_index()
daily_grouped = daily_grouped.rename(columns={"NOX": "NOX_일평균", "SOX": "SOX_일평균"})

# 5. 호기 문자열 → 리스트로 분해 (삼천포 특수 호기 포함)
def expand_hogi(hogi_str):
    special_mapping = {
        "#1~4호기": ["3A호기", "3B호기", "4A호기", "4B호기"],
        "#5~6호기": ["5A호기", "5B호기", "6A호기", "6B호기"]
    }
    if hogi_str in special_mapping:
        return special_mapping[hogi_str]

    match = re.match(r"#(\d+)~(\d+)호기", hogi_str)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return [f"{i}호기" for i in range(start, end + 1)]

    match = re.match(r"#?([0-9A-Za-z]+호기)", hogi_str)
    if match:
        return [match.group(1)]

    return [hogi_str]

# 🔁 호기 리스트 분해
monthly_df["호기_리스트"] = monthly_df["호기"].apply(expand_hogi)

# ✅ explode 후 새 DataFrame 생성
monthly_df_exploded = monthly_df.explode("호기_리스트")
monthly_df_exploded["호기"] = monthly_df_exploded["호기_리스트"]
monthly_df_exploded.drop(columns=["호기_리스트"], inplace=True)

# ✅ 사업소 이름 정규화
def normalize_station(name):
    return name.replace("발전본부", "").replace("화력본부", "").replace("복합발전처", "").replace("본부", "").replace("처", "").strip()

monthly_df_exploded["사업소"] = monthly_df_exploded["사업소"].apply(normalize_station)
daily_grouped["사업소"] = daily_grouped["사업소"].apply(normalize_station)

# ✅ 삼천포 특수 호기 반영
monthly_df_exploded.loc[monthly_df_exploded["호기"].str.contains("A호기|B호기"), "사업소"] = "삼천포"
daily_grouped.loc[daily_grouped["호기"].str.contains("A호기|B호기"), "사업소"] = "삼천포"

# 6. 병합
merged = pd.merge(
    monthly_df_exploded,
    daily_grouped,
    how="inner",
    on=["사업소", "호기", "월"]
)

# 7. 결과 확인 및 저장
print("✅ 병합 결과 (상위 5행):")
print(merged.head())
print("📍 monthly_df_exploded 사업소:", sorted(monthly_df_exploded["사업소"].unique()))
print("📍 daily_grouped 사업소:", sorted(daily_grouped["사업소"].unique()))
print("📌 [삼천포] monthly_df_exploded 중 삼천포 행 수:", monthly_df_exploded[monthly_df_exploded["사업소"] == "삼천포"].shape[0])
print("📌 [삼천포] daily_grouped 중 삼천포 행 수:", daily_grouped[daily_grouped["사업소"] == "삼천포"].shape[0])

merged.to_csv("data/병합된_NOX_데이터셋.csv", index=False)
