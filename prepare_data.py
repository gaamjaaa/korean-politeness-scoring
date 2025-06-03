import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer

# 1) Manual 라벨링 데이터 (361문장)
manual_path = "./data/manual_labels.csv"
manual_df = pd.read_csv(manual_path)

# 2) TyDiP 데이터 (500문장)
tydip_path = "./data/ko_test.csv"
tydip_df = pd.read_csv(tydip_path)

print("*** Manual dataset columns ***")
print(manual_df.columns.tolist())
print(manual_df.head(3))
print(manual_df.info())  # 데이터 타입, 결측치 확인

print("*** TyDiP dataset columns ***")
print(tydip_df.columns.tolist())
print(tydip_df.head(3))
print(tydip_df.info())


manual_df = pd.read_csv("data/manual_labels.csv")

feature_cols = [
    "feat_ending", "feat_strat_cnt", "feat_command",
    "feat_attack", "feat_power", "feat_distance", "feat_indirect"
]

for col in feature_cols:
    counts = manual_df[col].value_counts().sort_index()
    print(f">>> {col} 분포 (0~3):")
    print(counts)
    print()
