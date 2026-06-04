import pandas as pd

df = pd.read_csv("bk21_spr.csv", nrows=10)
df.to_csv("sample.csv", index=False)
