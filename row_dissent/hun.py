import pandas as pd

df = pd.read_csv("bkr21_spr.csv", nrows=30)
df.to_csv("rsample.csv", index=False)

df = pd.read_csv("bko21_spr.csv", nrows=30)
df.to_csv("osample.csv", index=False)
