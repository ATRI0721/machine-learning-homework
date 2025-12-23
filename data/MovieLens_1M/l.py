import pandas as pd
df = pd.read_csv(r'./dev.csv', sep='\t')   # 根据实际路径调整
print(df.columns)
print(df.head())