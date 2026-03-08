import pyarrow.parquet as pq

table = pq.read_table("data/auralis_signals_xgb.parquet")
df = table.to_pandas()

print(df.columns)