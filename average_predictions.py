import sys
import pandas as pd


print("reading", sys.argv[1])
df = pd.read_csv(sys.argv[1])
num = 1
for path in sys.argv[2:]:
    num += 1
    print("reading", path)
    df2 = pd.read_csv(path)
    df.is_bot += df2.is_bot

df.is_bot = df.is_bot / num

print(num)
df.to_csv("combo4.csv", index=False)
print(df.head())

