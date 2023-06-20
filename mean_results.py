import pandas as pd

df = pd.read_csv("/home/tuanlda78202/3ai24/ready.csv")
print(df.loc[0:600, "infer-ready640 - psnr"].mean())
