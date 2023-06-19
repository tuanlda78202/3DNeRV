import pandas as pd

df = pd.read_csv(
    "/home/tuanlda78202/3ai24/wandb_export_2023-06-20T01_00_54.896+07_00.csv"
)
print(df.loc[0:600, "celestial-elevator-27 - psnr"].mean())
