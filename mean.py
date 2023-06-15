import pandas as pd

df = pd.read_csv(
    "/Users/charles/Downloads/wandb_export_2023-06-14T14_15_56.717+07_00.csv"
)
print(df.loc[0 : 400 * 65, "grateful-flower-31 - psnr"].mean())
