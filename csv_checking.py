# Run this to see CLEAN ranges:

import pandas as pd

df = pd.read_csv("Wildfire_Dataset.csv")

# Remove nodata values (32767)
clean = df[df['tmmx'] < 400].copy()

print(f"Total rows:    {len(df):,}")
print(f"Clean rows:    {len(clean):,}")
print(f"Nodata rows:   {len(df)-len(clean):,}")
print()

cols = ['tmmx','tmmn','pr','rmax','rmin','sph','vs',
        'bi','fm100','fm1000','erc','etr','pet','vpd','srad']

for col in cols:
    if col in clean.columns:
        print(f"{col:<8} "
              f"min={clean[col].min():.3f}  "
              f"max={clean[col].max():.3f}  "
              f"mean={clean[col].mean():.3f}")