import pandas as pd

df  = pd.read_csv("data/all_predictions.csv")
df['date'] = pd.to_datetime(df['date'])

seattle = df[
    (df['latitude'].between(47.5, 48.2)) &
    (df['longitude'].between(-122.8, -122.0))
].copy()

# ── Monthly breakdown ──────────────────────────────────────────
print("=" * 55)
print("MONTHLY FIRE BREAKDOWN — SEATTLE")
print("=" * 55)
seattle['month'] = seattle['date'].dt.month
seattle['year']  = seattle['date'].dt.year

monthly = seattle.groupby(['year','month']).agg(
    true_fires    = ('true_fire',  'sum'),
    avg_prob      = ('fire_prob',  'mean'),
    max_prob      = ('fire_prob',  'max'),
    total_days    = ('true_fire',  'count')
).reset_index()

# Show only months with actual fires
real_fires = monthly[monthly['true_fires'] > 0]
print(real_fires.to_string())

# ── Overall stats ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("OVERALL STATS")
print("=" * 55)
print(f"Date range:      {seattle['date'].min().date()} → {seattle['date'].max().date()}")
print(f"Total samples:   {len(seattle):,}")
print(f"Real fires:      {seattle['true_fire'].sum():,}")
print(f"Predicted fires: {(seattle['fire_prob']>0.9).sum():,}")
print(f"True fire rate:  {seattle['true_fire'].mean()*100:.2f}%")

# ── Confusion check ────────────────────────────────────────────
tp = len(seattle[(seattle['fire_prob']>0.9) & (seattle['true_fire']==1)])
fp = len(seattle[(seattle['fire_prob']>0.9) & (seattle['true_fire']==0)])
tn = len(seattle[(seattle['fire_prob']<=0.9) & (seattle['true_fire']==0)])
fn = len(seattle[(seattle['fire_prob']<=0.9) & (seattle['true_fire']==1)])

print(f"\nTrue  Positives: {tp:,}  ← correctly caught fires")
print(f"False Positives: {fp:,}  ← false alarms")
print(f"True  Negatives: {tn:,}  ← correctly safe days")
print(f"False Negatives: {fn:,}  ← missed fires")

precision = tp / (tp + fp) if (tp+fp) > 0 else 0
recall    = tp / (tp + fn) if (tp+fn) > 0 else 0
print(f"\nPrecision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

# ── Where are the real fires? ──────────────────────────────────
print("\n" + "=" * 55)
print("REAL FIRE DATES IN SEATTLE")
print("=" * 55)
real = seattle[seattle['true_fire']==1].sort_values('date')
print(real[['date','latitude','longitude',
            'fire_prob','true_fire']].to_string())
