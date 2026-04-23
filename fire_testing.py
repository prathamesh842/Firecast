# find_low_fire_states.py
import pandas as pd

df = pd.read_csv("data/all_predictions.csv")

# State bounding boxes
states = {
    'Washington':   (45.5, 49.0, -124.8, -116.9),
    'Oregon':       (42.0, 46.3, -124.6, -116.5),
    'California':   (32.5, 42.0, -124.5, -114.1),
    'Nevada':       (35.0, 42.0, -120.0, -114.0),
    'Idaho':        (42.0, 49.0, -117.2, -111.0),
    'Montana':      (44.4, 49.0, -116.1, -104.0),
    'Wyoming':      (41.0, 45.1, -111.1, -104.0),
    'Colorado':     (37.0, 41.0, -109.1, -102.0),
    'Utah':         (37.0, 42.0, -114.1, -109.0),
    'Arizona':      (31.3, 37.0, -114.8, -109.0),
    'New Mexico':   (31.3, 37.0, -109.1, -103.0),
    'Texas':        (25.8, 36.5, -106.7, -93.5),
    'Oklahoma':     (33.6, 37.0, -103.0, -94.4),
    'Kansas':       (37.0, 40.0, -102.1, -94.6),
    'Nebraska':     (40.0, 43.0, -104.1, -95.3),
    'South Dakota': (42.5, 45.9, -104.1, -96.4),
    'North Dakota': (45.9, 49.0, -104.1, -96.6),
    'Minnesota':    (43.5, 49.0, -97.2,  -89.5),
    'Iowa':         (40.4, 43.5, -96.6,  -90.1),
    'Missouri':     (36.0, 40.6, -95.8,  -89.1),
    'Illinois':     (37.0, 42.5, -91.5,  -87.5),
    'Wisconsin':    (42.5, 47.1, -92.9,  -86.8),
    'Michigan':     (41.7, 48.3, -90.4,  -82.4),
    'Indiana':      (37.8, 41.8, -88.1,  -84.8),
    'Ohio':         (38.4, 42.3, -84.8,  -80.5),
    'Kentucky':     (36.5, 39.1, -89.6,  -81.9),
    'Tennessee':    (35.0, 36.7, -90.3,  -81.6),
    'Mississippi':  (30.2, 35.0, -91.7,  -88.1),
    'Alabama':      (30.2, 35.0, -88.5,  -84.9),
    'Georgia':      (30.4, 35.0, -85.6,  -80.8),
    'Florida':      (24.5, 31.0, -87.6,  -80.0),
    'South Carolina':(32.0, 35.2, -83.4, -78.5),
    'North Carolina':(33.8, 36.6, -84.3, -75.5),
    'Virginia':     (36.5, 39.5, -83.7,  -75.2),
    'West Virginia':(37.2, 40.6, -82.6,  -77.7),
    'Maryland':     (37.9, 39.7, -79.5,  -74.9),
    'Pennsylvania': (39.7, 42.3, -80.5,  -74.7),
    'New York':     (40.5, 45.0, -79.8,  -71.9),
    'Vermont':      (42.7, 45.0, -73.4,  -71.5),
    'Maine':        (43.1, 47.5, -71.1,  -67.0),
    'New Hampshire':(42.7, 45.3, -72.6,  -70.7),
    'Massachusetts':(41.2, 42.9, -73.5,  -69.9),
    'Connecticut':  (41.0, 42.1, -73.7,  -71.8),
    'Rhode Island': (41.1, 42.0, -71.9,  -71.1),
    'New Jersey':   (38.9, 41.4, -75.6,  -73.9),
    'Delaware':     (38.4, 39.8, -75.8,  -75.0),
}

results = []

for state, (lat_min, lat_max, lng_min, lng_max) in states.items():
    state_df = df[
        (df['latitude'].between(lat_min, lat_max)) &
        (df['longitude'].between(lng_min, lng_max))
    ]

    if len(state_df) < 100:
        continue

    true_fire_rate = state_df['true_fire'].mean() * 100
    avg_prob       = state_df['fire_prob'].mean() * 100
    total_fires    = state_df['true_fire'].sum()
    total_samples  = len(state_df)

    results.append({
        'state':          state,
        'true_fire_rate': round(true_fire_rate, 3),
        'avg_prob':       round(avg_prob, 2),
        'total_fires':    int(total_fires),
        'total_samples':  total_samples
    })

results_df = pd.DataFrame(results).sort_values(
    'true_fire_rate'
)

print("=" * 65)
print("STATES BY FIRE RATE (LOWEST FIRST)")
print("=" * 65)
print(f"{'State':<20} {'Fire Rate':>10} {'Avg Prob':>10} {'True Fires':>12}")
print("-" * 65)

for _, row in results_df.iterrows():
    print(
        f"{row['state']:<20} "
        f"{row['true_fire_rate']:>9.3f}% "
        f"{row['avg_prob']:>9.2f}% "
        f"{row['total_fires']:>12,}"
    )

print("=" * 65)
print(f"\nLOWEST FIRE RATE:  {results_df.iloc[0]['state']} "
      f"({results_df.iloc[0]['true_fire_rate']}%)")
print(f"HIGHEST FIRE RATE: {results_df.iloc[-1]['state']} "
      f"({results_df.iloc[-1]['true_fire_rate']}%)")