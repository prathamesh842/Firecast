# test_gridmet_debug.py
import requests

# Test 1: Basic internet
print("Test 1: Basic internet...")
try:
    r = requests.get("https://google.com", timeout=10)
    print(f"  ✅ Internet working: {r.status_code}")
except:
    print("  ❌ No internet!")

# Test 2: GridMET main site
print("\nTest 2: GridMET main site...")
try:
    r = requests.get(
        "https://www.climatologylab.org/gridmet.html",
        timeout=10
    )
    print(f"  ✅ GridMET site: {r.status_code}")
except Exception as e:
    print(f"  ❌ GridMET site blocked: {e}")

# Test 3: GridMET thredds server
print("\nTest 3: GridMET thredds server...")
try:
    r = requests.get(
        "https://thredds.northwestknowledge.net/thredds/catalog.html",
        timeout=15
    )
    print(f"  ✅ Thredds server: {r.status_code}")
except Exception as e:
    print(f"  ❌ Thredds blocked: {e}")

# Test 4: Actual data URL
print("\nTest 4: Actual data URL...")
url = (
    "https://thredds.northwestknowledge.net"
    "/thredds/ncss/grid/MET/tmmx/tmmx_2024.nc"
    "?var=air_temperature"
    "&latitude=37.7749"
    "&longitude=-119.4194"
    "&time_start=2024-01-01"
    "&time_end=2024-01-03"
    "&accept=csv"
)
try:
    r = requests.get(url, timeout=30)
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.text[:200]}")
except Exception as e:
    print(f"  ❌ Data URL failed: {e}")

# Test 5: Alternative GridMET API
print("\nTest 5: Alternative API (climate engine)...")
try:
    r = requests.get(
        "https://api.climatologylab.org/gridmet",
        timeout=10
    )
    print(f"  Status: {r.status_code}")
except Exception as e:
    print(f"  ❌ Alternative API: {e}")