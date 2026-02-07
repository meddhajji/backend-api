
import sys
import os
sys.path.append(os.getcwd())
from db import get_distinct_values

print("Fetching distinct brands...")
brands = get_distinct_values('brand')
print(f"Found {len(brands)} brands.")
print(f"First 10: {brands[:10]}")
print(f"Last 10: {brands[-10:]}")

if len(brands) > 50:
    print("SUCCESS: Retrieved more than 50 brands (likely all rows).")
else:
    print("WARNING: Retrieved few brands. Might still be truncated.")
