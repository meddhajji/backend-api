
import sys
import os
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()

try:
    from db import get_client
    client = get_client()
    # Fetch 1 row to see columns
    response = client.table('laptops').select('*').limit(1).execute()
    if response.data:
        print("Available columns:", list(response.data[0].keys()))
    else:
        print("Table is empty.")
except Exception as e:
    print(f"Error: {e}")
