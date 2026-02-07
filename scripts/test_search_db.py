
import sys
import os
import io

# Setup path and dummy env
sys.path.append(os.getcwd())
os.environ["GEMINI_API_KEY"] = "dummy"

# Ensure we're testing the DB integration, not a fallback or mock
# We need dotenv to load Supabase keys for db.py
from dotenv import load_dotenv
load_dotenv()

try:
    print("Testing DB-backed Search Engine...")
    from scraper.search import LaptopSearchEngine
    
    # Force re-initialization (ignoring if it was already imported/cached in memory in a real app)
    engine = LaptopSearchEngine()
    
    stats = engine.get_stats()
    print(f"Stats: {stats}")
    
    count = stats.get('total_laptops', 0)
    if count > 0:
        print(f"SUCCESS: Engine initialized with {count} laptops from DB.")
    else:
        print("WARNING: Engine initialized but found 0 laptops.")
        
    # Test search
    results = engine.search("hp laptop")
    print(f"Search 'hp laptop' returned {len(results)} results.")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
