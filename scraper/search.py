import pandas as pd
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers not found. Search will be limited.")

import time
from pathlib import Path
import pickle

script_dir = Path(__file__).parent

try:
    from db import fetch_all_laptops
except ImportError:
    # Handle direct script execution where imports might differ
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from db import fetch_all_laptops


class LaptopSearchEngine:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 cache_path='embeddings_cache.pkl'):
        """
        Initialize search engine with data from Supabase and optional caching
        """
        self.cache_path = Path(cache_path)
        
        # Load from DB instead of CSV
        try:
            print("Initializing search engine with database data...")
            raw_data = fetch_all_laptops()
            if not raw_data:
                print("Warning: Database returned 0 rows.")
                self.df = pd.DataFrame()
            else:
                self.df = pd.DataFrame(raw_data)
                
            self.df.fillna('Unknown', inplace=True)
            print(f"✓ Loaded {len(self.df)} laptops from database")
            
        except Exception as e:
            print(f"Critical Error loading data from DB: {e}")
            self.df = pd.DataFrame()
        
        self.model = None
        self.embeddings = None
        self.sentences = []
        
        if not TRANSFORMERS_AVAILABLE:
            print("Search engine running in fallback mode (text matching only).")
            return

        print(f"Loading model ({model_name})...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Using device: {device}")
        
        # Try to load cached embeddings
        if self._load_cache():
            print("✓ Loaded embeddings from cache")
        else:
            print("Creating search index...")
            if not self.df.empty:
                self.sentences = self._create_search_strings()
                
                start_time = time.time()
                self.embeddings = self.model.encode(
                    self.sentences, 
                    convert_to_tensor=True, 
                    show_progress_bar=True,
                    batch_size=32
                )
                elapsed = time.time() - start_time
                print(f"✓ Indexing complete in {elapsed:.2f}s")
                
                # Save cache
                self._save_cache()
            else:
                print("Skipping indexing (no data)")
    
    def _create_search_strings(self):
        """Enhanced search string with all important fields"""
        search_strings = []
        
        for _, row in self.df.iterrows():
            # Include ALL relevant fields for better matching
            parts = [
                f"Title: {row['title']}.",
                f"Brand: {row['brand']}.",
                f"Model: {row['model']} {row.get('model_num', '')}.",
                f"Processor: {row['cpu']}.",
                f"Graphics: {row['gpu']}.",
                f"Memory: {row['ram']} RAM.",
                f"Storage: {row['storage']}.",
            ]
            
            # Add optional fields if they exist
            if row.get('screen_size'):
                parts.append(f"Screen: {row['screen_size']} inches.")
            if row.get('refresh_rate'):
                parts.append(f"Refresh rate: {row['refresh_rate']}Hz.")
            if row.get('is_new'):
                parts.append("Condition: New.")
            
            # Add description last (it's often verbose)
            parts.append(f"Description: {row.get('description', '')}")
            
            search_strings.append(' '.join(parts))
        
        return search_strings
    
    def _save_cache(self):
        """Save embeddings to disk for faster startup next time"""
        try:
            cache_data = {
                'embeddings': self.embeddings.cpu(),  # Move to CPU for saving
                'sentences': self.sentences,
                'df_length': len(self.df)
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Saved embeddings cache to {self.cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _load_cache(self):
        """Load cached embeddings if available and valid"""
        if not self.cache_path.exists():
            return False
        
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache is still valid
            if cache_data['df_length'] != len(self.df):
                print("Cache outdated (data changed)")
                return False
            
            # Load embeddings and move to appropriate device
            self.embeddings = cache_data['embeddings'].to(self.model.device)
            self.sentences = cache_data['sentences']
            
            return True
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return False
    
    def search(self, query, top_k=10, min_score=0.0, filters=None):
        """
        Enhanced search with filtering
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None or self.embeddings is None:
            # Fallback string matching
            results = []
            query_lower = query.lower()
            for _, row in self.df.iterrows():
                # Simple boolean match against title/brand/cpu
                text_repr = f"{row['title']} {row['brand']} {row['cpu']}".lower()
                if query_lower in text_repr:
                    item = row.to_dict()
                    item['match_score'] = 100.0 if query_lower in row['title'].lower() else 50.0
                    
                    # Apply filters
                    if filters:
                        if 'brand' in filters and item['brand'] != filters['brand']:
                            continue
                        if 'price_max' in filters and item['price'] > filters['price_max']:
                            continue
                        if 'price_min' in filters and item['price'] < filters['price_min']:
                            continue
                            
                    results.append(item)
                    if len(results) >= top_k:
                        break
            return results

        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Search
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k*3)[0]
        
        results = []
        for hit in hits:
            idx = hit['corpus_id']
            score = hit['score']
            
            # Score filter
            if score < min_score:
                continue
            
            item = self.df.iloc[idx].to_dict()
            item['match_score'] = round(score * 100, 2)
            
            # Apply filters
            if filters:
                if 'brand' in filters and item['brand'] != filters['brand']:
                    continue
                if 'price_max' in filters and item['price'] > filters['price_max']:
                    continue
                if 'price_min' in filters and item['price'] < filters['price_min']:
                    continue
            
            results.append(item)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_stats(self):
        """Return index statistics"""
        return {
            'total_laptops': len(self.df),
            'embedding_dims': self.embeddings.shape[1],
            'device': str(self.model.device),
            'cache_exists': self.cache_path.exists()
        }


# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    # Initialize (runs once, uses cache on subsequent runs)
    print("Starting Main...")
    # Add parent dir to path if running directly to find db.py
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    engine = LaptopSearchEngine()
    
    print("\n" + "="*70)
    print(" LAPTOP AI SEARCH (DB POWERED)")
    print("="*70)
    print(f" Stats: {engine.get_stats()}")
    print(" Commands: 'exit' to quit, 'stats' for info")
    print("="*70)
    
    while True:
        print()
        user_query = input("🔍 Search: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            break
        
        if user_query.lower() == 'stats':
            print(engine.get_stats())
            continue
        
        if not user_query:
            continue
        
        # Search
        start = time.time()
        results = engine.search(user_query, top_k=5)
        duration = (time.time() - start) * 1000  # Convert to ms
        
        print(f"\n✓ Found {len(results)} matches in {duration:.1f}ms\n")
        
        for i, res in enumerate(results, 1):
            print(f"{i}. [{res['match_score']}%] {res['title'][:65]}")
            # Handle potentially missing fields gracefully
            brand = res.get('brand', 'Unknown')
            cpu = res.get('cpu', 'Unknown')
            ram = res.get('ram', 'Unknown')
            gpu = res.get('gpu', 'Unknown')
            price = res.get('price', 0)
            
            print(f"   {brand} | {cpu} | {ram} | {gpu}")
            print(f"   💰 {price:.0f} DH")
            print()