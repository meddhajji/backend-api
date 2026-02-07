"""
Supabase Database Client
========================
Async-compatible database access layer for the laptops table.
Replaces direct CSV reads for Vercel deployment.
"""
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

_client: Optional[Client] = None


def get_client() -> Client:
    """Get or create Supabase client (singleton)."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def get_all_laptops(limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Fetch laptops from Supabase with pagination.
    Returns list of laptop dictionaries.
    """
    client = get_client()
    response = client.table('laptops') \
        .select('*') \
        .order('laptop_score', desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()
    return response.data


def get_distinct_values(column: str) -> List[str]:
    """
    Get unique values for a column (for filter dropdowns).
    Fetches ALL rows to ensure complete list, then normalizes.
    """
    client = get_client()
    values = set()
    offset = 0
    limit = 1000
    
    while True:
        try:
            response = client.table('laptops') \
                .select(column) \
                .range(offset, offset + limit - 1) \
                .execute()
            
            if not response.data:
                break
                
            for row in response.data:
                val = row.get(column)
                # Exclude None, empty, and any variation of "unknown"
                if val and val not in ['Unknown', 'unknown', 'UNKNOWN', '', None]:
                    # Normalize strings
                    if isinstance(val, str):
                        val = val.strip()
                        # Simple capitalization: first letter uppercase only
                        if column in ['brand', 'city']:
                            lower_val = val.lower()
                            val = lower_val[0].upper() + lower_val[1:] if lower_val else val
                    
                    values.add(val)
            
            if len(response.data) < limit:
                break
                
            offset += limit
            
        except Exception as e:
            print(f"Error fetching distinct values: {e}")
            break
            
    return sorted(list(values))


def search_by_filters(
    filters: Dict[str, Any],
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Search laptops with multiple filters.
    """
    client = get_client()
    
    query = client.table('laptops').select('*', count='exact')
    
    # String filters (case-insensitive exact match via ilike)
    string_fields = ['brand', 'city', 'model', 'cpu', 'gpu', 'cpu_family', 
                     'gpu_type', 'gpu_family', 'gpu_vram', 'storage', 'ram']
    for field in string_fields:
        if field in filters and filters[field]:
            # Use ilike for case-insensitive matching
            # Note: ilike is typically for pattern matching, but without wildcards it acts as case-insensitive equals
            query = query.ilike(field, filters[field])
    
    # Numeric range filters
    range_fields = {
        'price': 'price',
        'ram_gb': 'ram_gb',
        'storage_gb': 'storage_gb',
        'screen_size': 'screen_size',
        'refresh_rate': 'refresh_rate',
    }
    for key, column in range_fields.items():
        min_key = f'{key}_min'
        max_key = f'{key}_max'
        if min_key in filters and filters[min_key] is not None:
            query = query.gte(column, filters[min_key])
        if max_key in filters and filters[max_key] is not None:
            query = query.lte(column, filters[max_key])
    
    # Boolean filters
    bool_fields = ['is_new', 'is_touchscreen', 'is_ssd']
    for field in bool_fields:
        if field in filters and filters[field] is not None:
            query = query.eq(field, filters[field])
    
    # Order by score and paginate
    query = query.order('laptop_score', desc=True)
    query = query.range(offset, offset + limit - 1)
    
    response = query.execute()
    
    return {
        'results': response.data,
        'total': response.count or len(response.data),
        'offset': offset,
        'limit': limit
    }


def get_laptop_by_id(laptop_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single laptop by ID."""
    client = get_client()
    response = client.table('laptops') \
        .select('*') \
        .eq('id', laptop_id) \
        .single() \
        .execute()
    return response.data


def get_stats() -> Dict[str, Any]:
    """Get database statistics."""
    client = get_client()
    
    # Total count
    count_response = client.table('laptops').select('id', count='exact').execute()
    total = count_response.count or 0
    
    # Price range - filter out nulls and zeros
    price_response = client.table('laptops') \
        .select('price') \
        .gt('price', 0) \
        .order('price', desc=False) \
        .limit(1) \
        .execute()
    min_price = price_response.data[0]['price'] if price_response.data else 0
    
    price_response = client.table('laptops') \
        .select('price') \
        .gt('price', 0) \
        .order('price', desc=True) \
        .limit(1) \
        .execute()
    max_price = price_response.data[0]['price'] if price_response.data else 0
    
    return {
        'total_laptops': total,
        'price_range': {'min': min_price, 'max': max_price}
    }


def fetch_all_laptops() -> List[Dict[str, Any]]:
    """
    Fetch ALL laptops from the database for the search engine.
    Handles pagination automatically to retrieve full dataset.
    """
    client = get_client()
    all_laptops = []
    offset = 0
    limit = 1000
    
    print("Fetching all laptops from database...")
    while True:
        try:
            response = client.table('laptops') \
                .select('*') \
                .range(offset, offset + limit - 1) \
                .execute()
            
            data = response.data
            if not data:
                break
                
            all_laptops.extend(data)
            
            if len(data) < limit:
                break
                
            offset += limit
            print(f"  Fetched {len(all_laptops)} rows...")
            
        except Exception as e:
            print(f"Error fetching laptops: {e}")
            break
            
    print(f"Total laptops fetched: {len(all_laptops)}")
    return all_laptops
