"""
Multi-Key LLM Data Cleaner
==========================
Uses 10 parallel Gemini API keys to process batches of laptop data concurrently.
Implements smart NaN threshold filtering based on total item count.
"""
import os
import re
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

MODEL_NAME = "gemma-3-27b-it"
BATCH_SIZE = 40  # Items per API call
MAX_PARALLEL_KEYS = 10

# System instruction for LLM
SYSTEM_INSTRUCTION = """You are a laptop specification extractor.
Your task is to extract MISSING specifications from laptop listings.

INPUT FORMAT:
ROW|id|title|number_contexts|MISSING:field1,field2...

Where number_contexts contains only the numbers from the description with their surrounding context.

FIELDS TO EXTRACT (only those listed after MISSING:):
- cpu: full name (e.g., "Core i5-1235U")
- ram: amount + unit (e.g., "16GB")
- storage: amount + type (e.g., "512GB SSD")
- gpu: full name (e.g., "RTX 3050")
- brand, model, screen_size (e.g. "15.6"), refresh_rate (e.g. "144Hz")
- is_new (True/False), is_touchscreen (True/False)

OUTPUT FORMAT (raw text only, one per line):
id|field=value|field2=value2

RULES:
- Do NOT guess. If not found, do not output that field.
- Use standard formats (e.g. "16GB", "512GB SSD").
- For is_new/is_touchscreen, output 'True' or 'False'."""


def load_all_api_keys() -> List[str]:
    """Load all available Gemini API keys from environment."""
    keys = []
    # Primary key
    primary = os.getenv("GEMINI_API_KEY")
    if primary:
        keys.append(primary)
    
    # Additional keys (GEMINI_API_KEY1 through GEMINI_API_KEY9)
    for i in range(1, 10):
        key = os.getenv(f"GEMINI_API_KEY{i}")
        if key:
            keys.append(key)
    
    logger.info(f"Loaded {len(keys)} API keys")
    return keys


def get_nan_threshold(total_items: int) -> int:
    """
    Determine NaN threshold based on total new items.
    Returns minimum number of NaN fields required to trigger LLM processing.
    """
    if total_items > 2000:
        return 5  # Only process if >5 fields are NaN
    elif total_items >= 500:
        return 4  # Only process if >4 fields are NaN
    else:
        return 3  # Only process if >3 fields are NaN


def count_nan_fields(item: Dict) -> int:
    """Count how many key spec fields are None/NaN."""
    key_fields = ['cpu', 'ram', 'storage', 'gpu', 'brand', 'model', 
                  'screen_size', 'refresh_rate', 'is_new', 'is_touchscreen']
    count = 0
    for field in key_fields:
        val = item.get(field)
        if val is None or (isinstance(val, str) and val.lower() in ['', 'nan', 'none', 'unknown']):
            count += 1
    return count


def extract_number_context(text: str, before: int = 12, after: int = 10) -> str:
    """Extract only context around numbers to reduce token count."""
    if not text:
        return ""
    
    contexts = []
    for match in re.finditer(r'\d+', text):
        start = max(0, match.start() - before)
        end = min(len(text), match.end() + after)
        contexts.append(text[start:end])
    
    if not contexts:
        return ""
    
    return " | ".join(contexts)[:500]


def format_batch(items: List[Dict]) -> str:
    """Format a batch of items for LLM prompt."""
    lines = []
    check_fields = ['cpu', 'ram', 'storage', 'gpu', 'brand', 'model',
                    'screen_size', 'refresh_rate', 'is_new', 'is_touchscreen']
    
    for item in items:
        item_id = item.get('link', item.get('id', 'unknown'))
        
        # Find missing fields
        missing = []
        for field in check_fields:
            val = item.get(field)
            if val is None or (isinstance(val, str) and val.lower() in ['', 'nan', 'none', 'unknown']):
                missing.append(field)
        
        if not missing:
            continue
        
        title = str(item.get('title', '')).replace('|', ' ')
        desc = str(item.get('description', ''))
        number_context = extract_number_context(desc)
        
        lines.append(f"ROW|{item_id}|{title}|{number_context}|MISSING:{','.join(missing)}")
    
    return "\n".join(lines)


def parse_llm_response(response_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse LLM response into updates dictionary."""
    updates = {}
    
    for line in response_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        
        parts = line.split("|")
        item_id = parts[0].strip()
        
        row_updates = {}
        for part in parts[1:]:
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"\'')
                
                # Type conversion
                if key in ['is_new', 'is_touchscreen']:
                    if val.lower() == 'true':
                        row_updates[key] = True
                    elif val.lower() == 'false':
                        row_updates[key] = False
                elif key == 'refresh_rate':
                    match = re.search(r'(\d+)', val)
                    if match:
                        row_updates[key] = int(match.group(1))
                elif key == 'screen_size':
                    match = re.search(r'(\d+(?:\.\d+)?)', val)
                    if match:
                        row_updates[key] = float(match.group(1))
                else:
                    if val.lower() not in ['unknown', 'none', 'nan', '']:
                        row_updates[key] = val
        
        if row_updates:
            updates[item_id] = row_updates
    
    return updates


class MultiKeyDataCleaner:
    """
    Parallel LLM processor using multiple API keys.
    Processes 10 batches of 40 items simultaneously = 400 items/cycle.
    """
    
    def __init__(self):
        self.api_keys = load_all_api_keys()
        if not self.api_keys:
            raise ValueError("No GEMINI_API_KEY found in environment")
        
        self.clients = [genai.Client(api_key=key) for key in self.api_keys]
        self.model = MODEL_NAME
        self._executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_KEYS)
    
    def _process_single_batch(self, batch: List[Dict], client_idx: int) -> List[Dict]:
        """Process a single batch with one API key (synchronous)."""
        if not batch:
            return batch
        
        client = self.clients[client_idx % len(self.clients)]
        batch_text = format_batch(batch)
        
        if not batch_text.strip():
            return batch  # No items need processing
        
        prompt = f"{SYSTEM_INSTRUCTION}\n\nDATA TO PROCESS:\n{batch_text}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                
                updates = parse_llm_response(response.text)
                
                # Apply updates to batch
                result = []
                for item in batch:
                    item_id = item.get('link', item.get('id', 'unknown'))
                    if item_id in updates:
                        item_copy = item.copy()
                        item_copy.update(updates[item_id])
                        result.append(item_copy)
                    else:
                        result.append(item)
                
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                if '429' in str(e) or 'quota' in error_str or 'rate' in error_str:
                    wait_time = 60 * (2 ** attempt)
                    logger.warning(f"Rate limit on key {client_idx}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error processing batch with key {client_idx}: {e}")
                    break
        
        return batch  # Return original on failure
    
    async def process_items(
        self, 
        items: List[Dict], 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict]:
        """
        Process items in parallel using all available API keys.
        
        Args:
            items: List of item dicts to process
            progress_callback: Callback(processed, total, message)
        
        Returns:
            List of processed items with filled specs
        """
        total_items = len(items)
        threshold = get_nan_threshold(total_items)
        
        # Filter items that need LLM processing
        needs_llm = [item for item in items if count_nan_fields(item) > threshold]
        already_complete = [item for item in items if count_nan_fields(item) <= threshold]
        
        logger.info(f"Total: {total_items}, Need LLM: {len(needs_llm)}, Already complete: {len(already_complete)} (threshold: >{threshold} NaN fields)")
        
        if progress_callback:
            progress_callback(0, len(needs_llm), f"Processing {len(needs_llm)} items with LLM...")
        
        if not needs_llm:
            if progress_callback:
                progress_callback(total_items, total_items, "All items already identified")
            return items
        
        # Split into batches of BATCH_SIZE
        batches = [needs_llm[i:i + BATCH_SIZE] for i in range(0, len(needs_llm), BATCH_SIZE)]
        
        processed_results = []
        processed_count = 0
        
        # Process in groups of MAX_PARALLEL_KEYS batches
        loop = asyncio.get_event_loop()
        
        for group_start in range(0, len(batches), MAX_PARALLEL_KEYS):
            group = batches[group_start:group_start + MAX_PARALLEL_KEYS]
            
            # Run batches in parallel using thread pool
            futures = []
            for idx, batch in enumerate(group):
                client_idx = idx % len(self.clients)
                future = loop.run_in_executor(
                    self._executor,
                    self._process_single_batch,
                    batch,
                    client_idx
                )
                futures.append(future)
            
            # Wait for all in group to complete
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch failed: {result}")
                    continue
                processed_results.extend(result)
                processed_count += len(result) if isinstance(result, list) else 0
            
            if progress_callback:
                progress_callback(processed_count, len(needs_llm), f"LLM: {processed_count}/{len(needs_llm)}")
            
            # Small delay between groups to avoid rate limits
            if group_start + MAX_PARALLEL_KEYS < len(batches):
                await asyncio.sleep(2)
        
        # Combine results: already complete + processed
        # Create lookup by link for processed items
        processed_lookup = {item.get('link', item.get('id')): item for item in processed_results}
        
        final_results = []
        for item in items:
            item_key = item.get('link', item.get('id'))
            if item_key in processed_lookup:
                final_results.append(processed_lookup[item_key])
            else:
                final_results.append(item)
        
        return final_results


# Convenience function for synchronous use
def clean_new_items(items: List[Dict], progress_callback=None) -> List[Dict]:
    """
    Clean a list of new items using multi-key parallel processing.
    This is the main entry point for the refresh router.
    """
    cleaner = MultiKeyDataCleaner()
    return asyncio.run(cleaner.process_items(items, progress_callback))
