"""
Avito Laptop Scraper V2 - Unified Module

Contains:
1. Fetching logic (HTTP, HTML parsing)
2. Scraping logic (Orchestration, CSV saving)
3. Main entry points
"""
import csv
import json
import random
import asyncio
import aiohttp
import logging
import re
import unicodedata
from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table


def clean_text(text: str) -> str:
    """
    Clean text for CSV safety - removes problematic characters.
    Applied to description field before saving.
    """
    if not text:
        return ""
    
    # 1. Replace newlines/returns with space
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 2. Remove CSV-problematic characters
    text = text.replace(',', ' ').replace('"', '').replace("'", '')
    
    # 3. Fix common unicode issues
    replacements = {
        '–': '-', '—': '-', '…': '...',
        ''': "'", ''': "'", '"': '"', '"': '"',
        '«': '', '»': '', '•': '-',
        '\u200b': '', '\u200c': '', '\u200d': '',  # Zero-width chars
        '\ufeff': '',  # BOM
        '\xa0': ' ',  # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 4. Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if ord(c) < 256 or unicodedata.category(c) in ['Ll', 'Lu', 'Nd', 'Zs'])
    
    # 5. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Configuration Imports
from .config import (
    BASE_URL, USER_AGENTS, REQUEST_TIMEOUT,
    OUTPUT_DIR, OUTPUT_FILE, OUTPUT_FIELDS,
    MAX_PAGES, BATCH_SIZE, DELAY_BETWEEN_BATCHES,
    MIN_PRICE, MAX_PRICE, MIN_COMPLETE_FIELDS
)

# Parser Imports
from .parser import SpecParser, LaptopListing

# Setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================================================================
# FETCHER LOGIC (Formerly fetcher.py)
# =========================================================================

def get_headers() -> dict:
    """Get request headers with random user agent"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Referer": "https://www.avito.ma/",
        "DNT": "1",
    }


def extract_next_data(html: str) -> Optional[dict]:
    """
    Extract __NEXT_DATA__ JSON from HTML page.
    """
    try:
        # First check if __NEXT_DATA__ exists in the raw HTML
        if '__NEXT_DATA__' not in html:
            logger.warning("__NEXT_DATA__ string not in HTML")
            return None
            
        soup = BeautifulSoup(html, 'html.parser')
        script_tag = soup.find('script', id='__NEXT_DATA__')
        
        if script_tag and script_tag.string:
            return json.loads(script_tag.string)
        
        # Try .get_text() if .string is None
        if script_tag:
            content = script_tag.get_text()
            if content:
                return json.loads(content)
        
        # Fallback: regex extraction
        logger.debug("Trying regex fallback for __NEXT_DATA__")
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        logger.warning("Could not extract __NEXT_DATA__")
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse __NEXT_DATA__ JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting __NEXT_DATA__: {e}")
        return None


def extract_listings(data: dict) -> list[dict]:
    """
    Extract listing data from parsed __NEXT_DATA__.
    """
    try:
        # Navigate to ads array
        # Path: props -> pageProps -> componentProps -> ads -> ads
        ads = (data.get('props', {})
               .get('pageProps', {})
               .get('componentProps', {})
               .get('ads', {})
               .get('ads', []))
        
        if not ads:
            logger.warning("No ads found in __NEXT_DATA__")
            return []
        
        listings = []
        for ad in ads:
            try:
                # Handle price which may be int or dict
                price_data = ad.get('price', {})
                if isinstance(price_data, dict):
                    price = float(price_data.get('value', 0))
                else:
                    price = float(price_data) if price_data else 0

                # Extract Avito ID from URL or JSON
                link = ad.get('href', '')
                avito_id = ad.get('id')
                if not avito_id and link:
                    # Extract ID from URL: ..._57060083.htm -> 57060083
                    id_match = re.search(r'_(\d+)\.htm', link)
                    if id_match:
                        avito_id = id_match.group(1)
                
                listing = {
                    'id': avito_id,  # Avito listing ID
                    'title': ad.get('subject', ''),
                    'price': price,
                    'city': ad.get('location', 'Unknown'),
                    'link': link,
                    'is_shop': ad.get('isShop', False),
                    'has_delivery': ad.get('hasShipping', False) or ad.get('isDelivery', False),
                    'description': ad.get('description', ''),
                }
                
                # Check for absolute URL
                if listing['link'] and not listing['link'].startswith('http'):
                    listing['link'] = f"https://www.avito.ma{listing['link']}"
                
                # Skip if no title or invalid price
                if listing['title'] and listing['price'] >= 0:
                    listings.append(listing)
                    
            except Exception as e:
                logger.debug(f"Error processing ad: {e}")
                continue
        
        return listings
        
    except Exception as e:
        logger.error(f"Error extracting listings: {e}")
        return []


async def fetch_page(session: aiohttp.ClientSession, page_num: int) -> list[dict]:
    """
    Fetch a single listing page and extract listings.
    """
    url = f"{BASE_URL}?o={page_num}"
    
    try:
        async with session.get(
            url,
            headers=get_headers(),
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        ) as response:
            
            if response.status != 200:
                logger.warning(f"Page {page_num}: HTTP {response.status}")
                return []
            
            html = await response.text()
            logger.debug(f"Page {page_num}: Got {len(html)} chars")
            
            data = extract_next_data(html)
            
            if not data:
                logger.warning(f"Page {page_num}: No data extracted")
                return []
            
            listings = extract_listings(data)
            logger.info(f"Page {page_num}: Found {len(listings)} listings")
            
            return listings
            
    except asyncio.TimeoutError:
        logger.warning(f"Page {page_num}: Timeout")
        return []
    except aiohttp.ClientError as e:
        logger.warning(f"Page {page_num}: Client error - {e}")
        return []
    except Exception as e:
        logger.error(f"Page {page_num}: Unexpected error - {e}")
        return []


async def fetch_pages_batch(
    session: aiohttp.ClientSession,
    start_page: int,
    end_page: int
) -> list[dict]:
    """
    Fetch multiple pages concurrently.
    """
    tasks = [
        fetch_page(session, page_num)
        for page_num in range(start_page, end_page)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_listings = []
    for result in results:
        if isinstance(result, list):
            all_listings.extend(result)
        elif isinstance(result, Exception):
            logger.error(f"Batch fetch exception: {result}")
    
    return all_listings


# =========================================================================
# SCRAPER LOGIC
# =========================================================================

def process_listing(raw: dict) -> LaptopListing | None:
    """
    Process a raw listing into a validated LaptopListing.
    
    EXTRACTION ORDER: Description first (has more details), title as fallback.
    """
    try:
        title = raw.get('title', '')
        description = clean_text(raw.get('description', ''))  # Clean for CSV safety
        
        # DESCRIPTION PRIORITY: Parse description first (usually has more specific info)
        # Example: Description has "i7-12800H" vs Title has just "i7 12eme"
        if description:
            desc_specs = SpecParser.parse(description)
        else:
            desc_specs = {}
        
        # Parse title
        title_specs = SpecParser.parse(title)
        
        # MERGE: Start with description, use title to fill gaps
        final_specs = desc_specs.copy() if desc_specs else {}
        for key in ['brand', 'model', 'model_num', 'model_detail', 'cpu', 'ram', 'storage', 
                    'gpu', 'gpu_vram', 'screen_size', 'is_new', 'is_touchscreen', 'refresh_rate']:
            # Use title value if description didn't extract it
            if final_specs.get(key) in ['Unknown', '', None, False, 0] and title_specs.get(key) not in ['Unknown', '', None, False, 0]:
                final_specs[key] = title_specs[key]
            # If neither extracted, ensure key exists
            elif key not in final_specs:
                final_specs[key] = title_specs.get(key, 'Unknown')
        
        # Infer integrated GPU if missing
        if final_specs.get('gpu') == 'Unknown':
            cpu = final_specs.get('cpu', '')
            if cpu.startswith('M1') or cpu.startswith('M2') or cpu.startswith('M3') or cpu.startswith('M4') or cpu.startswith('M5'):
                final_specs['gpu'] = 'Apple GPU'
            elif 'ULTRA' in cpu.upper():
                final_specs['gpu'] = 'Intel Arc'
            elif cpu.upper().startswith('I') or 'CORE' in cpu.upper():
                final_specs['gpu'] = 'Intel UHD'
            elif 'RYZEN' in cpu.upper():
                final_specs['gpu'] = 'AMD Radeon'
        
        # Create validated model (scores computed automatically via properties)
        listing = LaptopListing(
            title=title,
            description=description,
            price=raw.get('price', 0),
            city=raw.get('city', 'Unknown'),
            link=raw.get('link', ''),
            brand=final_specs.get('brand', 'Unknown'),
            model=final_specs.get('model', 'Unknown'),
            model_num=final_specs.get('model_num'),
            cpu=final_specs.get('cpu', 'Unknown'),
            ram=final_specs.get('ram', 'Unknown'),
            storage=final_specs.get('storage', 'Unknown'),
            gpu=final_specs.get('gpu', 'Unknown'),
            gpu_vram=final_specs.get('gpu_vram'),
            is_shop=raw.get('is_shop', False),
            has_delivery=raw.get('has_delivery', False),
            # Feature fields
            screen_size=final_specs.get('screen_size'),
            is_new=final_specs.get('is_new', False),
            is_touchscreen=final_specs.get('is_touchscreen', False),
            refresh_rate=final_specs.get('refresh_rate'),
        )
        
        return listing
        
    except ValueError as e:
        logger.debug(f"Validation failed: {e}")
        return None
    except Exception as e:
        logger.debug(f"Processing error: {e}")
        return None


async def scrape_all(
    max_pages: int = MAX_PAGES,
    batch_size: int = BATCH_SIZE,
    min_completeness: int = MIN_COMPLETE_FIELDS,
    progress_callback: Optional[Callable[[float, str, Dict], None]] = None,
    skip_filtering: bool = False,
) -> list[LaptopListing]:
    """
    Scrape all pages and return validated listings.
    Args:
        progress_callback: Function(percent, message, stats) called periodically.
        skip_filtering: If True, keep incomplete listings (useful for refresh/LLM fix).
    """
    """
    Scrape all pages and return validated listings.
    """
    all_listings: list[LaptopListing] = []
    seen_ids: set[str] = set()
    
    stats = {
        'total_raw': 0,
        'valid': 0,
        'duplicates': 0,
        'incomplete': 0,
        'invalid_price': 0,
    }
    
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[cyan]Scraping pages...",
                total=max_pages
            )
            
            for batch_start in range(1, max_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size, max_pages + 1)
                
                # Fetch batch
                raw_listings = await fetch_pages_batch(session, batch_start, batch_end)
                stats['total_raw'] += len(raw_listings)
                
                # Process each listing
                for raw in raw_listings:
                    # Skip duplicates
                    listing_id = raw.get('id')
                    if listing_id and listing_id in seen_ids:
                        stats['duplicates'] += 1
                        continue
                    if listing_id:
                        seen_ids.add(listing_id)
                    
                    # Validate and process
                    listing = process_listing(raw)
                    
                    if listing is None:
                        stats['invalid_price'] += 1
                        continue
                    

                    # Check completeness - REMOVED: Filter AFTER LLM cleaning
                    # if not listing.is_complete_enough():
                    #     stats['incomplete'] += 1
                    #     continue
                    
                    all_listings.append(listing)
                    stats['valid'] += 1
                
                # Update progress
                progress.update(task, advance=batch_end - batch_start)
                
                # Report progress via callback
                if progress_callback:
                    pct = min(80, (batch_end / max_pages) * 80) # Scraping is first 80%
                    progress_callback(pct, f"Scraping page {batch_start}-{min(batch_end, max_pages)}", stats)
                
                # Rate limiting
                if batch_end < max_pages:
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)
                
                # Periodic status
                if batch_start % 50 == 1:
                    console.print(
                        f"[dim]Progress: {len(all_listings)} fetched listings, "
                        f"{stats['duplicates']} duplicates[/dim]"
                    )
    
    # NOTE: LLM Cleaning removed - handled by refresh.py using multi_cleaner.py
    # This ensures only NEW items are cleaned, not all 14K+

    # =========================================================================
    # FINAL FILTERING STEP (Post-LLM)
    # =========================================================================
    console.print(f"\n[dim]Filtering rows with < {min_completeness} known fields...[/dim]")
    final_listings = []
    
    # Reset valid count to track final kept
    stats['valid'] = 0 
    
    if skip_filtering:
        console.print("[dim]Skipping completeness filter (Refresh Mode)...[/dim]")
        final_listings = all_listings
        stats['valid'] = len(all_listings)
        if progress_callback:
            progress_callback(95, "Finalizing data...", stats)
    else:
        for listing in all_listings:
            if listing.is_complete_enough(): # Uses the default or configured threshold
                 final_listings.append(listing)
                 stats['valid'] += 1
            else:
                 stats['incomplete'] += 1
    
    all_listings = final_listings
    
    return all_listings, stats



def save_to_csv(listings: list[LaptopListing], output_path: Path = OUTPUT_FILE):
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        
        for listing in listings:
            writer.writerow(listing.to_dict())
    
    console.print(f"[green]✓ Saved {len(listings)} listings to {output_path}[/green]")

# NOTE: save_to_db removed - DB operations are now handled by refresh.py in routers/


def print_stats(listings: list[LaptopListing], stats: dict):
    """Print scraping statistics."""
    
    console.print("\n[bold]Scraping Statistics[/bold]")
    
    # General stats
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Raw Listings", str(stats['total_raw']))
    table.add_row("Valid (Kept)", str(stats['valid']))
    table.add_row("Duplicates (Skipped)", str(stats['duplicates']))
    table.add_row("Incomplete (Filtered)", str(stats['incomplete']))
    table.add_row("Invalid Price (Rejected)", str(stats['invalid_price']))
    
    console.print(table)
    
    # Field completeness
    if listings:
        fields = ['brand', 'cpu', 'ram', 'storage', 'gpu']
        completeness = {}
        
        for field in fields:
            known = sum(1 for l in listings if getattr(l, field) != "Unknown")
            completeness[field] = (known / len(listings)) * 100
        
        table2 = Table(title="Field Completeness")
        table2.add_column("Field", style="cyan")
        table2.add_column("Known %", style="green")
        
        for field, pct in completeness.items():
            color = "green" if pct > 80 else "yellow" if pct > 60 else "red"
            table2.add_row(field.upper(), f"[{color}]{pct:.1f}%[/{color}]")
        
        console.print(table2)


def print_sample(listings: list[LaptopListing], n: int = 5):
    """Print sample listings."""
    
    table = Table(title=f"Sample Listings (First {n})")
    table.add_column("Title", width=30)
    table.add_column("Price", style="green")
    table.add_column("Brand", style="cyan")
    table.add_column("CPU", style="magenta")
    table.add_column("RAM", style="yellow")
    
    for listing in listings[:n]:
        table.add_row(
            listing.title[:27] + "..." if len(listing.title) > 30 else listing.title,
            f"{listing.price:.0f} DH",
            listing.brand,
            listing.cpu[:20] if len(listing.cpu) > 20 else listing.cpu,
            listing.ram,
        )
    
    console.print(table)


def main_scraper_entry(
    max_pages: int = None,
    min_completeness: int = None,
    output_path: Path = None,
    use_db: bool = False
):
    """Main entry point."""
    
    # Use defaults if not specified
    max_pages = max_pages or MAX_PAGES
    min_completeness = min_completeness or MIN_COMPLETE_FIELDS
    output_path = output_path or OUTPUT_FILE
    
    print(f"Scraping {max_pages} pages. Min fields: {min_completeness}.")
    
    start_time = datetime.now()
    
    # Scrape
    listings, stats = asyncio.run(scrape_all(
        max_pages=max_pages,
        min_completeness=min_completeness
    ))
    
    elapsed = datetime.now() - start_time
    
    # Save
    if listings:
        save_to_csv(listings, output_path)
        if use_db:
            save_to_db(listings)
    
    print(f"Completed in {elapsed.total_seconds():.1f}s")


def run(max_pages: int = 10, min_completeness: int = 3, use_db: bool = False):
    """
    Wrapper for running the scraper.
    """
    main_scraper_entry(
        max_pages=max_pages,
        min_completeness=min_completeness,
        use_db=use_db
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Avito Laptop Scraper")
    parser.add_argument(
        "--pages", "-p",
        type=int,
        default=10,
        help="Number of pages to scrape (default: 10, max ~450)"
    )
    parser.add_argument(
        "--min-fields", "-m",
        type=int,
        default=3,
        help="Minimum non-Unknown fields to keep row (default: 3)"
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Write to Supabase database"
    )
    
    args = parser.parse_args()
    run(
        max_pages=args.pages,
        min_completeness=args.min_fields,
        use_db=args.db
    )
