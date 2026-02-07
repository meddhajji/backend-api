"""
Configuration constants for the Avito scraper
"""
from pathlib import Path

# URLs
BASE_URL = "https://www.avito.ma/fr/maroc/ordinateurs_portables"

# Output - data folder inside scraper (like epl project structure)
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "laptops.csv"

# Scraping settings
MAX_PAGES = 450  # ~15,750 items (35/page)
ITEMS_PER_PAGE = 35
REQUEST_TIMEOUT = 15
DELAY_BETWEEN_BATCHES = 2.0  # seconds
BATCH_SIZE = 10  # pages per batch

# Validation thresholds
MIN_PRICE = 100  # DH
MAX_PRICE = 150000  # DH
MIN_COMPLETE_FIELDS = 4  # Minimum non-Unknown fields to keep row

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Fields to extract
OUTPUT_FIELDS = [
    "title",
    "description",  # Raw description from listing
    "price", 
    "city",
    "brand",
    "model",
    "model_num",
    "cpu",
    "ram",
    "storage",
    "gpu",
    "gpu_vram",
    "is_shop",
    "has_delivery",
    "link",
    # Feature fields
    "screen_size",
    "is_new",
    "is_touchscreen",
    "refresh_rate",
    # Derived features
    "cpu_family",
    "cpu_generation",
    "gpu_type",
    "gpu_family",
    "is_ssd",
    "ram_gb",
    "storage_gb",
    # Scores (0-1000 scale, weighted for laptop_score)
    "cpu_score",
    "gpu_score",  # NEW
    "ram_score",
    "storage_score",
    "screen_score",  # NEW
    "condition_score",  # NEW
    "laptop_score",  # Weighted combined
]

# Moroccan cities (normalized names for fuzzy matching)
MOROCCAN_CITIES = [
    "Casablanca", "Rabat", "Marrakech", "Fès", "Tanger", "Agadir", "Meknès",
    "Oujda", "Kénitra", "Témara", "Salé", "Tétouan", "Mohammedia", "El Jadida",
    "Nador", "Beni Mellal", "Khémisset", "Khouribga", "Settat", "Taza",
    "Berrechid", "Errachidia", "Guelmim", "Larache", "Safi", "Berkane",
    "Ifrane", "Ouarzazate", "Essaouira", "Dakhla", "Laâyoune", "Fnideq",
    "Al Hoceima", "Chefchaouen", "Azrou", "Taroudant", "Tiznit",
]
