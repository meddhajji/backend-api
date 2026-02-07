"""
Parser Module
Contains: SpecParser, LaptopListing model, city normalization, and CPU scoring utilities.
"""
import re
import csv
from pathlib import Path
from typing import Optional, List, Tuple, Dict, ClassVar
from dataclasses import dataclass
from pydantic import BaseModel, field_validator, model_validator
from rapidfuzz import fuzz, process

from .config import MIN_PRICE, MAX_PRICE, MIN_COMPLETE_FIELDS, MOROCCAN_CITIES


# =========================================================================
# CPU SCORING ENGINE
# =========================================================================

class CPUScorer:
    """
    CPU benchmark scoring with word-boundary matching and layered fallbacks.
    Normalizes scores to 0-1000 scale.
    """
    
    _instance = None
    _cpu_db: Dict[str, int] = {}
    _max_score: int = 1
    _loaded: bool = False
    
    @classmethod
    def _load_database(cls):
        """Load CPU database from csv on first use"""
        if cls._loaded:
            return
        
        # Find cpu.csv relative to this file
        data_path = Path(__file__).parent / "data" / "cpu.csv"
        if not data_path.exists():
            cls._loaded = True
            return
        
        # Mobile CPU suffixes (laptop CPUs)
        mobile_suffixes = ('H', 'HX', 'HK', 'HS', 'U', 'P', 'G', 'M', 'HQ', 'MQ', 'MX', 'H45', 'H55')
        laptop_max = 0
        
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cpu_name = row['cpu'].strip()
                mark = int(row['mark'])
                if cpu_name not in cls._cpu_db or mark > cls._cpu_db[cpu_name]:
                    cls._cpu_db[cpu_name] = mark
                
                # Track laptop-only max score for better normalization
                # Mobile CPUs have suffixes like U, H, HX, P, etc.
                if any(cpu_name.upper().endswith(s) for s in mobile_suffixes):
                    laptop_max = max(laptop_max, mark)
                elif 'Apple M' in cpu_name or 'M1' in cpu_name or 'M2' in cpu_name or 'M3' in cpu_name:
                    laptop_max = max(laptop_max, mark)
        
        if cls._cpu_db:
            # Use laptop-only max for better normalization (mobile CPUs top out ~55k vs ~65k desktop)
            cls._max_score = laptop_max if laptop_max > 0 else max(cls._cpu_db.values())
        cls._loaded = True
    
    @classmethod
    def _find_word_boundary_matches(cls, query: str) -> List[Tuple[str, int]]:
        """Match query as word boundary (not embedded in other words)"""
        query_lower = query.lower().strip()
        query_escaped = re.escape(query_lower)
        pattern = rf'(?:^|[\s\-])({query_escaped})(?:[\s\-@]|$)'
        
        matches = []
        for cpu_name, score in cls._cpu_db.items():
            if re.search(pattern, cpu_name.lower()):
                matches.append((cpu_name, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    @classmethod
    def _find_substring_matches(cls, query: str) -> List[Tuple[str, int]]:
        """Fallback: substring matching"""
        query_lower = query.lower().strip()
        matches = []
        for cpu_name, score in cls._cpu_db.items():
            if query_lower in cpu_name.lower():
                matches.append((cpu_name, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    @classmethod
    def _fallback_dash_to_space(cls, query: str) -> List[Tuple[str, int]]:
        """Replace '-' with ' ' for Intel Core/Ultra"""
        query_lower = query.lower()
        is_intel_new = any(x in query_lower for x in ['core 7', 'core 9', 'ultra 5', 'ultra 7', 'ultra 9', 'core ultra'])
        is_i_series = any(x in query_lower for x in ['i3', 'i5', 'i7', 'i9'])
        if is_intel_new and not is_i_series:
            return cls._find_word_boundary_matches(query.replace('-', ' '))
        return []
    
    @classmethod
    def _fallback_intel_4digit_expand(cls, query: str) -> List[Tuple[str, int]]:
        """For Intel i-series 4-digit, add '0' (i7-1085H → i7-10850H)"""
        match = re.search(r'(i[3579])-?(\d{4})([a-z]*)', query.lower())
        if match:
            prefix, digits, suffix = match.groups()
            expanded = digits[:3] + '0' + digits[3]
            return cls._find_word_boundary_matches(f"{prefix}-{expanded}{suffix}")
        return []
    
    @classmethod
    def _fallback_progressive_removal(cls, query: str) -> List[Tuple[str, int]]:
        """Remove digits progressively (i7-6300U → i7-6)"""
        match = re.search(r'(i[3579])-?(\d+)([a-z]*)', query.lower())
        if match:
            prefix, digits, _ = match.groups()
            for length in range(len(digits) - 1, 0, -1):
                matches = cls._find_word_boundary_matches(f"{prefix}-{digits[:length]}")
                if matches:
                    return matches
        return []
    
    @classmethod
    def _fallback_token_all_match(cls, query: str) -> List[Tuple[str, int]]:
        """All tokens must exist in CPU name"""
        tokens = re.split(r'[\s\-]+', query.lower())
        tokens = [t for t in tokens if len(t) >= 1]
        if not tokens:
            return []
        matches = []
        for cpu_name, score in cls._cpu_db.items():
            cpu_lower = cpu_name.lower()
            if all(token in cpu_lower for token in tokens):
                matches.append((cpu_name, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    @classmethod
    def _fallback_prioritized_tokens(cls, query: str) -> List[Tuple[str, int]]:
        """Prioritize family over model (Ryzen 7 7000 → Ryzen 7)"""
        query_lower = query.lower()
        priority_pairs = [
            ('ryzen', '3'), ('ryzen', '5'), ('ryzen', '7'), ('ryzen', '9'),
            ('ultra', '5'), ('ultra', '7'), ('ultra', '9'),
            ('core', '3'), ('core', '5'), ('core', '7'), ('core', '9'),
        ]
        for brand, tier in priority_pairs:
            if brand in query_lower and tier in re.split(r'[\s\-]+', query_lower):
                matches = []
                for cpu_name, score in cls._cpu_db.items():
                    cpu_lower = cpu_name.lower()
                    if brand in cpu_lower and re.search(rf'\b{tier}\b', cpu_lower):
                        matches.append((cpu_name, score))
                if matches:
                    matches.sort(key=lambda x: x[1], reverse=True)
                    return matches
        return []
    
    @classmethod
    def _fallback_separate_attached(cls, query: str) -> List[Tuple[str, int]]:
        """Separate attached (ryzen7 → ryzen 7)"""
        separated = re.sub(r'([a-z])(\d)', r'\1 \2', query.lower())
        separated = re.sub(r'(\d)([a-z])', r'\1 \2', separated)
        if separated != query.lower():
            return cls._fallback_token_all_match(separated)
        return []
    
    @classmethod
    def _fallback_normalize_generation(cls, query: str) -> List[Tuple[str, int]]:
        """Fix generation typos (2th → 2nd)"""
        query_lower = query.lower()
        typo_fixes = [(r'(\d)th\s*gen', r'\1 gen'), (r'2th', '2nd'), (r'3th', '3rd'), (r'1th', '1st')]
        new_query = query_lower
        for pattern, replacement in typo_fixes:
            new_query = re.sub(pattern, replacement, new_query)
        if new_query != query_lower:
            match = re.search(r'(i[3579]).*?(\d+)', new_query)
            if match:
                return cls._find_word_boundary_matches(f"{match.group(1)}-{match.group(2)}")
        return []
    
    @classmethod
    def get_score(cls, cpu: str, cpu_family: str = "", cpu_generation: Optional[int] = None) -> int:
        """
        Get normalized CPU score (0-1000 scale).
        Uses layered fallback matching.
        """
        cls._load_database()
        
        if not cls._cpu_db or not cpu or cpu == "Unknown":
            return 0
        
        # Build query string
        cpu_lower = cpu.lower()
        has_intel_core = any(x in cpu_lower for x in ['i3', 'i5', 'i7', 'i9'])
        has_model_number = bool(re.search(r'\d{3,}', cpu))
        
        if has_intel_core and not has_model_number and cpu_family and cpu_generation:
            query = f"{cpu_family}-{cpu_generation}"
        else:
            query = cpu
        
        # Try all fallback layers
        matches = cls._find_word_boundary_matches(query)
        if not matches:
            matches = cls._find_substring_matches(query)
        if not matches:
            matches = cls._fallback_dash_to_space(query)
        if not matches:
            matches = cls._fallback_intel_4digit_expand(query)
        if not matches:
            matches = cls._fallback_progressive_removal(query)
        if not matches:
            matches = cls._fallback_token_all_match(query)
        if not matches:
            matches = cls._fallback_prioritized_tokens(query)
        if not matches:
            matches = cls._fallback_separate_attached(query)
        if not matches:
            matches = cls._fallback_normalize_generation(query)
        
        if not matches:
            return 0
        
        # Calculate average and normalize to 0-1000
        # Multiplier increased from 3000 to 4500 for better score spread
        avg_score = sum(s for _, s in matches) / len(matches)
        normalized = int((avg_score / cls._max_score) * 4500)
        return min(normalized, 1000)


# =========================================================================
# GPU SCORING ENGINE
# =========================================================================

class GPUScorer:
    """
    GPU benchmark scoring with tiered dictionary and fallback matching.
    Normalizes scores to 0-1000 scale.
    """
    
    # GPU tier scores (normalized 0-1000)
    GPU_SCORES = {
        # ELITE TIER (950-1000) - Flagship desktop-class
        'RTX 5090': 1000, 'RTX 4090': 1000,
        'RTX 5080': 950, 'RTX 4080': 950,
        
        # HIGH TIER (800-900) - High-end gaming/workstation
        'RTX 5070': 900, 'RTX 4070': 880, 'RTX 3080': 870,
        'RTX 4070 TI': 890, 'RTX 3070 TI': 850, 'RTX 3070': 840,
        'RTX 2080': 820, 'QUADRO RTX 5000': 850,
        
        # MID-HIGH TIER (650-780) - Solid gaming
        'RTX 4060': 780, 'RTX 3060': 720, 'RTX 5060': 750,
        'RTX 4050': 700, 'RTX 2070': 690, 'RTX 2060': 650,
        'RTX A3000': 700, 'RTX A2000': 650, 'RTX A1000': 550,
        
        # MID TIER (450-600) - Entry gaming/prosumer
        'RTX 3050': 550, 'RTX 3050 TI': 580,
        'GTX 1660': 520, 'GTX 1660 TI': 540,
        'QUADRO P2000': 480, 'QUADRO P1000': 420,
        'RTX 2000': 500, 'RTX 3000': 520,
        
        # ENTRY DEDICATED (300-450) - Light gaming
        'GTX 1650': 420, 'GTX 1650 TI': 440,
        'GTX 1050': 350, 'GTX 1050 TI': 370,
        'MX550': 350, 'MX450': 330, 'MX350': 310, 'MX330': 300,
        'MX250': 280, 'MX150': 260, 'MX130': 250,
        'NVIDIA T2000': 450, 'NVIDIA T1000': 400,
        'NVIDIA T600': 350, 'NVIDIA T500': 320,
        
        # INTEGRATED HIGH (150-250) - Good integrated
        'INTEL IRIS XE': 220, 'INTEL IRIS PLUS': 200, 'INTEL IRIS': 190,
        'APPLE GPU': 250,  # M-series integrated is strong
        'RADEON 780M': 230, 'RADEON 680M': 210,
        
        # INTEGRATED LOW (80-150) - Basic integrated
        'INTEL UHD': 120, 'INTEL UHD 620': 110, 'INTEL UHD 630': 130,
        'UHD GRAPHICS 600': 100, 'UHD GRAPHICS': 110,
        'AMD RADEON': 130, 'RADEON GRAPHICS': 120,
        
        # UNKNOWN
        'UNKNOWN': 50,
    }
    
    @classmethod
    def get_score(cls, gpu_name: str, gpu_vram: Optional[int] = None) -> int:
        """
        Get GPU score (0-1000) with VRAM bonus.
        
        Args:
            gpu_name: GPU name string
            gpu_vram: VRAM in GB (optional)
            
        Returns:
            Normalized score 0-1000
        """
        if not gpu_name or gpu_name == "Unknown":
            return 50
        
        gpu_upper = gpu_name.upper().strip()
        
        # Try exact match first
        if gpu_upper in cls.GPU_SCORES:
            base_score = cls.GPU_SCORES[gpu_upper]
        else:
            # Fallback: pattern matching
            base_score = cls._fallback_score(gpu_upper)
        
        # Add VRAM bonus (max +50)
        vram_bonus = 0
        if gpu_vram:
            if gpu_vram >= 12:
                vram_bonus = 50
            elif gpu_vram >= 8:
                vram_bonus = 35
            elif gpu_vram >= 6:
                vram_bonus = 20
            elif gpu_vram >= 4:
                vram_bonus = 10
        
        return min(1000, base_score + vram_bonus)
    
    @classmethod
    def _fallback_score(cls, gpu_upper: str) -> int:
        """Fallback scoring for GPUs not in dictionary"""
        # RTX 50xx series
        if 'RTX 50' in gpu_upper or 'RTX5' in gpu_upper:
            return 850
        # RTX 40xx series
        if 'RTX 40' in gpu_upper or 'RTX4' in gpu_upper:
            return 750
        # RTX 30xx series
        if 'RTX 30' in gpu_upper or 'RTX3' in gpu_upper:
            return 600
        # RTX 20xx series
        if 'RTX 20' in gpu_upper or 'RTX2' in gpu_upper:
            return 500
        # Unknown RTX
        if 'RTX' in gpu_upper:
            return 550
        # GTX 16xx
        if 'GTX 16' in gpu_upper:
            return 450
        # GTX 10xx
        if 'GTX 10' in gpu_upper:
            return 350
        # Unknown GTX
        if 'GTX' in gpu_upper:
            return 400
        # Quadro workstation
        if 'QUADRO' in gpu_upper:
            return 450
        # NVIDIA T-series
        if 'NVIDIA T' in gpu_upper or ' T1' in gpu_upper or ' T2' in gpu_upper:
            return 380
        # MX series
        if 'MX' in gpu_upper:
            return 300
        # AMD dedicated RX
        if 'RADEON RX' in gpu_upper or 'RX ' in gpu_upper:
            return 500
        # AMD integrated with model
        if 'RADEON' in gpu_upper and any(c.isdigit() for c in gpu_upper):
            return 180
        # Intel Iris
        if 'IRIS' in gpu_upper:
            return 200
        # Intel UHD
        if 'UHD' in gpu_upper:
            return 120
        # Apple
        if 'APPLE' in gpu_upper:
            return 250
        # AMD integrated
        if 'AMD' in gpu_upper or 'RADEON' in gpu_upper:
            return 130
        # Intel integrated
        if 'INTEL' in gpu_upper:
            return 100
        
        # Unknown - assume basic integrated
        return 50


# =========================================================================
# RAM SCORING ENGINE
# =========================================================================

class RAMScorer:
    """RAM scoring with diminishing returns."""
    
    # RAM scores with diminishing returns
    RAM_SCORES = {
        0: 0,
        2: 100,
        4: 250,
        6: 350,
        8: 500,
        12: 650,
        16: 780,
        24: 870,
        32: 930,
        48: 970,
        64: 1000,
        128: 1000,
    }
    
    @classmethod
    def get_score(cls, ram_gb: int) -> int:
        """Get RAM score (0-1000) with interpolation."""
        if ram_gb <= 0:
            return 0
        if ram_gb >= 64:
            return 1000
        
        # Find surrounding keys for interpolation
        keys = sorted(cls.RAM_SCORES.keys())
        for i, k in enumerate(keys):
            if ram_gb <= k:
                if i == 0:
                    return cls.RAM_SCORES[k]
                # Linear interpolation
                lower_k, upper_k = keys[i-1], k
                lower_v, upper_v = cls.RAM_SCORES[lower_k], cls.RAM_SCORES[k]
                ratio = (ram_gb - lower_k) / (upper_k - lower_k)
                return int(lower_v + ratio * (upper_v - lower_v))
        
        return 1000


# =========================================================================
# STORAGE SCORING ENGINE
# =========================================================================

class StorageScorer:
    """Storage scoring with logarithmic scale."""
    
    @classmethod
    def get_score(cls, storage_gb: int, is_ssd: bool) -> int:
        """
        Get storage score (0-1000) with logarithmic scaling.
        SSD gets full value, HDD gets 60%.
        """
        import math
        
        if storage_gb <= 0:
            return 0
        
        # Logarithmic scale: 256→500, 512→700, 1024→850, 2048→950
        # Formula: score = 150 * log2(storage_gb / 64)
        base_score = min(1000, int(150 * math.log2(max(1, storage_gb / 64))))
        
        # SSD multiplier
        multiplier = 1.0 if is_ssd else 0.6
        
        return int(base_score * multiplier)


# =========================================================================
# SCREEN SCORING ENGINE
# =========================================================================

class ScreenScorer:
    """Screen scoring combining size, refresh rate, and touchscreen."""
    
    SIZE_SCORES = {
        11: 200, 12: 250, 13: 350, 14: 450, 15: 400, 16: 450, 17: 380, 18: 350,
    }
    
    @classmethod
    def get_score(cls, screen_size: Optional[float], refresh_rate: Optional[int], 
                  is_touchscreen: bool) -> int:
        """Get screen score (0-1000)."""
        score = 0
        
        # Size score (0-450)
        if screen_size:
            size_int = int(round(screen_size))
            score += cls.SIZE_SCORES.get(size_int, 300)
        else:
            score += 250  # Unknown size, assume average
        
        # Refresh rate score (0-500) - widened range for better differentiation
        if refresh_rate:
            if refresh_rate >= 240:
                score += 500
            elif refresh_rate >= 165:
                score += 400
            elif refresh_rate >= 144:
                score += 320
            elif refresh_rate >= 120:
                score += 220
            elif refresh_rate >= 90:
                score += 140
            else:
                score += 50  # 60Hz - lowered from 80
        else:
            score += 50  # Unknown, assume 60Hz
        
        # Touchscreen bonus (0-150)
        if is_touchscreen:
            score += 150
        
        return min(1000, score)


# =========================================================================
# CONDITION SCORING ENGINE
# =========================================================================

class ConditionScorer:
    """Condition scoring combining new/used status and brand premium."""
    
    BRAND_BONUS = {
        # Premium enterprise/consumer
        'Apple': 100, 'Microsoft': 80,
        'Dell': 70, 'HP': 70, 'Lenovo': 70,
        
        # Gaming/prosumer
        'Asus': 55, 'MSI': 55, 'Razer': 60,
        
        # Budget/mainstream
        'Acer': 40, 'Samsung': 50, 'Huawei': 45, 'LG': 45,
        'Gigabyte': 50, 'Toshiba': 35,
        
        # Other
        'Unknown': 20,
    }
    
    @classmethod
    def get_score(cls, is_new: bool, brand: str) -> int:
        """
        Get condition score (0-1000).
        
        Combines:
        - New vs Used: 0-700
        - Brand premium: 0-100
        """
        score = 0
        
        # New/Used component (0-800) - increased gap for more differentiation
        if is_new:
            score += 800
        else:
            score += 100  # Used baseline - lowered from 200 for better spread
        
        # Brand bonus (0-100)
        brand_bonus = cls.BRAND_BONUS.get(brand, 30)
        score += brand_bonus
        
        # Normalize to 0-1000 (max is 700 + 100 = 800, scale up)
        return min(1000, int(score * 1.25))


# =========================================================================
# SMART EXTRACTOR ENGINE
# =========================================================================

@dataclass
class NumberCandidate:
    """A number found in text with its context"""
    value: int
    context: str  # ±20 chars around the number
    unit: Optional[str]  # gb, go, tb, to, g, t, or None
    num_idx: int  # relative start index in context
    num_len: int  # length of number string
    

class SmartExtractor:
    """
    Multi-Candidate Scoring Engine for RAM and Storage extraction.
    
    Instead of single-match regex, this:
    1. Extracts ALL numbers with context
    2. Scores by keyword proximity
    3. Handles slash format specially
    4. Cross-validates RAM/Storage relationship
    """
    
    # Context keywords for classification
    RAM_CONTEXT = [
        'ram', 'mémoire', 'memoire', 'ddr', 'ddr3', 'ddr4', 'ddr5', 
        'lpddr', 'so-dimm', 'sodimm', 'memory'
    ]
    
    STORAGE_CONTEXT = [
        'ssd', 'nvme', 'hdd', 'disque', 'disc', 'disk', 'stockage', 
        'storage', 'pcie', 'm.2', 'm2', 'tera', 'téra', 'rom', 'emmc'
    ]
    
    GPU_CONTEXT = [
        'rtx', 'gtx', 'nvidia', 'radeon', 'rx', 'vram', 'gpu', 
        'graphique', 'graphics', 'geforce', 'quadro', 'mx'
    ]
    
    CPU_CONTEXT = [
        'core', 'intel', 'amd', 'ryzen', 'i3', 'i5', 'i7', 'i9',
        'gen', 'génération', 'generation', 'th', 'eme', 'ème', 'ghz'
    ]
    
    SCREEN_CONTEXT = [
        'pouces', 'pouce', 'inch', '"', 'ecran', 'écran', 'display',
        'hz', 'fhd', 'uhd', 'qhd', '4k', '1080p', '1440p', 'oled', 'ips'
    ]
    
    # Realistic thresholds (soft limits - flag, don't reject)
    RAM_SOFT_MIN = 2
    RAM_SOFT_MAX = 128  # Workstations can have 128GB
    RAM_HARD_MAX = 256  # Absolute max (servers)
    
    STORAGE_SOFT_MIN = 64
    STORAGE_SOFT_MAX = 8000  # 8TB
    STORAGE_HARD_MAX = 16000  # 16TB absolute max
    
    # Valid RAM sizes (relaxed - includes Apple's 36GB, 18GB configs)
    STANDARD_RAM_SIZES = [2, 4, 6, 8, 12, 16, 18, 24, 32, 36, 40, 48, 64, 96, 128]
    
    # Common storage sizes in GB (expanded for older/uncommon drives)
    STANDARD_STORAGE_SIZES = [64, 120, 128, 160, 240, 250, 256, 320, 480, 500, 512, 750, 1000, 1024, 2000, 2048, 3000, 4000, 8000]
    
    # Numbers to exclude (model numbers, screen sizes, years)
    EXCLUDE_PATTERNS = [
        r'\b20[12][0-9]\b',  # Years 2010-2029
        r'\bg\d{1,2}\b',     # G10, G11 model suffixes
        r'\b\d{4,5}[hkupvx]{1,2}\b',  # CPU model numbers like 13700H
        r'\b[hkupvx]\d{4,5}\b',  # Reversed CPU model numbers
    ]
    
    # Years that should never be RAM or Storage
    YEAR_VALUES = list(range(2005, 2030))  # 2005-2029
    
    @classmethod
    def extract_all_numbers(cls, text: str) -> List[NumberCandidate]:
        """
        Find all numbers in text with their context and units.
        Returns list of NumberCandidate objects.
        """
        text_lower = text.lower()
        candidates = []
        
        # Pattern to match numbers with optional unit
        # IMPROVED: Added 'tera', 'téra', 'mg' typo, 'gram', 'ram' suffix, 'sad' typo
        # Pattern captures: 8gb, 16go, 1tb, 1tera, 4gram, 16ram, 512sad, etc.
        number_pattern = r'\b(\d{1,4})\s*(gb|go|tb|to|t[eé]ra|gram|ram|g|t|ssd|nvme|hdd|mg|sad)?(?:\b|(?=\s|$))'
        
        for match in re.finditer(number_pattern, text_lower):
            try:
                value = int(match.group(1))
                if value == 0:
                    continue  # Skip zero
                    
                # Get context window (±20 chars)
                start_ctx = max(0, match.start() - 20)
                end_ctx = min(len(text_lower), match.end() + 20)
                context = text_lower[start_ctx:end_ctx]
                
                # Calculate relative position
                rel_start = match.start() - start_ctx
                match_str = match.group(0)
                
                # Extract unit if present
                unit = match.group(2) if match.group(2) else None
                
                # Skip if matches exclude patterns
                should_exclude = False
                for exclude_pat in cls.EXCLUDE_PATTERNS:
                    if re.search(exclude_pat, context):
                        # Check if this specific number is part of excluded pattern
                        if re.search(rf'\b{value}\b.*?(?:th|eme|gen)', context):
                            should_exclude = True
                            break
                
                if should_exclude:
                    continue
                
                candidates.append(NumberCandidate(
                    value=value,
                    context=context,
                    unit=unit,
                    num_idx=rel_start,
                    num_len=len(str(value)) # Length of just the number part
                ))
                
            except ValueError:
                continue
        
        return candidates
    
    @classmethod
    def _get_proximity_score(cls, context: str, keywords: List[str], num_idx: int, num_len: int) -> float:
        """Calculate max score based on proximity of nearest keyword"""
        max_score = 0.0
        
        for k in keywords:
            try:
                # Use regex with boundaries
                pat = r'\b' + re.escape(k) + r'\b'
                for m in re.finditer(pat, context):
                    k_start, k_end = m.span()
                    n_start, n_end = num_idx, num_idx + num_len
                    
                    # Calculate distance
                    if k_end <= n_start:
                        dist = n_start - k_end
                    elif n_end <= k_start:
                        dist = k_start - n_end
                    else:
                        dist = 0 # overlap (rare)
                    
                    # Distance-based scoring
                    if dist <= 3: score = 6.0    # Adjacent/Very close
                    elif dist <= 12: score = 4.0 # Nearby
                    elif dist <= 25: score = 2.0 # In context
                    else: score = 0.0
                    
                    if score > max_score:
                        max_score = score
            except:
                continue
                
        return max_score

    @classmethod
    def score_for_ram(cls, candidate: NumberCandidate) -> float:
        """Score a number candidate for RAM classification."""
        score = 0.0
        value = candidate.value
        unit = candidate.unit
        
        # 1. Unit scoring
        if unit in ['gb', 'go', 'gram', 'ram']:  # IMPROVED: gram, ram suffix
            score += 3.0
        elif unit in ['g', 'mg']:  # IMPROVED: mg typo for RAM
            score += 2.0
        elif unit in ['tb', 'to', 't', 'tera', 'téra', 'ssd', 'nvme', 'hdd', 'sad']:  # sad typo = storage
            score -= 10.0
        
        # CRITICAL: Exclude screen resolutions (2K, 4K, etc.)
        if unit is None and f"{value}k" in candidate.context.lower():
            score -= 20.0  # Strong penalty for resolution patterns
        
        # 2. Proximity scoring (RAM Context)
        prox_score = cls._get_proximity_score(candidate.context, cls.RAM_CONTEXT, candidate.num_idx, candidate.num_len)
        score += prox_score
        
        # 3. Negative contexts (Storage/GPU/Screen)
        # Check if STORAGE context is CLOSER than RAM context
        storage_score = cls._get_proximity_score(candidate.context, cls.STORAGE_CONTEXT, candidate.num_idx, candidate.num_len)
        if storage_score > prox_score:
            score -= 5.0
            
        gpu_score = cls._get_proximity_score(candidate.context, cls.GPU_CONTEXT, candidate.num_idx, candidate.num_len)
        if gpu_score > 0:
            score -= 5.0
            
        screen_score = cls._get_proximity_score(candidate.context, cls.SCREEN_CONTEXT, candidate.num_idx, candidate.num_len)
        if screen_score > 0:
            score -= 5.0
        
        # 4. Value plausibility
        if value in cls.STANDARD_RAM_SIZES:
            score += 2.0
        elif value < cls.RAM_SOFT_MIN or value > cls.RAM_HARD_MAX:
            score -= 10.0
        elif value > cls.RAM_SOFT_MAX:
            score -= 2.0
            
        # 5. Bonus for common RAM values
        if value in [8, 16, 32]:
            score += 1.0
        
        # 6. CRITICAL: Reject years
        if value in cls.YEAR_VALUES:
            score -= 20.0
        
        return score
    
    @classmethod
    def score_for_storage(cls, candidate: NumberCandidate) -> float:
        """Score a number candidate for Storage classification."""
        score = 0.0
        value = candidate.value
        unit = candidate.unit
        
        # 1. Unit scoring
        if unit in ['tb', 'to', 't', 'tera', 'téra']:  # IMPROVED: French tera
            score += 5.0
        elif unit in ['ssd', 'nvme', 'hdd', 'sad']:  # IMPROVED: sad typo for SSD
            score += 6.0
        elif unit in ['gb', 'go']:
            score += 2.0
        elif unit == 'g':
            score += 1.0
        elif unit in ['mg', 'gram', 'ram']:  # These are RAM indicators, NOT storage
            score -= 10.0
        
        # 2. Proximity scoring (Storage Context)
        prox_score = cls._get_proximity_score(candidate.context, cls.STORAGE_CONTEXT, candidate.num_idx, candidate.num_len)
        score += prox_score
        
        # 3. Negative contexts
        ram_score = cls._get_proximity_score(candidate.context, cls.RAM_CONTEXT, candidate.num_idx, candidate.num_len)
        if ram_score > prox_score:
             score -= 5.0
        
        gpu_score = cls._get_proximity_score(candidate.context, cls.GPU_CONTEXT, candidate.num_idx, candidate.num_len)
        if gpu_score > 0:
            score -= 4.0
            
        cpu_score = cls._get_proximity_score(candidate.context, cls.CPU_CONTEXT, candidate.num_idx, candidate.num_len)
        if cpu_score > 0:
            score -= 3.0
        
        # 4. Value plausibility
        if value in cls.STANDARD_STORAGE_SIZES:
            score += 2.0
        elif unit in ['tb', 'to', 't'] and 1 <= value <= 8:
            score += 3.0
        elif value < cls.STORAGE_SOFT_MIN:
            score -= 5.0
        elif value > cls.STORAGE_HARD_MAX:
            score -= 10.0
            
        # 5. Bonus for common storage values
        if value in [256, 512, 1000, 1024]:
            score += 2.0
        
        # 6. CRITICAL: Reject years
        if value in cls.YEAR_VALUES:
            score -= 20.0
        
        # 7. Reject model numbers (HP EliteBook/ProBook series: 3xx-8xx)
        # These should not be detected as storage
        HP_MODEL_NUMBERS = [310, 320, 330, 340, 350, 360, 370, 380, 390, 
                           410, 420, 430, 440, 450, 460, 470, 480, 490, 
                           510, 520, 530, 540, 550, 560, 570, 580, 590,
                           610, 620, 630, 640, 650, 660, 670, 680, 690, 
                           710, 720, 730, 740, 750, 760, 770, 780, 790, 
                           810, 820, 830, 840, 845, 850, 860, 870, 880,
                           1040, 1030, 1020]  # HP X2 series
        
        # Dell Latitude/Precision series: 3xxx, 5xxx, 7xxx, 9xxx
        DELL_MODEL_NUMBERS = [
            # Latitude 3000 series
            3300, 3310, 3320, 3330, 3340, 3380, 3390,
            3400, 3410, 3420, 3430, 3440, 3480, 3490,
            3500, 3510, 3520, 3530, 3540, 3580, 3590,
            # Latitude 5000 series
            5290, 5300, 5310, 5320, 5330, 5340, 5350,
            5400, 5410, 5420, 5430, 5440, 5450, 5480, 5490, 5491,
            5500, 5510, 5520, 5530, 5540, 5550, 5580, 5590, 5591,
            # Latitude 7000 series
            7280, 7290, 7300, 7310, 7320, 7330, 7340, 7350,
            7380, 7389, 7390, 7400, 7410, 7420, 7430, 7440, 7450,
            7480, 7490, 7520, 7530, 7540,
            # Latitude 9000/Precision series
            9410, 9420, 9430, 9440, 9450, 9510, 9520,
            5530, 5540, 5550, 5560, 5570, 5580,  # Precision mobile
            7530, 7540, 7550, 7560, 7570, 7580,
        ]
        
        # NVMe form factors (should NEVER be storage values)
        NVME_FORM_FACTORS = [2230, 2242, 2260, 2280]
        
        # Combine all model numbers
        ALL_MODEL_NUMBERS = set(HP_MODEL_NUMBERS + DELL_MODEL_NUMBERS + NVME_FORM_FACTORS)
        
        if value in ALL_MODEL_NUMBERS:
            # Strong penalty - these are almost certainly model numbers
            score -= 15.0
        elif 100 <= value < 1000 and value not in cls.STANDARD_STORAGE_SIZES:
            # Generic 3-digit number not in standard sizes - mild penalty
            if prox_score == 0:
                score -= 3.0
        
        return score
    
    @classmethod
    def extract_slash_format(cls, text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Handle special slash format: 32/512 (RAM/Storage)
        Returns (ram_value, storage_value) or (None, None)
        """
        text_lower = text.lower()
        
        # Pattern: number/number where first is small (RAM), second is large (storage)
        slash_pattern = r'\b(\d{1,2})\s*/\s*(\d{3,4})\b'
        
        match = re.search(slash_pattern, text_lower)
        if match:
            first = int(match.group(1))
            second = int(match.group(2))
            
            # Validate: first should be RAM-like, second should be Storage-like
            if first in cls.STANDARD_RAM_SIZES and second >= 128:
                return (first, second)
        
        return (None, None)
    
    @classmethod
    def extract_ram_and_storage(cls, text: str) -> Tuple[str, str]:
        """
        Main extraction method using multi-candidate scoring.
        
        Returns: (ram_string, storage_string) e.g. ("16GB", "512GB SSD")
        """
        text_lower = text.lower()
        
        # 1. Check for slash format first (highest priority)
        slash_ram, slash_storage = cls.extract_slash_format(text)
        if slash_ram is not None and slash_storage is not None:
            # Determine storage type from context
            storage_type = ""
            if 'ssd' in text_lower:
                storage_type = " SSD"
            elif 'nvme' in text_lower:
                storage_type = " NVMe"
            elif 'hdd' in text_lower:
                storage_type = " HDD"
            
            return (f"{slash_ram}GB", f"{slash_storage}GB{storage_type}")
        
        # 2. Extract all number candidates
        candidates = cls.extract_all_numbers(text)
        
        if not candidates:
            return ("Unknown", "Unknown")
        
        # 3. Score each candidate for RAM and Storage
        ram_scores = [(c, cls.score_for_ram(c)) for c in candidates]
        storage_scores = [(c, cls.score_for_storage(c)) for c in candidates]
        
        # Sort by score descending
        ram_scores.sort(key=lambda x: x[1], reverse=True)
        storage_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Pick best candidates with validation
        ram_result = "Unknown"
        storage_result = "Unknown"
        
        # Get best RAM candidate
        for candidate, score in ram_scores:
            if score > 0:  # Must have positive score
                value = candidate.value
                # Validate it's a plausible RAM value
                if value in cls.STANDARD_RAM_SIZES or (cls.RAM_SOFT_MIN <= value <= cls.RAM_SOFT_MAX):
                    ram_result = f"{value}GB"
                    break
        
        # Get best Storage candidate (must be different from RAM if possible)
        ram_value = int(ram_result.replace('GB', '')) if ram_result != "Unknown" else 0
        
        for candidate, score in storage_scores:
            if score > 0:
                value = candidate.value
                unit = candidate.unit
                
                # Handle TB units
                if unit in ['tb', 'to', 't'] and 1 <= value <= 16:
                    storage_result = f"{value}TB"
                    break
                    
                # Handle regular storage
                if value != ram_value:  # Avoid same number for both
                    if candidate.value >= 64:
                        # Determine type from context or unit
                        storage_type = ""
                        if unit in ['ssd', 'nvme']:
                            storage_type = f" {unit.upper()}"
                        elif 'ssd' in candidate.context:
                            storage_type = " SSD"
                        elif 'nvme' in candidate.context:
                            storage_type = " NVMe"
                        elif 'hdd' in candidate.context:
                            storage_type = " HDD"
                        
                        storage_result = f"{value}GB{storage_type}"
                        break
        
        # 5. Cross-field validation
        if ram_result != "Unknown" and storage_result != "Unknown":
            ram_gb = int(ram_result.replace('GB', ''))
            # Parse storage GB (handle TB conversion)
            if 'TB' in storage_result:
                storage_gb = int(storage_result.replace('TB', '').strip()) * 1000
            else:
                storage_gb = int(re.search(r'\d+', storage_result).group())
            
            # RAM should never exceed storage
            if ram_gb > storage_gb:
                # Swap them
                ram_result, storage_result = storage_result.replace('GB', 'GB').replace(' SSD', '').replace(' NVMe', '').replace(' HDD', ''), f"{ram_gb}GB"
                # Re-validate after swap
                try:
                    new_ram = int(ram_result.replace('GB', '').replace('TB', ''))
                    if new_ram not in cls.STANDARD_RAM_SIZES:
                        ram_result = "Unknown"
                except:
                    pass
        
        return (ram_result, storage_result)
# City normalization cache
_city_cache = {}


def normalize_city(raw_city: str, threshold: int = 85) -> str:
    """
    Normalize city name by:
    1. Extracting just the city (removing quarter like "Casablanca, Sidi Moumen")
    2. Fuzzy matching to standard Moroccan city names
    """
    if not raw_city:
        return "Unknown"
    
    if raw_city in _city_cache:
        return _city_cache[raw_city]
    
    city_part = raw_city.split(',')[0].strip()
    city_clean = city_part.lower().strip()
    city_clean = city_clean.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
    city_clean = city_clean.replace('à', 'a').replace('â', 'a')
    city_clean = city_clean.replace('ô', 'o').replace('î', 'i')
    city_clean = city_clean.replace('ù', 'u').replace('û', 'u')
    
    for city in MOROCCAN_CITIES:
        if city.lower() == city_clean:
            _city_cache[raw_city] = city
            return city
    
    result = process.extractOne(
        city_clean,
        [c.lower() for c in MOROCCAN_CITIES],
        scorer=fuzz.ratio
    )
    
    if result and result[1] >= threshold:
        matched_index = [c.lower() for c in MOROCCAN_CITIES].index(result[0])
        matched_city = MOROCCAN_CITIES[matched_index]
        _city_cache[raw_city] = matched_city
        return matched_city
    
    _city_cache[raw_city] = city_part
    return city_part

class SpecParser:
    """
    Extracts laptop specs (CPU, RAM, Storage, GPU, Brand, Model) from text.
    
    Design principles:
    - Specific patterns before generic (order matters)
    - Context awareness via lookahead/lookbehind
    - Normalization to consistent format
    - Return None instead of garbage data
    """
    
    # =========================================================================
    # BRAND PATTERNS (Order: Product lines first, then generic brand names)
    # =========================================================================
    
    BRAND_PRODUCT_LINES = [
        # Apple
        (r'\b(mac\s*boo?c?k|imac)\b', 'Apple'),
        
        # Lenovo
        (r'\b(thinkpad|ideapad|legion|yoga)\b', 'Lenovo'),
        
        # Dell
        (r'\b(latitude|inspiron|xps|precision|vostro|alienware)\b', 'Dell'),
        
        # HP
        (r'\b(elitebook|probook|zbook|omen|pavilion|envy|spectre|victus)\b', 'HP'),
        
        # Asus
        (r'\b(vivobook|zenbook|rog|tuf|expertbook)\b', 'Asus'),
        
        # Microsoft
        (r'\b(surface)\b', 'Microsoft'),
        
        # MSI
        (r'\b(katana|stealth|prestige|raider|pulse|crosshair|vector)\b', 'MSI'),
        
        # Acer
        (r'\b(aspire|nitro|predator|swift|spin)\b', 'Acer'),
        
        # Samsung
        (r'\b(galaxy\s*book)\b', 'Samsung'),
        
        # Huawei
        (r'\b(matebook)\b', 'Huawei'),
        
        # LG (Gram/GARM typo)
        (r'\b(gr?a[mr]m?)\b', 'LG'),
    ]
    
    BRAND_NAMES = [
        (r'\bhp\b', 'HP'),
        (r'\bdell?e?\b', 'Dell'),
        (r'\blenovo\b', 'Lenovo'),
        (r'\basus\b', 'Asus'),
        (r'\baz[uo]s\b', 'Asus'),  # TYPO: Azus, Azos
        (r'\bacer\b', 'Acer'),
        (r'\bmsi\b', 'MSI'),
        (r'\bapple\b', 'Apple'),
        (r'\bsamsung\b', 'Samsung'),
        (r'\bhuawei\b', 'Huawei'),
        (r'\bmicrosoft\b', 'Microsoft'),
        (r'\brazer\b', 'Razer'),
        (r'\blg\b', 'LG'),
        (r'\bgigabyte\b', 'Gigabyte'),
        (r'\bxiaomi\b', 'Xiaomi'),
        (r'\bpanasonic\b', 'Panasonic'),
        (r'\btoshiba\b', 'Toshiba'),
        (r'\bsony\b', 'Sony'),
        # NEW: Additional brands
        (r'\bmedion\b', 'Medion'),
        (r'\bzebra\b', 'Zebra'),
        (r'\bbeelink\b', 'Beelink'),
        (r'\bkuu\b', 'KUU'),
        (r'\bfujitsu\b', 'Fujitsu'),
        (r'\bclevo\b', 'Clevo'),
    ]
    
    # =========================================================================
    # CPU PATTERNS (Order: Specific with model → Generic)
    # =========================================================================
    
    # GPU model numbers to filter out before CPU extraction
    GPU_MODEL_NUMBERS = ['3050', '3060', '3070', '3080', '3090', 
                         '4050', '4060', '4070', '4080', '4090',
                         '5050', '5060', '5070', '5080', '5090',
                         '1650', '1660', '1050', '1060', '1070', '1080',
                         '2060', '2070', '2080', '2000', '3000', '1000']
    
    CPU_PATTERNS = [
        # Apple Silicon (Improved detection)
        # 1. Context-aware: "MacBook...M1" (Highest priority)
        (r'(?:macbook|apple|mac).*?\b(m[1-5])\s*(pro|max|ultra)\b', 'apple_m'),
        (r'(?:macbook|apple|mac).*?\b(m[1-5])\b(?!\s*\.?2)', 'apple_m'),
        
        # 2. Standalone M-series
        (r'\b(m[1-5])\s*(pro|max|ultra)\b(?!\s*(?:ssd|nvme|\.2))', 'apple_m'),
        (r'\b(m[1-5])\b(?!\s*(?:ssd|nvme|\.2|hamid|\'|\.))', 'apple_m'),
        
        # Intel Core Ultra (new naming scheme 2024+)
        # IMPROVED: Added Core 7/Core 9 detection
        (r'\b(?:intel\s*)?(?:core\s*)?(ultra\s*[5792])\s*(\d{3}[hupv]?)?-?(\d{3}[hupv])?\b', 'intel_ultra'),
        (r'\bcore\s*([79])\s+(\d{3}[hupv]?)\b', 'intel_core_numbered'),  # NEW: "Core 7 240H"
        (r'\bu([57])\s*[-:]?\s*(\d{3}[hupv]?)\b', 'intel_u_series'),  # NEW: "U5 238V", "U7-155H"
        
        # Intel with full model number (e.g., i7-13700H, i5-1235U)
        (r'\b(i[3579])[-\s]*(\d{4,5})([hkupg]?)\b', 'intel_full'),
        
        # Intel with generation (e.g., i7 10ème, i5 12th gen)
        (r'\b(i[3579])[-\s]*(\d{1,2})\s*(?:th|ème|eme|gen|éme|ième)\b', 'intel_gen'),
        
        # AMD Ryzen with explicit prefix (e.g., Ryzen 7 5800H) - MUST have Ryzen prefix
        (r'\b(ryzen\s*[3579])\s*(pro\s*)?(\d{4}[uhsx]?)?\b', 'amd_ryzen'),
        
        # IMPROVED: AMD Ryzen with standalone model (5600H, 7435HS)
        (r'\b(?:amd\s*)?ryzen\s*([5-9]\d{3}[uhsx]{1,2})\b', 'amd_ryzen_model'),  # NEW: "Ryzen 5600H"
        (r'\b([5-9]\d{3}[uhsx]{1,2})\b(?=.*(?:amd|ryzen))', 'amd_ryzen_model'),  # NEW: "5600H ... Ryzen"
        
        # AMD Ryzen AI (new 2024+ chips)
        (r'\b(ryzen\s*ai)\s*(\d+)?\b', 'amd_ryzen_ai'),
        
        # AMD Ryzen generic (e.g., Ryzen 5, Ryzen 7)
        (r'\b(ryzen\s*[3579])\s*(pro)?\b', 'amd_ryzen_gen'),
        
        # Qualcomm Snapdragon (for Surface/ARM laptops)
        (r'\b(snapdragon)\s*([x8]\s*\w+)?\b', 'snapdragon'),
        
        # Intel Celeron/Pentium/Xeon
        (r'\b(celeron)\s*([n]?\d{4})?\b', 'intel_low'),
        (r'\b(pentium)\s*(gold|silver)?\s*(\d{4})?\b', 'intel_low'),
        (r'\b(xeon)\s*([ewp])?\s*(-?\d{4})?\b', 'intel_xeon'),
        
        # Intel Core generic (e.g., "Core i7")
        (r'\b(?:intel\s*)?core\s*(i[3579])\b', 'intel_generic'),
        
        # Just i5/i7 standalone
        (r'\b(i[3579])\b', 'intel_simple'),
    ]
    
    # =========================================================================

    
    # =========================================================================
    # GPU PATTERNS
    # =========================================================================
    
    GPU_PATTERNS = [
        # NVIDIA RTX 20/30/40/50 series with optional suffix
        (r'\b(rtx\s*[2345]0[5678]0)\s*(ti|super)?\b', 'nvidia_rtx'),
        
        # NVIDIA Professional T-series (Quadro T500, T550, T600, T1000, T2000)
        (r'\bnvidia\s*t(\d{3,4})\b', 'nvidia_t_series'),
        (r'\bt(\d{3,4})(?:\s*nvidia)?\b', 'nvidia_t_series'),
        
        # NVIDIA RTX mobile (e.g., RTX 3050)
        (r'\b(rtx\s*\d{4})\b', 'nvidia_rtx'),
        
        # NVIDIA GTX
        (r'\b(gtx\s*\d{3,4})\s*(ti)?\b', 'nvidia_gtx'),
        
        # NVIDIA Quadro / RTX Professional
        (r'\b(quadro\s*[a-z]*\s*\d{3,4})\b', 'nvidia_quadro'),
        (r'\b(rtx\s*a\s*\d{3,4})\b', 'nvidia_quadro'),
        
        # NVIDIA MX series (laptop entry-level)
        (r'\b(?:nvidia\s*)?(mx\s*\d{3})\b', 'nvidia_mx'),
        
        # AMD Radeon RX
        (r'\b(rx\s*\d{3,4}[mx]?)\b', 'amd_rx'),
        (r'\b(radeon\s*\d{3}[mx]?)\b', 'amd_radeon'),
        
        # IMPROVED: AMD Radeon generic
        (r'\b(?:amd\s*)?(radeon)\b(?!\s*rx)', 'amd_radeon_generic'),
        
        # Intel integrated
        (r'\b(iris\s*xe)\b', 'intel_iris'),
        (r'\b(intel\s*arc)\b', 'intel_arc'),
        (r'\b(intel\s*uhd\s*\d{3})\b', 'intel_uhd'),
        (r'\b(uhd\s*graphics\s*\d{3})\b', 'intel_uhd'),
    ]
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    @classmethod
    def parse(cls, text: str) -> dict:
        """
        Extract all laptop specs from text.
        
        Uses GPU-first extraction to prevent GPU model numbers from being
        mistakenly detected as CPU values.
        
        Args:
            text: Combined title + description text
            
        Returns:
            Dictionary with brand, model, cpu, ram, storage, gpu
        """
        text_lower = text.lower()
        text_original = text
        
        # STEP 1: Extract GPU first (before any text cleaning)
        gpu = cls.extract_gpu(text_lower)
        
        # STEP 2: Extract Brand (before cleaning, uses product line matching)
        brand = cls.extract_brand(text_lower, text_original)
        
        # STEP 3: Extract Model (before cleaning)
        model, model_detail = cls.extract_model(text_lower, text_original)
        
        # STEP 4: Create cleaned text for CPU extraction
        text_for_cpu = cls._clean_text_for_cpu(text_lower, gpu, model, model_detail)
        
        # STEP 5: Extract RAM and Storage using SmartExtractor (multi-candidate scoring)
        ram, storage = SmartExtractor.extract_ram_and_storage(text)
        
        # STEP 6: Extract feature fields
        screen_size = cls.extract_screen_size(text_lower)
        is_new = cls.extract_is_new(text_lower)
        is_touchscreen = cls.extract_is_touchscreen(text_lower)
        
        # STEP 7: Extract NEW fields (model_num, gpu_vram, refresh_rate)
        model_num = cls.extract_model_num(text, model)
        gpu_vram = cls.extract_gpu_vram(text, gpu)
        refresh_rate = cls.extract_refresh_rate(text_lower)
        
        # STEP 8: Return all extracted specs
        return {
            'brand': brand,
            'model': model,
            'model_detail': model_detail,
            'model_num': model_num,
            'cpu': cls.extract_cpu(text_for_cpu),
            'ram': ram,
            'storage': storage,
            'gpu': gpu,
            'gpu_vram': gpu_vram,
            'screen_size': screen_size,
            'is_new': is_new,
            'is_touchscreen': is_touchscreen,
            'refresh_rate': refresh_rate,
        }
    
    @classmethod
    def _clean_text_for_cpu(cls, text_lower: str, gpu: str, model: str, model_detail: Optional[str] = None) -> str:
        """
        Remove GPU numbers, years, and model numbers from text before CPU extraction.
        This prevents values like '4070', '2024', '7480' from being detected as CPU.
        """
        cleaned = text_lower
        
        # Remove years (2010-2029) - these are NOT CPUs
        cleaned = re.sub(r'\b20[12][0-9]\b', ' ', cleaned)
        
        # Remove GPU model numbers (RTX 4070, GTX 1660, etc.)
        for gpu_num in cls.GPU_MODEL_NUMBERS:
            # Remove the number when it appears near RTX/GTX/Radeon context
            cleaned = re.sub(rf'\b(rtx|gtx|radeon)?\s*{gpu_num}\b', ' ', cleaned)
        
        # Remove detected GPU string entirely
        if gpu and gpu != 'Unknown':
            cleaned = cleaned.replace(gpu.lower(), ' ')
        
        # Remove detected model family (e.g., "ThinkPad")
        if model and model != 'Unknown':
            cleaned = cleaned.replace(model.lower(), ' ')
            
        # Remove detected model detail (e.g., "T14", "7480")
        if model_detail:
            # CRITICAL FIX: Do NOT remove if it looks like an Apple M-series chip
            is_apple_cpu = re.match(r'^m[1-5](?:\s*(?:pro|max|ultra))?$', model_detail.lower())
            if not is_apple_cpu:
                cleaned = cleaned.replace(model_detail.lower(), ' ')
        
        return cleaned
    
    @classmethod
    def extract_brand(cls, text_lower: str, text_original: str) -> str:
        """Extract laptop brand, preferring product line detection"""
        
        # First: Check product lines (more reliable)
        for pattern, brand in cls.BRAND_PRODUCT_LINES:
            if re.search(pattern, text_lower):
                return brand
        
        # Second: Check brand names
        for pattern, brand in cls.BRAND_NAMES:
            if re.search(pattern, text_lower):
                return brand
        
        return "Unknown"
    
    @classmethod
    def extract_model(cls, text_lower: str, text_original: str) -> tuple[str, Optional[str]]:
        """
        Extract laptop PRODUCT LINE AND MODEL NUMBER.
        Returns: (product_line, model_detail)
        Example: "HP EliteBook 840 G5" -> ("EliteBook", "840 G5")
        
        IMPROVED: Now detects HP 250, Laptop 15, V14, V15, X1 Carbon, and more.
        """
        
        # Product line patterns with capture groups for model numbers
        PRODUCT_LINES = [
            # HP: EliteBook 840 G5, ZBook Firefly 14 G8
            (r'\b(elite?book|probook|zbook)\s*(\d{3,4}\s*g\d{1,2}|x360|\d{3,4}\w?)\b', 'HP'),
            (r'\b(pavill?on|envy|spectre|victus|omen|omnibook)\s*(\d{2}|x360|1[3-7])\b', 'HP'),
            
            # IMPROVED: HP numbered models (HP 250, HP 255, HP 15, etc.)
            (r'\bhp\s*(2[0-9]{2})\s*(?:g\d{1,2})?\b', 'HP'),  # HP 250, HP 255
            (r'\bhp\s*laptop\s*(\d{2})\b', 'HP'),  # HP Laptop 15
            
            # Dell: Latitude 5420, XPS 13
            (r'\b(latt?it[u]*de|inspiron|vostro|precision)\s*(\d{3,4})\b', 'Dell'),
            (r'\b(xps)\s*(1[357]|[9753]\d{3})\b', 'Dell'),
            
            # IMPROVED: Dell Pro models
            (r'\b(dell\s*pro)\s*(?:max)?\s*(\d{2})(?:\s*plus)?\b', 'Dell'),  # Dell Pro 14, Dell Pro Max 16 Plus
            # IMPROVED: Dell G-series (G15, G16) gaming laptops
            (r'\b(dell\s*)?g(1[456])\b', 'Dell'),  # Dell G15, G16
            
            # Lenovo: ThinkPad T14, Legion 5
            (r'\b(t[h]?inkpad|ideapad|thinkbook)\s*([a-z]\d{2,3}|x1\s*carbon|x1\s*yoga|x1|z\d{2}|e\d{2,3}|l\d{2,3}|t\d{2,3}[s]?|p\d{2}[s]?)\b', 'Lenovo'),
            (r'\b(legion|loq|yoga)\s*(\d{1}|slim\s*\d|pro\s*\d|s\d{3}|[579]i?)\b', 'Lenovo'),
            
            # IMPROVED: Lenovo V-series
            (r'\blenovo\s*v(\d{2})\b', 'Lenovo'),  # Lenovo V14, V15
            
            # Asus
            (r'\b(vivobook|zenbook)\s*(\d{1,2}[a-z]*|s\s*\d{2}|pro\s*\d{1,2}|flip)\b', 'Asus'),
            (r'\b(rog|tuf)\s*([a-z0-9]+\s*[a-z0-9]*)\b', 'Asus'),
            # IMPROVED: Asus x-series (x515, x540, x555, x556, etc.)
            (r'\basus\s*(x\d{3}[a-z]*)\b', 'Asus'),  # Asus x515, x540
            (r'\b(x\d{3}[a-z]*)\b(?=.*asus)', 'Asus'),  # x515 followed by asus
            
            # Acer
            (r'\b(nitro|predator|aspire|swift|spin)\s*(\d+|v\d+|an\d+)\b', 'Acer'),
            
            # MSI
            (r'\b(katana|stealth|prestige|raider|pulse|crosshai?r|vector|cyborg|thin|modern)\s*([a-z0-9]+)\b', 'MSI'),
            # IMPROVED: MSI GL/GF series
            (r'\bmsi\s*(gl|gf)(\d{2})\b', 'MSI'),  # MSI GL65
            # IMPROVED: MSI SWORD
            (r'\b(sword)\s*(\d{2})\b', 'MSI'),
            
            # Apple
            (r'\b(mac\s*boo?c?k\s*(?:pro|air)?)\s*(m\d\s*(?:pro|max)?|20\d{2})\b', 'Apple'),
            
            # Samsung
            (r'\b(galaxy\s*book)\s*(\d*|pro|ultra|360)\b', 'Samsung'),
            
            # Microsoft
            (r'\b(surface)\s*(pro\s*\d+|laptop\s*\d+|go\s*\d+|book\s*\d+)\b', 'Microsoft'),
            
            # Razer
            (r'\b(razer\s*blade)\s*(\d{2})\b', 'Razer'),  # Razer Blade 14
        ]
        
        # 1. Try detailed extraction
        for pattern, brand_ctx in PRODUCT_LINES:
            match = re.search(pattern, text_lower)
            if match:
                raw_family = match.group(1)
                detail = match.group(2).strip() if len(match.groups()) > 1 else None
                
                # Normalize detail
                detail = detail.title() if detail else None
                
                if raw_family is None:
                    continue
                family = cls._resolve_family_name(raw_family)
                
                # CLEANUP: Remove CPU keywords if captured in model detail
                if detail:
                    detail_lower = detail.lower()
                    cpu_keywords = ['ryzen', 'core', 'intel', 'amd', 'i3', 'i5', 'i7', 'i9', 'rtx', 'gtx']
                    for kw in cpu_keywords:
                        pattern_kw = r'\b' + kw + r'\b'
                        if re.search(pattern_kw, detail_lower):
                            # Truncate at the keyword
                            match = re.search(pattern_kw, detail_lower)
                            if match:
                                detail = detail[:match.start()].strip()
                                
                    if not detail:
                        detail = None
                        
                return family, detail

        # 2. Fallback: Simple Product Line Extraction (No detail)
        SIMPLE_LINES = [
            (r'\bt[h]?inkpad\b', 'ThinkPad'),
            (r'\bideapad\b', 'IdeaPad'),
            (r'\blegion\b', 'Legion'),
            (r'\byoga\b', 'Yoga'),
            (r'\bthinkbook\b', 'ThinkBook'),
            (r'\bloq\b', 'LOQ'),
            (r'\blatt?it[u]*de\b', 'Latitude'),
            (r'\binspiron\b', 'Inspiron'),
            (r'\bxps\b', 'XPS'),
            (r'\bpr[ei]?cision\b', 'Precision'),
            (r'\bvostro\b', 'Vostro'),
            (r'\balienware\b', 'Alienware'),
            (r'\bdell\s*pro\b', 'Dell Pro'),  # NEW: Dell Pro
            (r'\belite?book\b', 'EliteBook'),
            (r'\bprobook\b', 'ProBook'),
            (r'\bhp\s*pro\b', 'ProBook'),
            (r'\bzbook\b', 'ZBook'),
            (r'\bpavill?on\b', 'Pavilion'),
            (r'\benvy\b', 'Envy'),
            (r'\bspectre\b', 'Spectre'),
            (r'\bvictus\b', 'Victus'),
            (r'\bomen\b', 'Omen'),
            (r'\bomnibook\b', 'OmniBook'),
            (r'\bvivobook\b', 'VivoBook'),
            (r'\bzenbook\b', 'ZenBook'),
            (r'\brog\b', 'ROG'),
            (r'\btuf\b', 'TUF'),
            (r'\bexpertbook\b', 'ExpertBook'),
            (r'\bsurface\b', 'Surface'),
            (r'\baspire\b', 'Aspire'),
            (r'\bnitro\b', 'Nitro'),
            (r'\bpredator\b', 'Predator'),
            (r'\bswift\b', 'Swift'),
            (r'\bkatana\b', 'Katana'),
            (r'\bsword\b', 'Sword'),
            (r'\bcyborg\b', 'Cyborg'),
            (r'\btitan\b', 'Titan'),  # NEW: MSI Titan
            (r'\braider\b', 'Raider'),  # NEW: MSI Raider
            (r'\bpulse\b', 'Pulse'),  # NEW: MSI Pulse
            (r'\bstealth\b', 'Stealth'),  # NEW: MSI Stealth
            (r'\bcrosshair\b', 'Crosshair'),  # NEW: MSI Crosshair
            (r'\bvector\b', 'Vector'),  # NEW: MSI Vector
            (r'\bx1\s*carbon\b', 'X1 Carbon'),  # NEW: Lenovo X1 Carbon
            (r'\bx360\b', 'x360'),  # NEW: HP x360 series
            (r'\be1[456]\b', 'ThinkPad'),  # NEW: Lenovo E14/E15/E16
            (r'\bmac\s*boo?c?k\s*pro\b', 'MacBook Pro'),
            (r'\bmac\s*boo?c?k\s*air\b', 'MacBook Air'),
            (r'\bmac\s*boo?c?k\b', 'MacBook'),
            (r'\bgalaxy\s*book\b', 'Galaxy Book'),
            (r'\bmatebook\b', 'MateBook'),
            (r'\blifebook\b', 'LifeBook'),
            (r'\bvaio\b', 'Vaio'),
            (r'\bgr?a[mr]m?\b', 'Gram'),  # IMPROVED: LG Gram with GARM typo
        ]
        
        for pattern, family in SIMPLE_LINES:
            if re.search(pattern, text_lower):
                return family, None
        
        return "Unknown", None

    @classmethod
    def _resolve_family_name(cls, raw: str) -> str:
        """Map raw matched family text (with typos) to canonical name"""
        r = raw.lower()
        if 'think' in r: return 'ThinkPad' if 'pad' in r else 'ThinkBook'
        if 'tink' in r: return 'ThinkPad'
        if 'idea' in r: return 'IdeaPad'
        if 'elite' in r: return 'EliteBook'
        if 'elit' in r: return 'EliteBook'
        if 'pro' in r and 'book' in r: return 'ProBook'
        if 'zbook' in r: return 'ZBook'
        if 'lat' in r: return 'Latitude'
        if 'insp' in r: return 'Inspiron'
        if 'prec' in r: return 'Precision'
        if 'dell' in r and 'pro' in r: return 'Dell Pro' 
        if 'mac' in r:
            if 'pro' in r: return 'MacBook Pro'
            if 'air' in r: return 'MacBook Air'
            return 'MacBook'
        if 'gal' in r: return 'Galaxy Book'
        if 'viv' in r: return 'VivoBook'
        if 'zen' in r: return 'ZenBook'
        if 'pav' in r: return 'Pavilion'
        if 'spec' in r: return 'Spectre'
        if 'sword' in r: return 'Sword'  # NEW
        if 'cyborg' in r: return 'Cyborg'  # NEW
        
        # NEW: Handle numbered models
        if r.isdigit():
            return f"Model {r}"
        
        return raw.title()
    
    @classmethod
    def extract_cpu(cls, text_lower: str) -> str:
        """Extract CPU with normalization - IMPROVED"""
        
        for pattern, cpu_type in cls.CPU_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return cls._normalize_cpu(match, cpu_type)
        
        return "Unknown"
    
    @classmethod
    def _normalize_cpu(cls, match: re.Match, cpu_type: str) -> str:
        """Normalize CPU to consistent format with proper casing - IMPROVED"""
        
        groups = match.groups()
        raw = match.group(0).strip()
        
        if cpu_type == 'apple_m':
            # M1/M2/M3/M4 with optional Pro/Max/Ultra
            base = groups[0].upper() if groups else raw.upper()
            suffix = groups[1].capitalize() if len(groups) > 1 and groups[1] else ""
            return f"{base} {suffix}".strip()
        
        elif cpu_type == 'intel_ultra':
            # Intel Core Ultra 5/7/9 -> "Core Ultra 5"
            result = raw.lower()
            result = result.replace('intel', '').strip()
            result = re.sub(r'(ultra)(\d)', r'\1 \2', result)
            return cls._title_case_cpu(result)
        
        elif cpu_type == 'intel_core_numbered':
            # NEW: "Core 7 240H" -> "Core 7-240H"
            tier = groups[0]
            model = groups[1].upper()
            return f"Core {tier}-{model}"
        
        elif cpu_type == 'intel_u_series':
            # NEW: "U5 238V" -> "Core Ultra 5-238V"
            tier = groups[0]
            model = groups[1].upper() if groups[1] else ""
            return f"Core Ultra {tier}-{model}".strip()
        
        elif cpu_type == 'intel_full':
            # i7-13700H format -> "Core i7-13700H"
            base = groups[0].lower()
            model = groups[1]
            suffix = groups[2].upper() if groups[2] else ""
            return f"Core {base}-{model}{suffix}"
        
        elif cpu_type == 'intel_gen':
            # i7 10th gen format -> "Core i7 10th Gen"
            base = groups[0].lower()
            gen = groups[1]
            return f"Core {base} {gen}th Gen"
        
        elif cpu_type == 'amd_ryzen':
            # Ryzen 7 5800H -> "Ryzen 7 5800H"
            result = raw.lower().replace("  ", " ")
            parts = result.split()
            normalized = []
            for part in parts:
                if part == 'ryzen':
                    normalized.append('Ryzen')
                elif part == 'pro':
                    normalized.append('Pro')
                elif part.isdigit() and len(part) == 1:
                    normalized.append(part)
                else:
                    normalized.append(part.upper())
            return ' '.join(normalized)
        
        elif cpu_type == 'amd_ryzen_model':
            # NEW: "5600H" -> "Ryzen 5 5600H"
            model = groups[0].upper()
            tier = model[0]  # First digit is the tier (5, 7, 9)
            return f"Ryzen {tier} {model}"
        
        elif cpu_type == 'amd_ryzen_gen':
            # Ryzen 5, Ryzen 7 generic
            result = raw.lower().replace("  ", " ")
            return result.replace('ryzen', 'Ryzen').replace('pro', 'Pro')
        
        elif cpu_type == 'amd_ryzen_ai':
            # AMD Ryzen AI chips -> "Ryzen AI 5"
            result = raw.lower().replace("  ", " ")
            return result.replace('ryzen', 'Ryzen').replace('ai', 'AI')
        
        elif cpu_type in ['intel_generic', 'intel_simple']:
            # Core i7, i5 -> "Core i7"
            base = groups[0].lower() if groups else raw.lower()
            return f"Core {base}"
        
        elif cpu_type == 'snapdragon':
            # Qualcomm Snapdragon -> "Snapdragon X Plus"
            result = raw.lower()
            parts = result.split()
            normalized = [parts[0].capitalize()]
            normalized.extend(p.upper() if len(p) <= 2 else p.capitalize() for p in parts[1:])
            return ' '.join(normalized)
        
        elif cpu_type in ['intel_low', 'intel_xeon']:
            # Celeron, Pentium, Xeon
            return raw.capitalize()
        
        else:
            return cls._title_case_cpu(raw)
    
    @classmethod
    def _title_case_cpu(cls, cpu: str) -> str:
        """Apply proper title case to CPU string, removing redundant words."""
        result = cpu.lower()
        
        # Remove redundant patterns
        result = re.sub(r'\bcore\s+core\b', 'core', result)
        result = re.sub(r'\bcore\s+intel\s+core\b', 'core', result)
        result = re.sub(r'\bintel\s+core\b', 'core', result)
        
        # Title case specific words
        replacements = {
            'core': 'Core',
            'ultra': 'Ultra',
            'intel': 'Intel',
            'ryzen': 'Ryzen',
            'pro': 'Pro',
            'gen': 'Gen',
        }
        for old, new in replacements.items():
            result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
        
        # Fix spacing issues
        result = re.sub(r'(ultra)(\d)', r'\1 \2', result, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    @classmethod
    def extract_ram(cls, text_lower: str) -> str:
        """Delegate to SmartExtractor"""
        return SmartExtractor.extract_ram(text_lower)
    
    @classmethod
    def extract_storage(cls, text_lower: str) -> str:
        """Delegate to SmartExtractor"""
        return SmartExtractor.extract_storage(text_lower)
    
    @classmethod
    def extract_gpu(cls, text_lower: str) -> str:
        """Extract GPU model - IMPROVED"""
        
        for pattern, gpu_type in cls.GPU_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                raw = match.group(0).upper().strip()
                
                # Clean up spacing
                raw = re.sub(r'\s+', ' ', raw)
                
                # Add suffix if present
                if len(match.groups()) > 1 and match.groups()[1]:
                    suffix = match.groups()[1].upper()
                    if suffix not in raw:
                        raw = f"{raw} {suffix}"
                
                # Custom normalization
                if gpu_type == 'nvidia_t_series':
                    model_match = re.search(r't(\d{3,4})', raw.lower())
                    if model_match:
                         return f"NVIDIA T{model_match.group(1)}"
                
                if gpu_type == 'amd_radeon_generic':
                    return "AMD Radeon"
                
                return raw
        
        return "Unknown"
    
    @classmethod
    def extract_screen_size(cls, text_lower: str) -> Optional[float]:
        """Extract screen size in inches"""
        patterns = [
            r'(\d{2})\s*(?:pouces|pouce|inch|inches|"|'')',  # 15 pouces, 14"
            r'(\d{2})\s*(?:p\b)',  # 15p, 14p
            r'ecran\s*(\d{2})',  # ecran 15
        ]
        for pat in patterns:
            match = re.search(pat, text_lower)
            if match:
                size = float(match.group(1))
                # Valid laptop screen sizes: 11-18 inches
                if 11 <= size <= 18:
                    return size
        return None
    
    @classmethod
    def extract_is_new(cls, text_lower: str) -> bool:
        """Detect if laptop is new (neuf) or used (occasion)"""
        new_patterns = [
            r'\bneuf\b', r'\bnew\b', r'\bscell[eé]\b', 
            r'\bnon\s*ouvert\b', r'\bboite\s*ferm[eé]e?\b',
            r'\bneuve\b', r'\bnouveau\b'
        ]
        # Check for new indicators
        for pat in new_patterns:
            if re.search(pat, text_lower):
                return True
        return False
    
    @classmethod
    def extract_is_touchscreen(cls, text_lower: str) -> bool:
        """Detect touchscreen capability"""
        touch_patterns = [
            r'\btactile\b', r'\btouch\s*screen\b', r'\btouch\b',
            r'\bx360\b', r'\bflip\b', r'\b2\s*(?:in|en)\s*1\b',
            r'\bconvertible\b'
        ]
        for pat in touch_patterns:
            if re.search(pat, text_lower):
                return True
        return False
    
    # =========================================================================
    # NEW EXTRACTION METHODS
    # =========================================================================
    
    # Blacklist: terms that should NOT be captured as model numbers
    MODEL_NUM_BLACKLIST = {
        'gb', 'go', 'tb', 'to', 'ssd', 'hdd', 'nvme', 'ram', 'ddr', 'ddr4', 'ddr5',
        'i3', 'i5', 'i7', 'i9', 'ryzen', 'core', 'intel', 'amd', 'nvidia', 'ultra',
        'rtx', 'gtx', 'mx', 'quadro', 'radeon', 'ghz', 'mhz', 'hz',
        'neuf', 'new', 'occasion', 'gaming', 'pro', 'plus', 'max', 'mini',
        'gen', 'eme', 'ème', 'th', 'nd', 'rd', 'st', 'pouces', 'pouce', 'inch',
        'tactile', 'touch', 'oled', 'led', 'ips', 'fhd', 'qhd', 'uhd', '4k', '2k',
        'windows', 'win', 'linux', 'mac', 'os', 'usb', 'hdmi', 'wifi', 'bluetooth',
        'm1', 'm2', 'm3', 'm4', 'm5', 'air', 'celeron', 'pentium', 'xeon', 'athlon',
    }
    
    @classmethod
    def extract_model_num(cls, text: str, model_family: str) -> Optional[str]:
        """
        Extract model NUMBER after family name (e.g., "840 G8" from "EliteBook 840 G8")
        
        Args:
            text: Original text (title + description)
            model_family: Already extracted family (e.g., "EliteBook", "Latitude")
        
        Returns:
            Model number string or None
        """
        if model_family == "Unknown" or not model_family:
            return None
        
        text_lower = text.lower()
        family_lower = model_family.lower()
        
        # Find position of family in text
        family_pos = text_lower.find(family_lower)
        if family_pos == -1:
            # Try common variations
            for alt in [family_lower.replace(' ', ''), family_lower[:4]]:
                pos = text_lower.find(alt)
                if pos != -1:
                    family_pos = pos
                    break
        
        if family_pos == -1:
            return None
        
        # Get text after family name (up to 30 chars)
        after_family = text[family_pos + len(family_lower):family_pos + len(family_lower) + 35]
        
        # Pattern 1: Alphanumeric model with optional generation (840 G8, T14 Gen 3, 7290)
        patterns = [
            # "840 G8", "T14 G3", "E14 Gen 3"
            r'^\s*([A-Z]?\d{2,4})\s*(G\d+|Gen\s*\d+)?',
            # "X1 Carbon Gen 11" - already captured X1 Carbon, get Gen 11
            r'^\s*(Gen\s*\d+|G\d+)',
            # Just numbers: "7290", "3420"
            r'^\s*(\d{3,4})\b',
            # Letter + numbers: "T14", "E15", "X540"
            r'^\s*([A-Z]\d{2,3}[A-Z]?)\b',
        ]
        
        for pat in patterns:
            match = re.search(pat, after_family, re.IGNORECASE)
            if match:
                # Combine all captured groups
                parts = [g for g in match.groups() if g]
                model_num = ' '.join(parts).strip()
                
                # Validate: not in blacklist
                tokens = model_num.lower().split()
                if all(t not in cls.MODEL_NUM_BLACKLIST for t in tokens):
                    # Additional validation: should have at least one digit
                    if re.search(r'\d', model_num):
                        return model_num
        
        return None
    
    @classmethod
    def extract_gpu_vram(cls, text: str, gpu: str) -> Optional[int]:
        """
        Extract GPU VRAM in GB.
        
        Args:
            text: Original text
            gpu: Already extracted GPU name
            
        Returns:
            VRAM in GB or None
        """
        if gpu == "Unknown" or not gpu:
            return None
        
        # Find GPU mention in text
        gpu_lower = gpu.lower()[:8]  # First 8 chars for matching
        text_lower = text.lower()
        gpu_pos = text_lower.find(gpu_lower)
        
        if gpu_pos == -1:
            # Try RTX/GTX pattern
            if 'rtx' in gpu_lower or 'gtx' in gpu_lower:
                gpu_pos = text_lower.find('rtx') if 'rtx' in gpu_lower else text_lower.find('gtx')
        
        if gpu_pos == -1:
            return None
        
        # Look in window around GPU mention (30 chars before and after)
        start = max(0, gpu_pos - 10)
        end = min(len(text), gpu_pos + 40)
        nearby = text[start:end]
        
        # Pattern: number + GB/Go (VRAM is typically 2-24GB)
        match = re.search(r'(\d{1,2})\s*(?:GB|Go)\b', nearby, re.IGNORECASE)
        if match:
            vram = int(match.group(1))
            # Valid VRAM range: 2-24GB (laptop dedicated GPUs)
            if 2 <= vram <= 24:
                return vram
        
        return None
    
    @classmethod
    def extract_refresh_rate(cls, text_lower: str) -> Optional[int]:
        """
        Extract screen refresh rate in Hz.
        
        Returns:
            Refresh rate (60-360) or None
        """
        match = re.search(r'(\d{2,3})\s*Hz\b', text_lower, re.IGNORECASE)
        if match:
            hz = int(match.group(1))
            # Valid laptop refresh rates: 60-360Hz
            if 60 <= hz <= 360:
                return hz
        return None

class LaptopListing(BaseModel):
    """Validated laptop listing with all extracted fields"""
    
    title: str
    description: str = ""  # Raw description from listing
    price: float
    city: str
    brand: str = "Unknown"
    model: str = "Unknown"
    model_detail: Optional[str] = None
    model_num: Optional[str] = None  # NEW: Model number (e.g., "840 G8", "7290")
    cpu: str = "Unknown"
    ram: str = "Unknown"
    storage: str = "Unknown"
    gpu: str = "Unknown"
    gpu_vram: Optional[int] = None  # NEW: GPU VRAM in GB
    is_shop: bool = False
    has_delivery: bool = False
    link: str
    
    # Feature fields
    screen_size: Optional[float] = None  # in inches
    is_new: bool = False  # neuf, new, scellé
    is_touchscreen: bool = False  # tactile, touch, x360, flip
    refresh_rate: Optional[int] = None  # NEW: Screen refresh rate in Hz
    
    # ML Scoring Fields
    cpu_score: int = 0
    gpu_score: int = 0
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        """Ensure price is within valid range (allow 0 for negotiable)"""
        # Allow 0 for "Negotiable" / "Contact us"
        if v == 0:
            return v
            
        if not MIN_PRICE <= v <= MAX_PRICE:
            # But the user said: "determine the root cause of listings being flagged as Invalid (price)... e.g zero prices"
            # User instruction: "extract also items with price = 0"
             raise ValueError(f'Price {v} out of valid range ({MIN_PRICE}-{MAX_PRICE})')
        return v
    
    @field_validator('city')
    @classmethod
    def normalize_city_field(cls, v):
        """Normalize city name using fuzzy matching"""
        return normalize_city(v)
    
    @field_validator('title')
    @classmethod
    def clean_title(cls, v):
        """Ensure title is reasonable length"""
        if len(v) > 300:
            return v[:300]
        return v
    
    @model_validator(mode='after')
    def cross_validate(self):
        """Cross-field validation for consistency"""
        # Apple laptops shouldn't have AMD Ryzen CPU
        if self.brand == 'Apple':
            if 'ryzen' in self.cpu.lower():
                self.cpu = "Unknown"
        
        # Non-Apple with M-series is wrong
        if self.brand not in ['Apple', 'Unknown']:
            if self.cpu.startswith('M') and self.cpu[1:2].isdigit():
                self.cpu = "Unknown"
        
        return self
    
    def completeness_score(self) -> int:
        """Count how many fields are NOT 'Unknown'"""
        spec_fields = ['brand', 'cpu', 'ram', 'storage', 'gpu']
        return sum(1 for f in spec_fields if getattr(self, f) != "Unknown")
    
    def is_complete_enough(self) -> bool:
        """Check if listing has enough data to keep"""
        return self.completeness_score() >= MIN_COMPLETE_FIELDS
    
    # Feature engineering properties
    
    @property
    def cpu_family(self) -> str:
        """Extract CPU family for ML features"""
        cpu = self.cpu
        if cpu == "Unknown": return "Unknown"
        cpu_lower = cpu.lower()
        
        if cpu.startswith('M') and len(cpu) >= 2 and cpu[1].isdigit(): return cpu.split()[0]
        match = re.search(r'core\s*(i[3579])', cpu_lower)
        if match: return f"Core {match.group(1)}"
        if 'ultra' in cpu_lower: return "Core Ultra"
        match = re.search(r'ryzen\s*(\d)', cpu_lower)
        if match: return f"Ryzen {match.group(1)}"
        return "Other"



    @property
    def cpu_generation(self) -> Optional[int]:
        """Extract CPU generation number"""
        cpu = self.cpu
        cpu_lower = cpu.lower() if cpu != "Unknown" else ""
        if cpu != "Unknown":
            if cpu.startswith('M') and len(cpu) >= 2 and cpu[1].isdigit(): return int(cpu[1])
            match = re.search(r'i[3579]-(1[0-4])(\d{3})', cpu_lower)
            if match: return int(match.group(1))
            match = re.search(r'i[3579]-([6-9])(\d{3})', cpu_lower)
            if match: return int(match.group(1))
            match = re.search(r'ryzen\s*\d\s*(?:pro\s*)?(\d)\d{3}', cpu_lower)
            if match: return int(match.group(1))
        
        # Check title
        combined = (cpu_lower + " " + self.title.lower()).replace('è','e').replace('é','e').replace('ème','e')
        match = re.search(r'\b(\d{1,2})\s*(?:th|nd|rd|st|eme|e|er)(?:gen)?\b', combined)
        if match:
            gen = int(match.group(1))
            if 4 <= gen <= 14: return gen
        return None
    
    @property
    def gpu_type(self) -> str:
        """Classify GPU as 'integrated', 'dedicated', or 'unknown'"""
        gpu = self.gpu.lower()
        if gpu == "Unknown": return "unknown"
        if any(x in gpu for x in ['rtx', 'gtx', 'quadro', 'rx ', 'radeon rx', 'mx', 't1000', 't2000']): return "dedicated"
        if any(x in gpu for x in ['intel', 'uhd', 'iris', 'vega', 'graphics', 'apple']): return "integrated"
        return "unknown"


    
    @property
    def gpu_family(self) -> str:
        """Extract GPU family"""
        g = self.gpu.lower()
        if "rtx" in g: return "RTX"
        if "gtx" in g: return "GTX"
        if "apple" in g: return "Apple"
        if "intel" in g or "uhd" in g or "iris" in g: return "Intel"
        if "amd" in g or "radeon" in g or "rx" in g: return "AMD"
        return "Other"
    
    @property
    def is_ssd(self) -> bool:
        """Check if storage is SSD"""
        return 'ssd' in self.storage.lower() or 'nvme' in self.storage.lower()
    
    @property
    def ram_gb(self) -> int:
        """Extract RAM size as integer GB"""
        if self.ram == "Unknown": return 0
        match = re.search(r'(\d+)', self.ram)
        return int(match.group(1)) if match else 0
    
    @property
    def storage_gb(self) -> int:
        """Extract storage size as integer GB"""
        if self.storage == "Unknown": return 0
        s = self.storage.lower()
        match = re.search(r'(\d+)\s*tb', s)
        if match: return int(match.group(1)) * 1000
        match = re.search(r'(\d+)', s)
        return int(match.group(1)) if match else 0
        
    def rederive_derived_fields(self):
        """
        Manually trigger re-derivation of fields that might depend on updated raw data.
        Since these are properties in the class, we don't strictly *store* them, 
        but if we export to dict/csv, we want to ensure the logic runs on the *new* raw values.
        
        This method is actually a placeholder because Pydantic models with @property
        will automatically recalculate when accessed.
        
        However, for the ML features stored in the dict (if any), 
        we rely on accessing the properties (gpu_type, cpu_family, etc.) 
        when converting to dict for DataFrame.
        """
        pass
    
    # ==================== SCORING PROPERTIES (NEW WEIGHTED SYSTEM) ====================
    
    # Weights for combined score (total = 100%)
    SCORE_WEIGHTS: ClassVar[Dict[str, float]] = {
        'cpu': 0.35,       # 35% - most valuable, non-upgradable
        'gpu': 0.25,       # 25% - major differentiator
        'ram': 0.12,       # 12% - important but upgradable
        'storage': 0.08,   # 8% - easily upgradable
        'screen': 0.10,    # 10% - size, refresh, touch
        'condition': 0.10, # 10% - new/used + brand
    }
    
    @property
    def computed_cpu_score(self) -> int:
        """Get normalized CPU score (0-1000) using CPUScorer."""
        return CPUScorer.get_score(self.cpu, self.cpu_family, self.cpu_generation)
    
    @property
    def computed_gpu_score(self) -> int:
        """Get normalized GPU score (0-1000) using GPUScorer with VRAM bonus."""
        return GPUScorer.get_score(self.gpu, self.gpu_vram)
    
    @property
    def ram_score(self) -> int:
        """Get RAM score (0-1000) with diminishing returns."""
        return RAMScorer.get_score(self.ram_gb)
    
    @property
    def storage_score(self) -> int:
        """Get storage score (0-1000) with logarithmic scaling."""
        return StorageScorer.get_score(self.storage_gb, self.is_ssd)
    
    @property
    def screen_score(self) -> int:
        """Get screen score (0-1000) combining size, refresh rate, touchscreen."""
        return ScreenScorer.get_score(self.screen_size, self.refresh_rate, self.is_touchscreen)
    
    @property
    def condition_score(self) -> int:
        """Get condition score (0-1000) combining new/used and brand."""
        return ConditionScorer.get_score(self.is_new, self.brand)
    
    @property
    def laptop_score(self) -> int:
        """
        Combined weighted laptop score (0-1000).
        
        Formula:
        - CPU: 35% weight (max 350 points)
        - GPU: 25% weight (max 250 points)
        - RAM: 12% weight (max 120 points)
        - Storage: 8% weight (max 80 points)
        - Screen: 10% weight (max 100 points)
        - Condition: 10% weight (max 100 points)
        """
        weighted = (
            self.computed_cpu_score * self.SCORE_WEIGHTS['cpu'] +
            self.computed_gpu_score * self.SCORE_WEIGHTS['gpu'] +
            self.ram_score * self.SCORE_WEIGHTS['ram'] +
            self.storage_score * self.SCORE_WEIGHTS['storage'] +
            self.screen_score * self.SCORE_WEIGHTS['screen'] +
            self.condition_score * self.SCORE_WEIGHTS['condition']
        )
        return int(min(1000, weighted))

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export with feature engineering and scores"""
        return {
            'title': self.title,
            'description': self.description,
            'price': self.price,
            'city': self.city,
            'brand': self.brand,
            'model': self.model,
            'model_num': self.model_num,
            'cpu': self.cpu,
            'ram': self.ram,
            'storage': self.storage,
            'gpu': self.gpu,
            'gpu_vram': self.gpu_vram,
            'is_shop': self.is_shop,
            'has_delivery': self.has_delivery,
            'link': self.link,
            # Feature fields
            'screen_size': self.screen_size,
            'is_new': self.is_new,
            'is_touchscreen': self.is_touchscreen,
            'refresh_rate': self.refresh_rate,
            # Derived Features
            'cpu_family': self.cpu_family,
            'cpu_generation': self.cpu_generation,
            'gpu_type': self.gpu_type,
            'gpu_family': self.gpu_family,
            'is_ssd': self.is_ssd,
            'ram_gb': self.ram_gb,
            'storage_gb': self.storage_gb,
            # Scores (0-1000 scale, weighted for laptop_score)
            'cpu_score': self.computed_cpu_score,
            'gpu_score': self.computed_gpu_score,  # NEW
            'ram_score': self.ram_score,
            'storage_score': self.storage_score,
            'screen_score': self.screen_score,  # NEW
            'condition_score': self.condition_score,  # NEW
            'laptop_score': self.laptop_score,  # Weighted combined
        }

