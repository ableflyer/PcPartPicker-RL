import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import gymnasium as gym
from gymnasium import spaces
import re
from preprocess import load_and_preprocess_all_data
from enhanced_reward import calculate_total_reward, calculate_enhanced_cpu_reward, calculate_enhanced_gpu_reward

# --- Check for CUDA availability ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- File Paths for CSV Data ---
CSV_FILE_PATHS = {
    "Case": "data/cases_detailed.csv",
    "CPU Cooler": "data/cpu_coolers_detailed.csv",
    "CPU": "data/cpus_detailed.csv",
    "GPU": "data/gpus_detailed.csv",
    "RAM": "data/memory_detailed.csv",
    "Motherboard": "data/motherboards_detailed.csv",
    "PSU": "data/power_supplies_detailed.csv",
    "Storage": "data/storage_detailed.csv",
}

# --- Component Priority Order (Critical components first) ---
COMPONENT_PRIORITY = [
    "GPU",       # Prioritize GPU selection
    "CPU",       # Then CPU
    "Motherboard", # Required for compatibility with CPU
    "RAM",       # Important for system performance
    "Storage",   # Required for system functionality
    "PSU",       # Critical for system stability
    "Case",      # Required for housing components
    "CPU Cooler" # Optional for some CPUs
]

# --- Essential Components ---
ESSENTIAL_COMPONENTS = ["CPU", "Motherboard", "RAM", "GPU", "Storage", "PSU", "Case"]

# --- Performance Critical Components ---
PERFORMANCE_CRITICAL = ["CPU", "GPU", "RAM"]

# --- Modern CPU Generations ---
MODERN_CPU_GENERATIONS = {
    "Intel": [10, 11, 12, 13, 14],  # 10th Gen and newer
    "AMD": [3, 5, 7, 9]             # Ryzen 3000 series and newer
}

# --- Premium CPU Models (for high-budget builds) ---
PREMIUM_CPU_MODELS = {
    "Intel": [
        r"i9-\d{4,}", r"i9 \d{4,}",  # All i9 series
        r"i7-1[2-9]\d{3}", r"i7 1[2-9]\d{3}",  # i7 12th gen and newer
        r"i7-\d{4,}k", r"i7 \d{4,}k",  # K-series i7
    ],
    "AMD": [
        r"ryzen 9 \d{4,}", r"ryzen 9",  # All Ryzen 9
        r"ryzen 7 [5-9]\d{3}", r"ryzen 7 [5-9]",  # Ryzen 7 5000 series and newer
        r"threadripper"  # All Threadripper
    ]
}

# --- Premium GPU Models (for high-budget builds) ---
PREMIUM_GPU_MODELS = [
    r"rtx 40\d0", r"rtx 4\d{3}",  # RTX 4000 series
    r"rtx 30\d0", r"rtx 3\d{3}",  # RTX 3000 series (high-end)
    r"rtx 4090", r"rtx 4080", r"rtx 3090", r"rtx 3080",  # Specific high-end models
    r"rx 7\d00", r"rx 7\d{3}",  # RX 7000 series
    r"rx 6\d00 xt", r"rx 6\d{3} xt",  # RX 6000 XT models
    r"rx 7900", r"rx 6900", r"rx 6800 xt"  # Specific high-end AMD models
]

# --- High-End CPU Models ---
HIGH_END_CPU_MODELS = {
    "Intel": [
        r"i9-\d{4,}", r"i9 \d{4,}",  # i9 series
        r"i7-\d{4,}k", r"i7 \d{4,}k",  # K-series i7
        r"i7-\d{5,}", r"i7 \d{5,}"   # Newer i7 models
    ],
    "AMD": [
        r"ryzen 9", r"ryzen 7 \d{4,}x", r"threadripper"  # High-end AMD models
    ]
}

# --- High-End GPU Models ---
HIGH_END_GPU_MODELS = [
    r"rtx 40\d0", r"rtx 30\d0", r"rx 7\d00", r"rx 6\d00",  # Latest gen
    r"rtx 4090", r"rtx 4080", r"rtx 3090", r"rtx 3080",    # Top models
    r"rx 7900", r"rx 6900"                                # Top AMD models
]

# --- CPU Brand Patterns ---
CPU_BRAND_PATTERNS = {
    "Intel": [
        r"intel", r"core i\d", r"i\d-\d{4,}", r"i\d \d{4,}", 
        r"celeron", r"pentium", r"xeon"
    ],
    "AMD": [
        r"amd", r"ryzen", r"threadripper", r"epyc", r"athlon"
    ]
}

# --- Preferred CPU Models ---
PREFERRED_CPU_MODELS = {
    "Intel": [
        r"i9-\d{4,}", r"i7-\d{4,}", r"i5-\d{4,}", r"i9 \d{4,}", r"i7 \d{4,}", r"i5 \d{4,}"
    ],
    "AMD": [
        r"ryzen 9", r"ryzen 7", r"ryzen 5", r"threadripper"
    ]
}

# --- CPU Blacklist Patterns (outdated or poor value) ---
CPU_BLACKLIST_PATTERNS = [
    r"celeron", r"pentium", r"xeon", r"athlon", r"fx-", 
    r"i3-[0-9]{3}", r"i5-[0-9]{3}", r"i7-[0-9]{3}",  # Very old Intel generations
    r"ryzen 3 1[0-9]{3}", r"ryzen 5 1[0-9]{3}", r"ryzen 7 1[0-9]{3}",  # First gen Ryzen
    r"ryzen 5 [2-4][0-9]{3}"  # Older Ryzen 5 for high budgets
]

# --- GPU Blacklist Patterns (outdated or poor value for high budgets) ---
GPU_BLACKLIST_PATTERNS = [
    r"gtx 10\d0", r"gtx \d{3}", r"rtx 20\d0",  # Older NVIDIA generations
    r"rx 5\d00", r"rx 5\d{3}", r"rx 580", r"rx 570",  # Older AMD generations
    r"rx 6600", r"rtx 3050", r"rtx 3060"  # Entry-level current gen for high budgets
]

# --- Tier price boundaries ---
TIER_PRICES = {
    "CPU": [150, 300, 550, float("inf")],
    "GPU": [200, 450, 800, 1500, float("inf")],
    "RAM": [50, 100, 200, float("inf")],
    "Storage": [50, 100, 200, float("inf")],
    "Motherboard": [120, 250, 400, float("inf")],
    "PSU": [70, 120, float("inf")],
    "Case": [float("inf")], # Simplified: all cases in one tier
    "CPU Cooler": [50, 100, float("inf")], # Simplified: all coolers in one tier
}

# --- Budget Allocation Guidelines (Dynamic based on budget, INCREASED GPU PRIORITY) ---
def get_dynamic_budget_allocation(initial_budget):
    """Returns dynamic budget allocation percentages based on initial budget."""
    if initial_budget <= 800:
        # Budget build focus (slightly more GPU)
        return {
            "GPU": 0.35,  # Increased
            "CPU": 0.25,
            "Motherboard": 0.10,
            "RAM": 0.10,
            "PSU": 0.08,  # Reduced slightly
            "Storage": 0.08, # Reduced slightly
            "Case": 0.04,  # Reduced slightly
            "CPU Cooler": 0.00,
        }
    elif initial_budget <= 1500:
        # Mid-range build focus (more GPU)
        return {
            "GPU": 0.40,  # Increased
            "CPU": 0.30,
            "Motherboard": 0.10,
            "RAM": 0.08,  # Reduced slightly
            "PSU": 0.05,
            "Storage": 0.05,
            "Case": 0.02,  # Reduced slightly
            "CPU Cooler": 0.00,
        }
    else:
        # High-end build focus (even more GPU)
        return {
            "GPU": 0.45,  # Increased
            "CPU": 0.30,  # Reduced slightly
            "Motherboard": 0.10,
            "RAM": 0.08,  # Reduced slightly
            "PSU": 0.05,
            "Storage": 0.02, # Reduced slightly
            "Case": 0.00,
            "CPU Cooler": 0.00,
        }

# --- Maximum Budget Allocation (Hard ceiling per component, INCREASED GPU PRIORITY) ---
def get_max_budget_allocation(initial_budget):
    """Returns maximum budget allocation percentages to prevent overspending."""
    if initial_budget <= 800:
        return {
            "GPU": 0.40,  # Increased
            "CPU": 0.30,
            "Motherboard": 0.15,
            "RAM": 0.15,
            "PSU": 0.15,
            "Storage": 0.15,
            "Case": 0.15,
            "CPU Cooler": 0.10,
        }
    elif initial_budget <= 1500:
        return {
            "GPU": 0.45,  # Increased
            "CPU": 0.35,
            "Motherboard": 0.15,
            "RAM": 0.15,
            "PSU": 0.12,
            "Storage": 0.12,
            "Case": 0.10,
            "CPU Cooler": 0.08,
        }
    else:
        # For high budgets, allow more spending on premium components
        return {
            "GPU": 0.55,  # Increased significantly
            "CPU": 0.40,  # Reduced slightly
            "Motherboard": 0.20,
            "RAM": 0.15,  # Reduced slightly
            "PSU": 0.15,
            "Storage": 0.10, # Reduced slightly
            "Case": 0.10,  # Reduced slightly
            "CPU Cooler": 0.08, # Reduced slightly
        }

# --- Minimum Budget Allocation (Dynamic based on budget, INCREASED GPU PRIORITY) ---
def get_dynamic_min_budget_allocation(initial_budget):
    """Returns dynamic minimum budget allocation percentages."""
    # For high budgets, enforce higher minimums on performance components
    if initial_budget <= 800:
        return {
            "GPU": 0.25,  # Increased
            "CPU": 0.15,
            "Motherboard": 0.08,
            "RAM": 0.08,
            "PSU": 0.05,
            "Storage": 0.05,
            "Case": 0.05,
            "CPU Cooler": 0.0,
        }
    elif initial_budget <= 1500:
        return {
            "GPU": 0.30,  # Increased
            "CPU": 0.20,
            "Motherboard": 0.08,
            "RAM": 0.08,
            "PSU": 0.05,
            "Storage": 0.05,
            "Case": 0.03,
            "CPU Cooler": 0.0,
        }
    else:
        # For high budgets, enforce higher minimums to ensure premium components
        return {
            "GPU": 0.35,  # Increased significantly
            "CPU": 0.25,
            "Motherboard": 0.10,
            "RAM": 0.10,
            "PSU": 0.07,
            "Storage": 0.07,
            "Case": 0.05,
            "CPU Cooler": 0.03,
        }

# --- Socket Mapping for CPUs and Motherboards ---
SOCKET_MAPPING = {
    # Intel
    "LGA1151": ["LGA1151", "LGA 1151", "Socket 1151"],
    "LGA1200": ["LGA1200", "LGA 1200", "Socket 1200"],
    "LGA1700": ["LGA1700", "LGA 1700", "Socket 1700"],
    "LGA2066": ["LGA2066", "LGA 2066", "Socket 2066"],
    "LGA2011": ["LGA2011", "LGA 2011", "Socket 2011", "LGA2011-3", "LGA 2011-3"],
    "LGA1366": ["LGA1366", "LGA 1366", "Socket 1366"],
    "LGA775": ["LGA775", "LGA 775", "Socket 775"],
    # AMD
    "AM4": ["AM4", "Socket AM4"],
    "AM5": ["AM5", "Socket AM5"],
    "AM3+": ["AM3+", "AM3 Plus", "Socket AM3+"],
    "TR4": ["TR4", "Socket TR4"],
    "sTRX4": ["sTRX4", "Socket TRX4"],
}

# Reverse mapping for lookup
SOCKET_LOOKUP = {}
for primary, variants in SOCKET_MAPPING.items():
    for variant in variants:
        SOCKET_LOOKUP[variant.lower()] = primary

# --- Memory Type Mapping ---
MEMORY_TYPE_MAPPING = {
    "DDR4": ["DDR4", "DDR 4", "DDR-4", "SDRAM DDR4"],
    "DDR5": ["DDR5", "DDR 5", "DDR-5", "SDRAM DDR5"],
    "DDR3": ["DDR3", "DDR 3", "DDR-3", "SDRAM DDR3"],
}

# Reverse mapping for memory type lookup
MEMORY_TYPE_LOOKUP = {}
for primary, variants in MEMORY_TYPE_MAPPING.items():
    for variant in variants:
        MEMORY_TYPE_LOOKUP[variant.lower()] = primary

# --- Compatibility Rules ---
def check_cpu_motherboard_compatibility(mobo, cpu):
    """Enhanced compatibility check for CPU and motherboard with fallback strategies."""
    mobo_socket = mobo.get("Socket/CPU")
    cpu_socket = cpu.get("Socket")
    
    # Debug output
    # print(f"Checking compatibility: CPU Socket: {cpu_socket}, Mobo Socket: {mobo_socket}")
    
    if mobo_socket is None or cpu_socket is None: 
        return False  # If either socket is unknown, consider incompatible
    
    mobo_socket_str = str(mobo_socket).lower()
    cpu_socket_str = str(cpu_socket).lower()
    
    # Direct match
    if mobo_socket_str == cpu_socket_str: 
        return True
    
    # Normalized match
    mobo_normalized = SOCKET_LOOKUP.get(mobo_socket_str)
    cpu_normalized = SOCKET_LOOKUP.get(cpu_socket_str)
    
    if mobo_normalized and cpu_normalized: 
        return mobo_normalized == cpu_normalized
    
    # Check if normalized socket is in the other socket string
    if mobo_normalized and mobo_normalized.lower() in cpu_socket_str: 
        return True
    if cpu_normalized and cpu_normalized.lower() in mobo_socket_str: 
        return True
    
    # Check if they belong to the same socket family
    for socket_family in SOCKET_MAPPING:
        if socket_family.lower() in mobo_socket_str and socket_family.lower() in cpu_socket_str: 
            return True
    
    # Check for substring matches (at least 4 chars)
    for i in range(len(mobo_socket_str) - 3):
        substr = mobo_socket_str[i:i+4]
        if substr.isalnum() and substr in cpu_socket_str: 
            return True
    
    # Intel vs AMD check - these are never compatible
    is_intel_cpu = any(re.search(pattern, str(cpu.get("Name", "")).lower()) for pattern in CPU_BRAND_PATTERNS["Intel"])
    is_amd_cpu = any(re.search(pattern, str(cpu.get("Name", "")).lower()) for pattern in CPU_BRAND_PATTERNS["AMD"])
    
    is_intel_mobo = "intel" in mobo_socket_str or "lga" in mobo_socket_str
    is_amd_mobo = "amd" in mobo_socket_str or "am" in mobo_socket_str or "tr" in mobo_socket_str
    
    if (is_intel_cpu and is_amd_mobo) or (is_amd_cpu and is_intel_mobo):
        return False
    
    # Special case for unknown sockets
    if "unknown" in mobo_socket_str or "unknown" in cpu_socket_str:
        return False  # Changed to be more strict
    
    # Special case for very short socket strings
    if len(mobo_socket_str) <= 3 or len(cpu_socket_str) <= 3:
        return False  # Changed to be more strict
    
    return False  # Default to incompatible if no match found

def check_ram_motherboard_compatibility(mobo, ram):
    """Enhanced compatibility check for RAM and motherboard with fallback strategies."""
    mobo_mem_type = mobo.get("Memory Type")
    ram_mem_type = ram.get("Memory Type")
    if mobo_mem_type is None or ram_mem_type is None: return True
    mobo_mem_type_str = str(mobo_mem_type).lower()
    ram_mem_type_str = str(ram_mem_type).lower()
    if mobo_mem_type_str == ram_mem_type_str: return True
    mobo_normalized = MEMORY_TYPE_LOOKUP.get(mobo_mem_type_str)
    ram_normalized = MEMORY_TYPE_LOOKUP.get(ram_mem_type_str)
    if mobo_normalized and ram_normalized: return mobo_normalized == ram_normalized
    if mobo_normalized and mobo_normalized.lower() in ram_mem_type_str: return True
    if ram_normalized and ram_normalized.lower() in mobo_mem_type_str: return True
    for mem_type in MEMORY_TYPE_MAPPING:
        if mem_type.lower() in mobo_mem_type_str and mem_type.lower() in ram_mem_type_str: return True
    for i in range(len(mobo_mem_type_str) - 2):
        substr = mobo_mem_type_str[i:i+3]
        if substr.isalnum() and substr in ram_mem_type_str: return True
    if "unknown" in mobo_mem_type_str or "unknown" in ram_mem_type_str: return True
    return False

# --- Compatibility Rules ---
COMPATIBILITY_RULES = {
    ("Motherboard", "CPU"): check_cpu_motherboard_compatibility,
    ("Motherboard", "RAM"): check_ram_motherboard_compatibility,
}

# --- Build Tiers based on Budget ---
BUILD_TIERS = [
    ("Entry-Level / Basic PC", 0, 500),
    ("Budget Gaming PC", 500, 800),
    ("Mid-Range Gaming / Balanced PC", 800, 1200),
    ("High-End Gaming / Performance PC", 1200, 2000),
    ("Enthusiast / Premium PC", 2000, 3000),
    ("Extreme / Workstation / Dream PC", 3000, float("inf")),
]

# --- Dynamic Tier Recommendation based on Budget ---
def get_dynamic_tier_recommendation(initial_budget):
    """Returns a recommended target tier (1-5+) based on budget."""
    if initial_budget < 500: return 1
    if initial_budget < 800: return 2
    if initial_budget < 1200: return 3
    if initial_budget < 2000: return 4
    if initial_budget < 3000: return 5
    return 6 # Max tier for very high budgets

def _clean_numeric_string(value, unit_to_remove=None, type_cast=float):
    """Cleans a string to extract a numeric value."""
    if pd.isna(value) or value is None: return None
    s = str(value).strip().replace("$", "").replace(",", "")
    if unit_to_remove: s = s.replace(unit_to_remove, "")
    match = re.search(r"(\d+\.?\d*)", s)
    if match:
        try: return type_cast(match.group(1))
        except ValueError: return None
    return None

def extract_socket_from_name(name):
    """Extract socket information from component name."""
    if not name or not isinstance(name, str): return None
    name_lower = name.lower()
    socket_patterns = [r"lga\s*(\d+)", r"socket\s*(\d+)", r"am(\d+)(\+)?", r"fm(\d+)(\+)?", r"tr(\d+)", r"strx(\d+)"]
    for pattern in socket_patterns:
        match = re.search(pattern, name_lower)
        if match:
            socket_text = match.group(0)
            for socket_family, variants in SOCKET_MAPPING.items():
                if any(variant.lower() in socket_text for variant in variants): return socket_family
            return socket_text.upper()
    for socket_family, variants in SOCKET_MAPPING.items():
        if any(variant.lower() in name_lower for variant in variants): return socket_family
    return None

def extract_memory_type_from_name(name):
    """Extract memory type information from component name."""
    if not name or not isinstance(name, str): return None
    name_lower = name.lower()
    memory_patterns = [r"ddr(\d+)", r"ddr\s*(\d+)"]
    for pattern in memory_patterns:
        match = re.search(pattern, name_lower)
        if match:
            mem_type_text = match.group(0)
            for mem_type, variants in MEMORY_TYPE_MAPPING.items():
                if any(variant.lower() in mem_type_text for variant in variants): return mem_type
            return mem_type_text.upper()
    for mem_type, variants in MEMORY_TYPE_MAPPING.items():
        if any(variant.lower() in name_lower for variant in variants): return mem_type
    return None

def extract_cpu_generation(name):
    """Extract CPU generation from name."""
    if not name or not isinstance(name, str): return None
    name_lower = name.lower()
    
    # Intel patterns
    intel_gen_patterns = [
        r"i[3579]-(\d)(\d{3})",  # i5-10400, i7-11700K, etc.
        r"i[3579] (\d)(\d{3})",   # i5 10400, i7 11700K, etc.
        r"(\d+)th gen",          # 10th gen, 11th gen, etc.
    ]
    
    for pattern in intel_gen_patterns:
        match = re.search(pattern, name_lower)
        if match:
            if len(match.groups()) >= 2:
                return int(match.group(1))  # First digit is the generation
            elif len(match.groups()) == 1:
                return int(match.group(1).replace("th", ""))
    
    # AMD Ryzen patterns
    amd_gen_patterns = [
        r"ryzen [3579] (\d)(\d{3})",  # Ryzen 5 5600X, Ryzen 7 3700X, etc.
        r"ryzen [3579](\d{4})",       # Ryzen 5600X, Ryzen 3700X, etc.
    ]
    
    for pattern in amd_gen_patterns:
        match = re.search(pattern, name_lower)
        if match:
            if len(match.groups()) >= 2:
                return int(match.group(1))  # First digit is the generation
            elif len(match.groups()) == 1:
                return int(str(match.group(1))[0])  # First digit of the model number
    
    return None

def extract_cpu_cores(name, default_cores=None):
    """Extract CPU core count from name if available."""
    if not name or not isinstance(name, str): return default_cores
    name_lower = name.lower()
    
    # Look for core count patterns
    core_patterns = [
        r"(\d+)[ -]cores?",  # 8-core, 6 cores, etc.
        r"(\d+)c",           # 8C, 6C, etc.
    ]
    
    for pattern in core_patterns:
        match = re.search(pattern, name_lower)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    
    return default_cores

def is_modern_cpu(cpu_data):
    """Check if a CPU is modern based on generation and other factors."""
    name = cpu_data.get("Name", "")
    if not name:
        return False
    
    name_lower = name.lower()
    
    # Check against blacklist patterns
    for pattern in CPU_BLACKLIST_PATTERNS:
        if re.search(pattern, name_lower):
            return False
    
    # Extract generation
    generation = extract_cpu_generation(name)
    
    # Determine brand
    brand = None
    for b, patterns in CPU_BRAND_PATTERNS.items():
        if any(re.search(pattern, name_lower) for pattern in patterns):
            brand = b
            break
    
    # Check if generation is modern
    if brand and generation:
        return generation in MODERN_CPU_GENERATIONS.get(brand, [])
    
    # If we can't determine generation, use core count as a fallback
    core_count = cpu_data.get("Core Count")
    if core_count is None:
        core_count = extract_cpu_cores(name)
    
    # Modern CPUs typically have at least 6 cores
    if core_count and core_count >= 6:
        return True
    
    # Check for preferred model patterns as a last resort
    if brand:
        return any(re.search(pattern, name_lower) for pattern in PREFERRED_CPU_MODELS.get(brand, []))
    
    return False

def is_premium_cpu(cpu_data, budget):
    """Check if a CPU is premium (high-end) based on model and features."""
    if budget < 1500:  # Only apply for high budgets
        return False
        
    name = cpu_data.get("Name", "")
    if not name:
        return False
    
    name_lower = name.lower()
    
    # Determine brand
    brand = None
    for b, patterns in CPU_BRAND_PATTERNS.items():
        if any(re.search(pattern, name_lower) for pattern in patterns):
            brand = b
            break
    
    # Check if it's a premium model
    if brand:
        return any(re.search(pattern, name_lower) for pattern in PREMIUM_CPU_MODELS.get(brand, []))
    
    # Check core count as fallback
    core_count = cpu_data.get("Core Count")
    if core_count is None:
        core_count = extract_cpu_cores(name)
    
    # Premium CPUs typically have at least 8 cores for high budgets
    if budget >= 2000:
        return core_count and core_count >= 8
    else:
        return core_count and core_count >= 6
    
    return False

def is_premium_gpu(gpu_data, budget):
    """Check if a GPU is premium (high-end) based on model and features."""
    if budget < 1500:  # Only apply for high budgets
        return False
        
    name = gpu_data.get("Name", "")
    chipset = gpu_data.get("Chipset", "")
    
    if not name and not chipset:
        return False
    
    # Check both name and chipset
    for field in [name, chipset]:
        if not field:
            continue
            
        field_lower = str(field).lower()
        
        # Check if it's a premium model
        if any(re.search(pattern, field_lower) for pattern in PREMIUM_GPU_MODELS):
            return True
    
    # Check memory as fallback
    memory = gpu_data.get("Memory")
    if memory:
        try:
            memory_val = float(memory)
            # Premium GPUs typically have at least 8GB VRAM for high budgets
            if budget >= 2000:
                return memory_val >= 10  # Higher threshold for $2000+ builds
            else:
                return memory_val >= 8
        except (ValueError, TypeError):
            pass
    
    return False

def is_blacklisted_gpu_for_high_budget(gpu_data, budget):
    """Check if a GPU is blacklisted for high-budget builds."""
    if budget < 1500:  # Only apply for high budgets
        return False
        
    name = gpu_data.get("Name", "")
    chipset = gpu_data.get("Chipset", "")
    
    if not name and not chipset:
        return False
    
    # Check both name and chipset
    for field in [name, chipset]:
        if not field:
            continue
            
        field_lower = str(field).lower()
        
        # Check if it's a blacklisted model
        if any(re.search(pattern, field_lower) for pattern in GPU_BLACKLIST_PATTERNS):
            return True
    
    return False

def calculate_cpu_value_score(cpu_data, budget=1500):
    """Calculate a value score for a CPU based on performance and price."""
    price = cpu_data.get("Price", 0)
    if price is None or price <= 0:
        return 0  # Can't calculate value without a valid price
    
    # Base score from core count
    core_count = cpu_data.get("Core Count")
    if core_count is None:
        core_count = extract_cpu_cores(cpu_data.get("Name", ""), 4)  # Default to 4 cores if unknown
    
    core_score = core_count * 10  # 10 points per core
    
    # Boost from clock speed
    boost_clock = None
    if "Performance Core Boost Clock" in cpu_data and cpu_data["Performance Core Boost Clock"] is not None:
        try:
            boost_clock_str = str(cpu_data["Performance Core Boost Clock"])
            boost_clock = _clean_numeric_string(boost_clock_str.replace("GHz", "").strip())
        except (ValueError, TypeError, AttributeError):
            pass
    
    clock_score = 0
    if boost_clock:
        clock_score = boost_clock * 20  # 20 points per GHz
    
    # Generation bonus
    generation = extract_cpu_generation(cpu_data.get("Name", ""))
    gen_score = 0
    if generation:
        # More recent generations get higher scores
        if "intel" in cpu_data.get("Name", "").lower():
            gen_score = (generation - 8) * 15 if generation > 8 else 0  # 15 points per generation above 8
        else:  # AMD
            gen_score = (generation - 2) * 15 if generation > 2 else 0  # 15 points per generation above 2
    
    # Premium bonus for high-budget builds
    premium_bonus = 0
    if budget >= 1500 and is_premium_cpu(cpu_data, budget):
        if budget >= 2000:
            premium_bonus = 200  # Very significant bonus for premium CPUs in $2000+ builds
        else:
            premium_bonus = 100  # Significant bonus for premium CPUs in high-end builds
    
    # Calculate total performance score
    performance_score = core_score + clock_score + gen_score + premium_bonus
    
    # For high budgets, prioritize performance over value
    if budget >= 2000:
        # Almost entirely based on absolute performance for premium builds
        return (performance_score * 0.9) + ((performance_score / price) * 0.1 * 100)
    elif budget >= 1500:
        # Blend of absolute performance and value
        return (performance_score * 0.7) + ((performance_score / price) * 0.3 * 100)
    else:
        # Calculate value (performance per dollar)
        return performance_score / price

def calculate_gpu_value_score(gpu_data, budget=1500):
    """Calculate a value score for a GPU based on performance and price (INCREASED GPU PRIORITY)."""
    price = gpu_data.get("Price", 0)
    if price is None or price <= 0:
        return 0  # Can't calculate value without a valid price
    
    # Check if GPU is blacklisted for high budgets
    if budget >= 1500 and is_blacklisted_gpu_for_high_budget(gpu_data, budget):
        return -1000  # Heavily penalize blacklisted GPUs for high budgets
    
    # Base score from memory size
    memory = gpu_data.get("Memory")
    memory_score = 0
    if memory is not None:
        try:
            memory_val = float(memory)
            memory_score = memory_val * 20  # Increased weight for VRAM
        except (ValueError, TypeError):
            pass
    
    # Clock speed bonus
    boost_clock = gpu_data.get("Boost Clock")
    clock_score = 0
    if boost_clock is not None:
        try:
            boost_clock_val = float(boost_clock)
            clock_score = boost_clock_val / 50  # Increased weight for clock speed
        except (ValueError, TypeError):
            pass
    
    # Model generation bonus based on chipset name
    chipset = str(gpu_data.get("Chipset", "")).lower()
    gen_score = 0
    
    # NVIDIA RTX series
    if "rtx" in chipset:
        if "4" in chipset:  # RTX 4000 series
            gen_score = 250 # Increased
        elif "3" in chipset:  # RTX 3000 series
            gen_score = 200 # Increased
        elif "2" in chipset:  # RTX 2000 series
            gen_score = 75  # Increased
    
    # AMD RX series
    elif "rx" in chipset:
        if "7" in chipset:  # RX 7000 series
            gen_score = 250 # Increased
        elif "6" in chipset:  # RX 6000 series
            gen_score = 200 # Increased
        elif "5" in chipset:  # RX 5000 series
            gen_score = 75  # Increased
    
    # Premium bonus for high-budget builds
    premium_bonus = 0
    if budget >= 1500 and is_premium_gpu(gpu_data, budget):
        if budget >= 2000:
            premium_bonus = 400  # Increased significantly
        else:
            premium_bonus = 300  # Increased significantly
    
    # Calculate total performance score
    performance_score = memory_score + clock_score + gen_score + premium_bonus
    
    # For high budgets, prioritize performance over value
    if budget >= 2000:
        # Almost entirely based on absolute performance for premium builds
        return (performance_score * 0.95) + ((performance_score / price) * 0.05 * 100) # Increased performance weight
    elif budget >= 1500:
        # Blend of absolute performance and value
        return (performance_score * 0.8) + ((performance_score / price) * 0.2 * 100) # Increased performance weight
    else:
        # Calculate value (performance per dollar)
        return performance_score / price

def load_parts_from_csv(file_path, component_type):
    """Loads part data from a CSV file and preprocesses numeric columns."""
    if not os.path.exists(file_path): print(f"Error: CSV file not found at {file_path}"); return []
    try:
        df = pd.read_csv(file_path)
        if "Price" in df.columns: df["Price"] = df["Price"].apply(lambda x: _clean_numeric_string(x, type_cast=float))
        if component_type == "CPU":
            if "Core Count" in df.columns: df["Core Count"] = df["Core Count"].apply(lambda x: _clean_numeric_string(x, type_cast=int)).fillna(4)
            if "TDP" in df.columns: df["TDP"] = df["TDP"].apply(lambda x: _clean_numeric_string(x, unit_to_remove=" W", type_cast=float))
            if "Socket" in df.columns:
                df["Socket"] = df["Socket"].apply(lambda x: extract_socket_from_name(x) if pd.isna(x) or str(x).lower() == "unknown" else x)
                for idx, row in df.iterrows():
                    if pd.isna(row["Socket"]) or str(row["Socket"]).lower() == "unknown":
                        socket_from_name = extract_socket_from_name(row["Name"])
                        if socket_from_name: df.at[idx, "Socket"] = socket_from_name
            else: df["Socket"] = df["Name"].apply(extract_socket_from_name)
        elif component_type == "Motherboard":
            if "Socket/CPU" in df.columns:
                df["Socket/CPU"] = df["Socket/CPU"].apply(lambda x: extract_socket_from_name(x) if pd.isna(x) or str(x).lower() == "unknown" else x)
                for idx, row in df.iterrows():
                    if pd.isna(row["Socket/CPU"]) or str(row["Socket/CPU"]).lower() == "unknown":
                        socket_from_name = extract_socket_from_name(row["Name"])
                        if socket_from_name: df.at[idx, "Socket/CPU"] = socket_from_name
            else: df["Socket/CPU"] = df["Name"].apply(extract_socket_from_name)
            if "Memory Type" in df.columns:
                df["Memory Type"] = df["Memory Type"].apply(lambda x: extract_memory_type_from_name(x) if pd.isna(x) or str(x).lower() == "unknown" else x)
                for idx, row in df.iterrows():
                    if pd.isna(row["Memory Type"]) or str(row["Memory Type"]).lower() == "unknown":
                        mem_type_from_name = extract_memory_type_from_name(row["Name"])
                        df.at[idx, "Memory Type"] = mem_type_from_name if mem_type_from_name else "DDR4"
            else: df["Memory Type"] = df["Name"].apply(lambda x: extract_memory_type_from_name(x) if extract_memory_type_from_name(x) else "DDR4")
        elif component_type == "RAM":
            if "Speed" in df.columns: df["Speed"] = df["Speed"].apply(lambda x: _clean_numeric_string(x, unit_to_remove=" MHz", type_cast=float)).fillna(3200.0)
            if "CAS Latency" in df.columns: df["CAS Latency"] = df["CAS Latency"].apply(lambda x: _clean_numeric_string(x, type_cast=float))
            if "Modules" in df.columns: df["Modules"] = df["Modules"].apply(lambda x: _clean_numeric_string(x, type_cast=int))
            if "Memory Type" in df.columns:
                df["Memory Type"] = df["Memory Type"].apply(lambda x: extract_memory_type_from_name(x) if pd.isna(x) or str(x).lower() == "unknown" else x)
                for idx, row in df.iterrows():
                    if pd.isna(row["Memory Type"]) or str(row["Memory Type"]).lower() == "unknown":
                        mem_type_from_name = extract_memory_type_from_name(row["Name"])
                        df.at[idx, "Memory Type"] = mem_type_from_name if mem_type_from_name else "DDR4"
            else: df["Memory Type"] = df["Name"].apply(lambda x: extract_memory_type_from_name(x) if extract_memory_type_from_name(x) else "DDR4")
        elif component_type == "GPU":
            if "Memory" in df.columns: df["Memory"] = df["Memory"].apply(lambda x: _clean_numeric_string(x, unit_to_remove=" GB", type_cast=float)).fillna(4.0)
            if "Core Clock" in df.columns: df["Core Clock"] = df["Core Clock"].apply(lambda x: _clean_numeric_string(x, unit_to_remove=" MHz", type_cast=float))
            if "Boost Clock" in df.columns: df["Boost Clock"] = df["Boost Clock"].apply(lambda x: _clean_numeric_string(x, unit_to_remove=" MHz", type_cast=float))
        elif component_type == "Storage":
            if "Capacity" in df.columns: df["Capacity"] = df["Capacity"].apply(lambda x: _clean_numeric_string(x, unit_to_remove=" GB", type_cast=float))
        elif component_type == "PSU":
            if "Wattage" in df.columns: df["Wattage"] = df["Wattage"].apply(lambda x: _clean_numeric_string(x, unit_to_remove=" W", type_cast=float))
        
        # Convert DataFrame to list of dictionaries
        return df.to_dict("records")
    except Exception as e:
        print(f"Error loading {component_type} data: {e}")
        return []

# --- PC Building Environment ---
class PCBuildingEnv(gym.Env):
    """Custom Environment for PC Building with DQN."""
    
    def __init__(self, budget=1000.0, component_data=None):
        super(PCBuildingEnv, self).__init__()
        
        self.budget = budget
        self.remaining_budget = budget
        self.current_build = {}
        self.component_data = component_data or {}
        self.component_indices = {}
        self.current_component_type = None
        self.step_count = 0
        self.max_steps = 50
        
        # Load component data if not provided
        if not component_data:
            for component_type, file_path in CSV_FILE_PATHS.items():
                self.component_data[component_type] = load_parts_from_csv(file_path, component_type)
                
        # Create component indices for action space
        total_components = 0
        for component_type, components in self.component_data.items():
            self.component_indices[component_type] = (total_components, total_components + len(components))
            total_components += len(components)
            
        # Define action and observation spaces
        self.action_space = spaces.Discrete(total_components + 1)  # +1 for "skip" action
        
        # Observation space: budget + binary flags for selected components + component features
        self.observation_space = spaces.Dict({
            "budget": spaces.Box(low=0, high=float("inf"), shape=(1,), dtype=np.float32),
            "selected_components": spaces.MultiBinary(len(COMPONENT_PRIORITY)),
            "current_component_type": spaces.Discrete(len(COMPONENT_PRIORITY) + 1),  # +1 for "none"
            "compatibility": spaces.MultiBinary(len(COMPONENT_PRIORITY)),
        })
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.remaining_budget = self.budget
        self.current_build = {}
        self.step_count = 0
        self.current_component_type = COMPONENT_PRIORITY[0]
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment based on the action."""
        self.step_count += 1
        
        # Check if action is "skip"
        if action >= sum(len(components) for components in self.component_data.values()):
            # Skip current component
            reward = -5.0  # Penalty for skipping
            self._advance_component_type()
        else:
            # Find which component type and index the action corresponds to
            component_type = None
            component_idx = None
            
            for c_type, (start_idx, end_idx) in self.component_indices.items():
                if start_idx <= action < end_idx:
                    component_type = c_type
                    component_idx = action - start_idx
                    break
            
            if component_type and component_idx is not None:
                # Check if component_idx is valid for this component type
                if component_idx < len(self.component_data[component_type]):
                    # Get the selected component
                    selected_component = self.component_data[component_type][component_idx]
                    
                    # Check if we can afford it
                    price = selected_component.get("Price", 0)
                    if price is None: price = 0
                    
                    if price <= self.remaining_budget:
                        # Check compatibility with existing components
                        is_compatible = self._check_compatibility(component_type, selected_component)
                        
                        if is_compatible:
                            # Add component to build
                            self.current_build[component_type] = selected_component
                            self.remaining_budget -= price
                            
                            # Calculate reward based on component quality and compatibility
                            reward = self._calculate_reward(component_type, selected_component)
                            
                            # Move to next component type
                            self._advance_component_type()
                        else:
                            # Penalty for incompatible selection
                            reward = -10.0
                    else:
                        # Penalty for exceeding budget
                        reward = -15.0
                else:
                    # Invalid component index
                    print(f"Warning: Invalid component index {component_idx} for {component_type}. Max index is {len(self.component_data[component_type])-1}")
                    reward = -20.0
            else:
                # Invalid action
                print(f"Warning: Invalid action {action}. Action space size is {self.action_space.n}")
                reward = -20.0
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps or 
                self.current_component_type is None or
                all(c_type in self.current_build for c_type in ESSENTIAL_COMPONENTS))
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "build": self.current_build,
            "remaining_budget": self.remaining_budget,
            "step_count": self.step_count,
        }
        
        # If done, add final build evaluation to reward
        if done:
            final_reward = self._evaluate_final_build()
            reward += final_reward
            info["final_reward"] = final_reward
            info["total_reward"] = reward
        
        return observation, reward, done, False, info
    
    def _advance_component_type(self):
        """Move to the next component type in priority order."""
        if self.current_component_type is None:
            return
            
        current_idx = COMPONENT_PRIORITY.index(self.current_component_type)
        next_idx = current_idx + 1
        
        if next_idx < len(COMPONENT_PRIORITY):
            self.current_component_type = COMPONENT_PRIORITY[next_idx]
        else:
            self.current_component_type = None
    
    def _check_compatibility(self, component_type, component):
        """Check if a component is compatible with the current build."""
        for (type1, type2), check_func in COMPATIBILITY_RULES.items():
            if component_type == type1 and type2 in self.current_build:
                if not check_func(component, self.current_build[type2]):
                    return False
            elif component_type == type2 and type1 in self.current_build:
                if not check_func(self.current_build[type1], component):
                    return False
        return True
    
    def _calculate_reward(self, component_type, component):
        """Calculate reward for selecting a component (INCREASED GPU PRIORITY)."""
        # Base reward for selecting a component
        reward = 5.0
        
        # Enhanced reward calculation for CPU
        if component_type == "CPU":
            cpu_reward = calculate_enhanced_cpu_reward(component)
            
            # Additional reward for premium CPUs in high-budget builds
            if self.budget >= 1500 and is_premium_cpu(component, self.budget):
                if self.budget >= 2000:
                    cpu_reward += 100.0  # Very significant bonus for premium CPUs in $2000+ builds
                else:
                    cpu_reward += 50.0  # Significant bonus for premium CPUs in high-end builds
                
            reward += cpu_reward
        
        # Enhanced reward calculation for GPU (INCREASED WEIGHT)
        elif component_type == "GPU":
            gpu_reward = calculate_enhanced_gpu_reward(component)
            
            # Additional reward for premium GPUs in high-budget builds (INCREASED BONUS)
            if self.budget >= 1500 and is_premium_gpu(component, self.budget):
                if self.budget >= 2000:
                    gpu_reward += 150.0  # Increased bonus
                else:
                    gpu_reward += 75.0  # Increased bonus
                
            # Penalty for blacklisted GPUs in high-budget builds
            if self.budget >= 1500 and is_blacklisted_gpu_for_high_budget(component, self.budget):
                gpu_reward -= 100.0  # Heavy penalty for selecting outdated GPUs in high-budget builds
                
            reward += gpu_reward * 1.2 # Apply extra weight to GPU reward
        
        # Reward for other components based on price tier
        else:
            price = component.get("Price", 0)
            if price is None: price = 0
            
            # Find the tier of the component based on price
            tier = 0
            for i, threshold in enumerate(TIER_PRICES.get(component_type, [])):
                if price <= threshold:
                    tier = i
                    break
            
            # Higher tier components get higher rewards
            reward += tier * 2.0
        
        # Additional reward for staying within budget allocation guidelines
        price = component.get("Price", 0)
        if price is None: price = 0
        
        max_allocation = get_max_budget_allocation(self.budget).get(component_type, 0.5)
        if price <= self.budget * max_allocation:
            reward += 2.0
        else:
            reward -= 5.0
        
        # Reward for balanced build (spending appropriate amounts on each component)
        min_allocation = get_dynamic_min_budget_allocation(self.budget).get(component_type, 0.0)
        if price >= self.budget * min_allocation:
            reward += 3.0
        
        return reward
    
    def _evaluate_final_build(self):
        """Evaluate the final build and return a reward (INCREASED GPU PRIORITY)."""
        # Check if all essential components are present
        missing_components = [c for c in ESSENTIAL_COMPONENTS if c not in self.current_build]
        if missing_components:
            # Heavily penalize missing essential components
            return -50.0 * len(missing_components)
        
        # Calculate total build reward using enhanced reward function
        build_reward = calculate_total_reward(self.current_build)
        
        # Add extra weight to GPU contribution in final reward
        if "GPU" in self.current_build:
            gpu_score = calculate_gpu_value_score(self.current_build["GPU"], self.budget)
            build_reward += gpu_score * 0.2 # Add 20% of GPU score to final reward
        
        # Add bonus for remaining budget (efficiency)
        budget_efficiency = (self.remaining_budget / self.budget) * 10.0
        if budget_efficiency > 5.0:  # Cap at 5.0 to prevent too much underspending
            budget_efficiency = 5.0
        
        # Add bonus for balanced spending across components
        balance_score = self._calculate_balance_score()
        
        # Combine all factors
        final_reward = build_reward + budget_efficiency + balance_score
        
        return final_reward
    
    def _calculate_balance_score(self):
        """Calculate a score for how well-balanced the build is."""
        # Get ideal budget allocation
        ideal_allocation = get_dynamic_budget_allocation(self.budget)
        
        # Calculate actual allocation
        total_spent = self.budget - self.remaining_budget
        if total_spent == 0:
            return -10.0  # Penalty for not spending anything
        
        actual_allocation = {}
        for c_type, component in self.current_build.items():
            price = component.get("Price", 0)
            if price is None: price = 0
            actual_allocation[c_type] = price / self.budget
        
        # Calculate deviation from ideal allocation
        deviation = 0.0
        for c_type, ideal in ideal_allocation.items():
            actual = actual_allocation.get(c_type, 0.0)
            deviation += abs(ideal - actual)
        
        # Convert to a score (lower deviation is better)
        balance_score = 10.0 - (deviation * 20.0)
        if balance_score < -10.0:
            balance_score = -10.0
        
        return balance_score
    
    def _get_observation(self):
        """Get the current observation."""
        # Budget observation
        budget_obs = np.array([self.remaining_budget / self.budget], dtype=np.float32)
        
        # Selected components observation
        selected_components = np.zeros(len(COMPONENT_PRIORITY), dtype=np.int8)
        for i, c_type in enumerate(COMPONENT_PRIORITY):
            if c_type in self.current_build:
                selected_components[i] = 1
        
        # Current component type observation
        current_component_idx = (
            COMPONENT_PRIORITY.index(self.current_component_type)
            if self.current_component_type in COMPONENT_PRIORITY
            else len(COMPONENT_PRIORITY)
        )
        
        # Compatibility observation
        compatibility = np.ones(len(COMPONENT_PRIORITY), dtype=np.int8)
        if self.current_component_type:
            for i, c_type in enumerate(COMPONENT_PRIORITY):
                if c_type != self.current_component_type and c_type in self.current_build:
                    # Check if there's a compatibility rule for these components
                    rule_key = (self.current_component_type, c_type)
                    reverse_rule_key = (c_type, self.current_component_type)
                    
                    if rule_key in COMPATIBILITY_RULES or reverse_rule_key in COMPATIBILITY_RULES:
                        # There is a rule, so we need to check compatibility for each potential component
                        compatibility[i] = 0  # Assume incompatible by default
                        
                        # We'll set it to 1 later if we find compatible components
        
        return {
            "budget": budget_obs,
            "selected_components": selected_components,
            "current_component_type": current_component_idx,
            "compatibility": compatibility,
        }

# --- DQN Agent ---
class DQN(nn.Module):
    """Deep Q-Network for PC Building."""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        
        # Define network architecture
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
    
    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# --- Replay Buffer ---
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences from buffer."""
        experiences = random.sample(self.buffer, k=batch_size)
        
        states = torch.from_numpy(np.vstack([self._flatten_state(e.state) for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([np.array([e.action]) for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([np.array([e.reward]) for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self._flatten_state(e.next_state) for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([np.array([e.done]) for e in experiences]).astype(np.uint8)).float().to(device)
        
        return states, actions, rewards, next_states, dones
    
    def _flatten_state(self, state):
        """Flatten the dictionary state into a single vector."""
        budget = state["budget"]
        selected_components = state["selected_components"]
        current_component_type = np.array([state["current_component_type"]])
        compatibility = state["compatibility"]
        
        return np.concatenate([budget, selected_components, current_component_type, compatibility])
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

# --- DQN Agent ---
class DQNAgent:
    """Agent implementing DQN for PC Building."""
    
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Networks
        self.qnetwork_local = DQN(state_size, action_size).to(device)
        self.qnetwork_target = DQN(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3
        self.update_every = 4
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn from batch of experiences."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
    
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy."""
        # Flatten state
        state_flat = np.concatenate([
            state["budget"],
            state["selected_components"],
            np.array([state["current_component_type"]]),
            state["compatibility"]
        ])
        
        state_tensor = torch.from_numpy(state_flat).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).item()  # Convert to Python scalar
        else:
            # Fix: Convert numpy array to list before using random.choice
            return random.choice(range(self.action_size))
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# --- Training Function ---
def train_dqn(budget=1000.0, n_episodes=1000, max_t=50, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Train DQN agent for PC building."""
    # Load preprocessed data
    processed_data = load_and_preprocess_all_data()
    
    # Create environment
    env = PCBuildingEnv(budget=budget, component_data=processed_data)
    
    # Print component counts for debugging
    print("Component counts:")
    for component_type, components in env.component_data.items():
        print(f"  {component_type}: {len(components)}")
    print(f"Total action space size: {env.action_space.n}")
    
    # Calculate state and action sizes
    state_size = (
        1 +  # budget
        len(COMPONENT_PRIORITY) +  # selected_components
        1 +  # current_component_type
        len(COMPONENT_PRIORITY)    # compatibility
    )
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)
    
    # Training loop
    scores = []
    eps = eps_start
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            # Select action
            action = agent.act(state, eps)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Update agent
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Update epsilon
        eps = max(eps_end, eps_decay * eps)
        
        # Save score
        scores.append(score)
        
        # Print progress
        if i_episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.2f}")
            
            # Check if goal is achieved
            if avg_score >= 100.0:  # Adjust this threshold as needed
                print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}")
                torch.save(agent.qnetwork_local.state_dict(), "pc_builder_dqn.pth")
                break
    
    return agent, scores

# --- Validation Function ---
def validate_agent(agent, budget=1000.0, n_episodes=10):
    """Validate trained agent on test episodes."""
    # Load preprocessed data
    processed_data = load_and_preprocess_all_data()
    
    # Create environment
    env = PCBuildingEnv(budget=budget, component_data=processed_data)
    
    # Validation loop
    scores = []
    builds = []
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(50):  # max steps
            # Select action (no exploration)
            action = agent.act(state, eps=0.0)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Save score and build
        scores.append(score)
        builds.append(info["build"])
        
        print(f"Test Episode {i_episode}\tScore: {score:.2f}")
    
    # Print average score
    avg_score = np.mean(scores)
    print(f"\nAverage Test Score: {avg_score:.2f}")
    
    # Return best build
    best_idx = np.argmax(scores)
    return builds[best_idx], scores[best_idx]

def print_complete_build(build, budget):
    """Print complete details of the PC build."""
    print("\nComplete PC Build Details:")
    print(f"Budget: ${budget:.2f}")
    
    total_cost = 0.0
    
    # Print components in priority order
    for component_type in COMPONENT_PRIORITY:
        if component_type in build:
            component = build[component_type]
            price = component.get("Price", 0.0)
            if price is None: price = 0.0
            total_cost += price
            
            print(f"\n{component_type}: {component.get('Name', 'Unknown')}")
            print(f"  Price: ${price:.2f}")
            
            # Print additional details based on component type
            if component_type == "CPU":
                print(f"  Socket: {component.get('Socket', 'Unknown')}")
                print(f"  Cores: {component.get('Core Count', 'Unknown')}")
                print(f"  Base Clock: {component.get('Performance Core Clock', 'Unknown')}")
                print(f"  Boost Clock: {component.get('Performance Core Boost Clock', 'Unknown')}")
                print(f"  TDP: {component.get('TDP', 'Unknown')}")
            
            elif component_type == "GPU":
                print(f"  Chipset: {component.get('Chipset', 'Unknown')}")
                print(f"  Memory: {component.get('Memory', 'Unknown')} GB")
                print(f"  Memory Type: {component.get('Memory Type', 'Unknown')}")
                print(f"  Core Clock: {component.get('Core Clock', 'Unknown')}")
                print(f"  Boost Clock: {component.get('Boost Clock', 'Unknown')}")
            
            elif component_type == "Motherboard":
                print(f"  Socket: {component.get('Socket/CPU', 'Unknown')}")
                print(f"  Form Factor: {component.get('Form Factor', 'Unknown')}")
                print(f"  Memory Type: {component.get('Memory Type', 'Unknown')}")
            
            elif component_type == "RAM":
                print(f"  Speed: {component.get('Speed', 'Unknown')}")
                print(f"  Modules: {component.get('Modules', 'Unknown')}")
                print(f"  Memory Type: {component.get('Memory Type', 'Unknown')}")
            
            elif component_type == "Storage":
                print(f"  Capacity: {component.get('Capacity', 'Unknown')} GB")
                print(f"  Type: {component.get('Type', 'Unknown')}")
            
            elif component_type == "PSU":
                print(f"  Wattage: {component.get('Wattage', 'Unknown')} W")
                print(f"  Efficiency Rating: {component.get('Efficiency Rating', 'Unknown')}")
            
            elif component_type == "Case":
                print(f"  Type: {component.get('Type', 'Unknown')}")
                print(f"  Color: {component.get('Color', 'Unknown')}")
            
            elif component_type == "CPU Cooler":
                print(f"  Fan RPM: {component.get('Fan RPM', 'Unknown')}")
                print(f"  Noise Level: {component.get('Noise Level', 'Unknown')}")
        else:
            print(f"\n{component_type}: Not selected")
    
    # Print summary
    print("\nBuild Summary:")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Remaining Budget: ${budget - total_cost:.2f}")
    
    # Check for missing essential components
    missing = [c for c in ESSENTIAL_COMPONENTS if c not in build]
    if missing:
        print(f"Warning: Missing essential components: {', '.join(missing)}")
    
    # Check compatibility
    compatibility_issues = []
    if "CPU" in build and "Motherboard" in build:
        cpu = build["CPU"]
        mobo = build["Motherboard"]
        if not check_cpu_motherboard_compatibility(mobo, cpu):
            compatibility_issues.append("CPU and Motherboard socket mismatch")
    
    if "RAM" in build and "Motherboard" in build:
        ram = build["RAM"]
        mobo = build["Motherboard"]
        if not check_ram_motherboard_compatibility(mobo, ram):
            compatibility_issues.append("RAM and Motherboard memory type mismatch")
    
    if compatibility_issues:
        print(f"Warning: Compatibility issues detected: {', '.join(compatibility_issues)}")
    else:
        print("All selected components are compatible")

# --- Create a sample complete build for demonstration ---
def create_sample_complete_build(budget=1500.0):
    """Create a sample complete build for demonstration purposes with improved CPU selection."""
    # Load preprocessed data
    processed_data = load_and_preprocess_all_data()
    
    # Create a sample build with all essential components
    sample_build = {}
    remaining_budget = budget
    
    # Select GPU first (INCREASED PRIORITY)
    gpus = processed_data["GPU"]
    
    # Filter GPUs by price
    max_gpu_budget = remaining_budget * get_max_budget_allocation(budget).get("GPU", 0.5)
    min_gpu_budget = remaining_budget * get_dynamic_min_budget_allocation(budget).get("GPU", 0.2)
    affordable_gpus = [g for g in gpus if g.get("Price", 0) is not None and g.get("Price", 0) <= max_gpu_budget and g.get("Price", 0) >= min_gpu_budget]
    
    # For high budgets, prioritize premium GPUs
    if budget >= 1500:
        premium_gpus = [g for g in affordable_gpus if is_premium_gpu(g, budget)]
        if premium_gpus:
            affordable_gpus = premium_gpus
    
    # Filter out blacklisted GPUs for high budgets
    if budget >= 1500:
        affordable_gpus = [g for g in affordable_gpus if not is_blacklisted_gpu_for_high_budget(g, budget)]
    
    # Calculate value score for each GPU
    for gpu in affordable_gpus:
        gpu["value_score"] = calculate_gpu_value_score(gpu, budget)
    
    # Sort by value score (descending)
    affordable_gpus.sort(key=lambda g: g.get("value_score", 0), reverse=True)
    
    # Select the best value GPU from the top options
    top_n = max(1, int(len(affordable_gpus) * 0.1))
    selected_gpu = affordable_gpus[0] if affordable_gpus else None
    
    if selected_gpu:
        sample_build["GPU"] = selected_gpu
        price = selected_gpu.get("Price", 0)
        if price is None: price = 0
        remaining_budget -= price
    
    # Then select CPU
    cpus = processed_data["CPU"]
    
    # Filter CPUs by price (must be affordable)
    max_cpu_budget = remaining_budget * get_max_budget_allocation(budget).get("CPU", 0.35)
    min_cpu_budget = remaining_budget * get_dynamic_min_budget_allocation(budget).get("CPU", 0.15)
    affordable_cpus = [c for c in cpus if c.get("Price", 0) is not None and c.get("Price", 0) <= max_cpu_budget and c.get("Price", 0) >= min_cpu_budget]
    
    # For high budgets, prioritize premium CPUs
    if budget >= 1500:
        premium_cpus = [c for c in affordable_cpus if is_premium_cpu(c, budget)]
        if premium_cpus:
            affordable_cpus = premium_cpus
    
    # Filter for modern CPUs
    modern_cpus = [c for c in affordable_cpus if is_modern_cpu(c)]
    
    # If no modern CPUs found, fall back to affordable ones
    if not modern_cpus:
        modern_cpus = affordable_cpus
    
    # Calculate value score for each CPU
    for cpu in modern_cpus:
        cpu["value_score"] = calculate_cpu_value_score(cpu, budget)
    
    # Sort by value score (descending)
    modern_cpus.sort(key=lambda c: c.get("value_score", 0), reverse=True)
    
    # Select the best value CPU from the top options
    top_n = max(1, int(len(modern_cpus) * 0.1))
    selected_cpu = modern_cpus[0] if modern_cpus else None
    
    if selected_cpu:
        sample_build["CPU"] = selected_cpu
        price = selected_cpu.get("Price", 0)
        if price is None: price = 0
        remaining_budget -= price
    
    # Next, select a compatible motherboard
    if "CPU" in sample_build:
        motherboards = processed_data["Motherboard"]
        cpu_socket = sample_build["CPU"].get("Socket")
        
        # Filter motherboards by price and compatibility
        max_mobo_budget = remaining_budget * get_max_budget_allocation(budget).get("Motherboard", 0.3)
        min_mobo_budget = remaining_budget * get_dynamic_min_budget_allocation(budget).get("Motherboard", 0.08)
        affordable_mobos = [m for m in motherboards if m.get("Price", 0) is not None and m.get("Price", 0) <= max_mobo_budget and m.get("Price", 0) >= min_mobo_budget]
        
        # Filter for compatible motherboards
        compatible_mobos = []
        for mobo in affordable_mobos:
            # Check if sockets match
            if check_cpu_motherboard_compatibility(mobo, sample_build["CPU"]):
                compatible_mobos.append(mobo)
        
        # If no compatible motherboards found, try to find any with matching socket
        if not compatible_mobos:
            # Try to find motherboards with the same socket family
            cpu_socket_str = str(cpu_socket).lower() if cpu_socket else ""
            for mobo in affordable_mobos:
                mobo_socket = str(mobo.get("Socket/CPU", "")).lower()
                
                # Check if they share the same socket family
                for socket_family, variants in SOCKET_MAPPING.items():
                    socket_family_lower = socket_family.lower()
                    if socket_family_lower in cpu_socket_str and socket_family_lower in mobo_socket:
                        compatible_mobos.append(mobo)
                        break
        
        # If still no compatible motherboards, use any affordable ones
        if not compatible_mobos:
            compatible_mobos = affordable_mobos
        
        # Sort by price (descending) to get better components first
        compatible_mobos.sort(key=lambda m: m.get("Price", 0) if m.get("Price", 0) is not None else 0, reverse=True)
        
        # Select a motherboard from the top options
        top_n = max(1, int(len(compatible_mobos) * 0.2))
        selected_mobo = compatible_mobos[min(2, top_n-1) if top_n > 1 else 0] if compatible_mobos else None
        
        if selected_mobo:
            # Double-check compatibility
            if not check_cpu_motherboard_compatibility(selected_mobo, sample_build["CPU"]):
                # If not compatible, try to find a better match
                for mobo in compatible_mobos:
                    if check_cpu_motherboard_compatibility(mobo, sample_build["CPU"]):
                        selected_mobo = mobo
                        break
            
            sample_build["Motherboard"] = selected_mobo
            price = selected_mobo.get("Price", 0)
            if price is None: price = 0
            remaining_budget -= price
    
    # Verify CPU and motherboard compatibility
    if "CPU" in sample_build and "Motherboard" in sample_build:
        cpu = sample_build["CPU"]
        mobo = sample_build["Motherboard"]
        
        # If they're not compatible, start over with a different CPU
        if not check_cpu_motherboard_compatibility(mobo, cpu):
            print("Warning: Selected CPU and motherboard are not compatible. Reselecting components...")
            
            # Try to find a CPU that matches the motherboard
            mobo_socket = mobo.get("Socket/CPU")
            compatible_cpus = []
            
            for cpu in modern_cpus:
                if check_cpu_motherboard_compatibility(mobo, cpu):
                    compatible_cpus.append(cpu)
            
            if compatible_cpus:
                # Sort by value score
                for cpu in compatible_cpus:
                    cpu["value_score"] = calculate_cpu_value_score(cpu, budget)
                compatible_cpus.sort(key=lambda c: c.get("value_score", 0), reverse=True)
                
                # Replace CPU with compatible one
                selected_cpu = compatible_cpus[0]
                
                # Update build and budget
                old_price = sample_build["CPU"].get("Price", 0)
                if old_price is None: old_price = 0
                
                new_price = selected_cpu.get("Price", 0)
                if new_price is None: new_price = 0
                
                remaining_budget = remaining_budget + old_price - new_price
                sample_build["CPU"] = selected_cpu
            else:
                # If no compatible CPU found, try to find a compatible motherboard
                cpu = sample_build["CPU"]
                compatible_mobos = []
                
                for mobo in affordable_mobos:
                    if check_cpu_motherboard_compatibility(mobo, cpu):
                        compatible_mobos.append(mobo)
                
                if compatible_mobos:
                    # Sort by price
                    compatible_mobos.sort(key=lambda m: m.get("Price", 0) if m.get("Price", 0) is not None else 0, reverse=True)
                    
                    # Replace motherboard with compatible one
                    selected_mobo = compatible_mobos[0]
                    
                    # Update build and budget
                    old_price = sample_build["Motherboard"].get("Price", 0)
                    if old_price is None: old_price = 0
                    
                    new_price = selected_mobo.get("Price", 0)
                    if new_price is None: new_price = 0
                    
                    remaining_budget = remaining_budget + old_price - new_price
                    sample_build["Motherboard"] = selected_mobo
                else:
                    # If no compatible combination found, start over with different components
                    sample_build = {}
                    remaining_budget = budget
                    
                    # Try to find a known compatible pair
                    for cpu in modern_cpus[:10]:  # Try the top 10 CPUs
                        for mobo in affordable_mobos[:10]:  # Try the top 10 motherboards
                            if check_cpu_motherboard_compatibility(mobo, cpu):
                                sample_build["CPU"] = cpu
                                sample_build["Motherboard"] = mobo
                                
                                cpu_price = cpu.get("Price", 0)
                                if cpu_price is None: cpu_price = 0
                                
                                mobo_price = mobo.get("Price", 0)
                                if mobo_price is None: mobo_price = 0
                                
                                remaining_budget = budget - cpu_price - mobo_price
                                break
                        if "CPU" in sample_build:
                            break
    
    # Select remaining components in priority order
    for component_type in COMPONENT_PRIORITY:
        if component_type in ["CPU", "Motherboard", "GPU"] or component_type not in ESSENTIAL_COMPONENTS and component_type != "CPU Cooler":
            continue
            
        components = processed_data[component_type]
        
        # Filter components by price (must be affordable)
        max_component_budget = remaining_budget * get_max_budget_allocation(budget).get(component_type, 0.4)
        min_component_budget = remaining_budget * get_dynamic_min_budget_allocation(budget).get(component_type, 0.05)
        affordable_components = [c for c in components if c.get("Price", 0) is not None and c.get("Price", 0) <= max_component_budget and c.get("Price", 0) >= min_component_budget]
        
        if not affordable_components:
            continue
        
        # Sort by price (descending) to get better components first
        affordable_components.sort(key=lambda c: c.get("Price", 0) if c.get("Price", 0) is not None else 0, reverse=True)
        
        # Check compatibility with existing components
        compatible_components = []
        for component in affordable_components:
            is_compatible = True
            for existing_type, existing_component in sample_build.items():
                rule_key = (component_type, existing_type)
                reverse_rule_key = (existing_type, component_type)
                
                if rule_key in COMPATIBILITY_RULES:
                    if not COMPATIBILITY_RULES[rule_key](component, existing_component):
                        is_compatible = False
                        break
                elif reverse_rule_key in COMPATIBILITY_RULES:
                    if not COMPATIBILITY_RULES[reverse_rule_key](existing_component, component):
                        is_compatible = False
                        break
            
            if is_compatible:
                compatible_components.append(component)
        
        if not compatible_components:
            continue
        
        # Select a component from the top 10%
        top_n = max(1, int(len(compatible_components) * 0.1))
        selected_component = compatible_components[min(3, top_n-1) if top_n > 1 else 0]
        
        # Add to build and update remaining budget
        sample_build[component_type] = selected_component
        price = selected_component.get("Price", 0)
        if price is None: price = 0
        remaining_budget -= price
    
    # Final compatibility check
    if "CPU" in sample_build and "Motherboard" in sample_build:
        cpu = sample_build["CPU"]
        mobo = sample_build["Motherboard"]
        
        if not check_cpu_motherboard_compatibility(mobo, cpu):
            print("Error: Final CPU and motherboard are still incompatible!")
            # As a last resort, print the socket information
            print(f"CPU Socket: {cpu.get('Socket')}, Motherboard Socket: {mobo.get('Socket/CPU')}")
    
    return sample_build, budget

if __name__ == "__main__":
    # Set budget based on user requirements
    budget = 3000.0  # Higher budget for better components
    
    # Train agent
    print("Starting DQN training for PC building...")
    agent, scores = train_dqn(budget=budget, n_episodes=2000)
    
    # Validate agent
    print("\nValidating trained agent...")
    best_build, best_score = validate_agent(agent, budget=budget)
    
    # Print complete build details
    print_complete_build(best_build, budget)
    print(f"Total Score: {best_score:.2f}")
    
    # If the agent didn't select all essential components, show a sample complete build
    missing = [c for c in ESSENTIAL_COMPONENTS if c not in best_build]
    if missing:
        print("\n\nThe agent didn't select all essential components. Here's a sample complete build for demonstration:")
        sample_build, budget = create_sample_complete_build(budget)
        print_complete_build(sample_build, budget)
