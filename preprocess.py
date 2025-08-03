import os
import pandas as pd
import re
import numpy as np

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

def extract_cpu_generation(cpu_name):
    """Extract CPU generation information from name."""
    if not cpu_name or not isinstance(cpu_name, str):
        return None
    
    cpu_name = cpu_name.lower()
    
    # Intel Core CPUs (e.g., i7-14700K)
    intel_match = re.search(r'i[3579]-(\d{1,2})\d{3}[a-z]*', cpu_name)
    if intel_match:
        gen = intel_match.group(1)
        return int(gen)
    
    # AMD Ryzen Desktop CPUs (e.g., Ryzen 7 7600X)
    amd_match = re.search(r'ryzen\s+[3579]\s+(\d)(\d{3})[a-z]*', cpu_name)
    if amd_match:
        gen = amd_match.group(1)
        return int(gen)
    
    return None

def extract_cpu_suffix(cpu_name):
    """Extract CPU suffix information from name."""
    if not cpu_name or not isinstance(cpu_name, str):
        return None
    
    cpu_name = cpu_name.lower()
    
    # Intel suffixes
    intel_match = re.search(r'i[3579]-\d{1,2}\d{3}([a-z]+)', cpu_name)
    if intel_match:
        return intel_match.group(1).upper()
    
    # AMD suffixes
    amd_match = re.search(r'ryzen\s+[3579]\s+\d{4}([a-z]+\d*)', cpu_name)
    if amd_match:
        return amd_match.group(1).upper()
    
    return None

def extract_cpu_tier(cpu_name):
    """Extract CPU tier information from name."""
    if not cpu_name or not isinstance(cpu_name, str):
        return None
    
    cpu_name = cpu_name.lower()
    
    # Intel tier (i3, i5, i7, i9)
    intel_match = re.search(r'i([3579])', cpu_name)
    if intel_match:
        return int(intel_match.group(1))
    
    # AMD tier (Ryzen 3, 5, 7, 9)
    amd_match = re.search(r'ryzen\s+([3579])', cpu_name)
    if amd_match:
        return int(amd_match.group(1))
    
    return None

def extract_gpu_memory_type(gpu_row):
    """Extract GPU memory type information."""
    if 'Memory Type' in gpu_row and not pd.isna(gpu_row['Memory Type']):
        return str(gpu_row['Memory Type']).upper()
    return None

def normalize_price(price_value):
    """Normalize price to float value."""
    if pd.isna(price_value) or price_value is None:
        return 0.0
    
    if isinstance(price_value, (int, float)):
        return float(price_value)
    
    # If it's a string, clean it
    if isinstance(price_value, str):
        price_str = price_value.strip().replace("$", "").replace(",", "")
        match = re.search(r"(\d+\.?\d*)", price_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
    
    return 0.0

def preprocess_cpu_data(cpu_df):
    """Preprocess CPU data to extract features for reward function."""
    # Create new columns for generation, suffix, and tier
    cpu_df['Generation'] = cpu_df['Name'].apply(extract_cpu_generation)
    cpu_df['Suffix'] = cpu_df['Name'].apply(extract_cpu_suffix)
    cpu_df['Tier'] = cpu_df['Name'].apply(extract_cpu_tier)
    
    # Determine CPU brand (Intel or AMD)
    cpu_df['Brand'] = cpu_df['Name'].apply(lambda x: 'Intel' if 'intel' in str(x).lower() else 'AMD' if 'amd' in str(x).lower() or 'ryzen' in str(x).lower() else 'Other')
    
    # Determine if CPU has P-cores and E-cores (Intel 12th Gen+)
    cpu_df['Has_P_E_Cores'] = (cpu_df['Efficiency Core Clock'].notna() & 
                              cpu_df['Performance Core Clock'].notna())
    
    # Normalize price to float
    if 'Price' in cpu_df.columns:
        cpu_df['Price'] = cpu_df['Price'].apply(normalize_price)
    
    # Convert DataFrame to list of dictionaries
    return cpu_df.to_dict('records')

def preprocess_gpu_data(gpu_df):
    """Preprocess GPU data to extract features for reward function."""
    # Extract memory type for reward multiplier
    gpu_df['Memory_Type_Clean'] = gpu_df.apply(extract_gpu_memory_type, axis=1)
    
    # Determine GPU brand (NVIDIA or AMD)
    gpu_df['Brand'] = gpu_df['Chipset'].apply(
        lambda x: 'NVIDIA' if any(nvidia_type in str(x).upper() for nvidia_type in ['GEFORCE', 'RTX', 'GTX']) 
        else 'AMD' if any(amd_type in str(x).upper() for amd_type in ['RADEON', 'RX']) 
        else 'Other'
    )
    
    # Normalize price to float
    if 'Price' in gpu_df.columns:
        gpu_df['Price'] = gpu_df['Price'].apply(normalize_price)
    
    # Convert DataFrame to list of dictionaries
    return gpu_df.to_dict('records')

def load_and_preprocess_all_data():
    """Load and preprocess all component data."""
    processed_data = {}
    
    # Load and preprocess CPU data
    cpu_df = pd.read_csv(CSV_FILE_PATHS["CPU"])
    processed_data["CPU"] = preprocess_cpu_data(cpu_df)
    
    # Load and preprocess GPU data
    gpu_df = pd.read_csv(CSV_FILE_PATHS["GPU"])
    processed_data["GPU"] = preprocess_gpu_data(gpu_df)
    
    # Load other component data
    for component, path in CSV_FILE_PATHS.items():
        if component not in ["CPU", "GPU"]:
            df = pd.read_csv(path)
            
            # Normalize price to float for all components
            if 'Price' in df.columns:
                df['Price'] = df['Price'].apply(normalize_price)
            
            # Convert all DataFrames to lists of dictionaries
            processed_data[component] = df.to_dict('records')
    
    # Verify all data is in list format, not DataFrame
    for component, data in processed_data.items():
        if not isinstance(data, list):
            print(f"Warning: {component} data is not a list, converting now")
            processed_data[component] = list(data)
    
    # Verify all prices are floats
    for component, data in processed_data.items():
        for item in data:
            if 'Price' in item and not isinstance(item['Price'], float):
                item['Price'] = normalize_price(item['Price'])
    
    return processed_data

def verify_data_types(processed_data):
    """Verify data types for critical fields."""
    print("Verifying data types for critical fields...")
    
    for component, data in processed_data.items():
        if not data:  # Skip empty lists
            print(f"  {component}: No data")
            continue
        
        # Check first item
        first_item = data[0]
        
        # Check price type
        if 'Price' in first_item:
            price_type = type(first_item['Price'])
            print(f"  {component} price type: {price_type}")
            if price_type is not float:
                print(f"    WARNING: {component} price is not float!")
        else:
            print(f"  {component}: No Price field")
        
        # Check a few random items
        import random
        sample_size = min(5, len(data))
        sample_indices = random.sample(range(len(data)), sample_size)
        
        for idx in sample_indices:
            item = data[idx]
            if 'Price' in item and not isinstance(item['Price'], float):
                print(f"    WARNING: {component}[{idx}] price is not float: {item['Price']} ({type(item['Price'])})")

if __name__ == "__main__":
    # Test preprocessing
    processed_data = load_and_preprocess_all_data()
    
    # Verify data types
    verify_data_types(processed_data)
    
    # Print sample of processed CPU data
    print("\nSample of processed CPU data:")
    print(f"Type: {type(processed_data['CPU'])}")
    print(f"First item type: {type(processed_data['CPU'][0])}")
    print(f"First item: {processed_data['CPU'][0]}")
    
    # Print sample of processed GPU data
    print("\nSample of processed GPU data:")
    print(f"Type: {type(processed_data['GPU'])}")
    print(f"First item type: {type(processed_data['GPU'][0])}")
    print(f"First item: {processed_data['GPU'][0]}")
