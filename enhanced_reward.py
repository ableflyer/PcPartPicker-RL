import re
import numpy as np
import pandas as pd

# --- CPU Generation Reward Mapping ---
CPU_GENERATION_REWARDS = {
    # Intel generations
    14: 10.0,  # Intel 14th Gen
    13: 8.0,   # Intel 13th Gen
    12: 6.0,   # Intel 12th Gen
    11: 4.0,   # Intel 11th Gen
    10: 3.0,   # Intel 10th Gen
    9: 2.0,    # Intel 9th Gen
    8: 1.0,    # Intel 8th Gen
    # AMD generations (Ryzen)
    7: 10.0,   # AMD Zen 4 (7000-series)
    5: 6.0,    # AMD Zen 3 (5000-series)
    3: 3.0,    # AMD Zen 2 (3000-series)
}

# --- CPU Tier Bonus Mapping ---
CPU_TIER_BONUS = {
    # Intel tiers
    9: 2.5,    # i9
    7: 1.5,    # i7
    5: 0.8,    # i5
    3: 0.3,    # i3
    # AMD tiers (same values)
    # 9: 2.5,  # Ryzen 9
    # 7: 1.5,  # Ryzen 7
    # 5: 0.8,  # Ryzen 5
    # 3: 0.3,  # Ryzen 3
}

# --- CPU Suffix Bonus Mapping ---
CPU_SUFFIX_BONUS = {
    # Intel suffixes
    "K": 1.5,     # Unlocked for overclocking
    "KF": 1.2,    # Unlocked, no integrated graphics
    "KS": 2.0,    # Special edition, high performance
    "F": 0.5,     # No integrated graphics
    # AMD suffixes
    "X": 1.5,     # High performance
    "XT": 1.8,    # Enhanced X series
    "X3D": 3.0,   # With 3D V-Cache, significant gaming boost
    "G": 1.0,     # Integrated graphics
}

# --- GPU Memory Type Multiplier ---
GPU_MEMORY_TYPE_MULTIPLIER = {
    "GDDR6X": 1.5,  # Highest multiplier for performance
    "HBM2": 1.8,    # Often for high-end professional cards
    "GDDR6": 1.5,   # Standard for modern GPUs
    "GDDR5X": 1.1,  # Slightly better than GDDR5
    "GDDR5": 1.0,   # Base multiplier
}

def calculate_cpu_generation_reward(cpu_data):
    """Calculate reward based on CPU generation."""
    generation = cpu_data.get('Generation')
    
    # Handle None, NaN, or invalid values
    if generation is None or (isinstance(generation, float) and pd.isna(generation)):
        return 0.0
    
    try:
        # Convert to int if it's a float
        if isinstance(generation, float):
            generation = int(generation)
        elif isinstance(generation, str):
            generation = int(float(generation))
    except (ValueError, TypeError):
        # If conversion fails, return default reward
        return 0.0
    
    return CPU_GENERATION_REWARDS.get(generation, 0.0)

def calculate_cpu_tier_bonus(cpu_data):
    """Calculate bonus based on CPU tier (i3/i5/i7/i9 or Ryzen 3/5/7/9)."""
    tier = cpu_data.get('Tier')
    
    # Handle None, NaN, or invalid values
    if tier is None or (isinstance(tier, float) and pd.isna(tier)):
        return 0.0
    
    try:
        # Convert to int if it's a float
        if isinstance(tier, float):
            tier = int(tier)
        elif isinstance(tier, str):
            tier = int(float(tier))
    except (ValueError, TypeError):
        # If conversion fails, return default bonus
        return 0.0
    
    return CPU_TIER_BONUS.get(tier, 0.0)

def calculate_cpu_suffix_bonus(cpu_data):
    """Calculate bonus based on CPU suffix."""
    suffix = cpu_data.get('Suffix')
    
    # Handle None, NaN, or invalid values
    if suffix is None or suffix == 'None' or (isinstance(suffix, float) and pd.isna(suffix)):
        return 0.0
    
    # Ensure suffix is a string
    if not isinstance(suffix, str):
        try:
            suffix = str(suffix)
        except:
            return 0.0
    
    return CPU_SUFFIX_BONUS.get(suffix, 0.0)

def calculate_cpu_hybrid_architecture_bonus(cpu_data):
    """Calculate bonus for Intel hybrid architecture (P-cores/E-cores)."""
    has_p_e_cores = cpu_data.get('Has_P_E_Cores')
    
    # Handle None, NaN, or invalid values
    if has_p_e_cores is None or (isinstance(has_p_e_cores, float) and pd.isna(has_p_e_cores)):
        return 0.0
    
    # Convert to boolean if needed
    if isinstance(has_p_e_cores, str):
        has_p_e_cores = has_p_e_cores.lower() in ('true', 'yes', '1')
    
    return 2.0 if has_p_e_cores else 0.0

def calculate_gpu_memory_type_multiplier(gpu_data):
    """Calculate multiplier based on GPU memory type."""
    memory_type = gpu_data.get('Memory_Type_Clean')
    
    # Handle None, NaN, or invalid values
    if memory_type is None or memory_type == 'None' or (isinstance(memory_type, float) and pd.isna(memory_type)):
        return 1.0  # Default multiplier
    
    # Ensure memory_type is a string
    if not isinstance(memory_type, str):
        try:
            memory_type = str(memory_type)
        except:
            return 1.0
    
    return GPU_MEMORY_TYPE_MULTIPLIER.get(memory_type, 0.8)  # 0.8 for older/other types

def safe_float_conversion(value, default=0.0):
    """Safely convert a value to float with fallback to default."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    
    if isinstance(value, str):
        # Remove any non-numeric characters except decimal point
        value = re.sub(r'[^\d.]', '', value)
        if not value:
            return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_enhanced_cpu_reward(cpu_data):
    """Calculate the enhanced CPU reward incorporating all factors."""
    # Handle None case
    if cpu_data is None:
        return 0.0
    
    # Base reward (can be from existing logic)
    base_reward = 0.0
    
    # Core count reward (if available)
    if 'Core Count' in cpu_data and cpu_data['Core Count'] is not None and cpu_data['Core Count'] >= 6:
        core_count = safe_float_conversion(cpu_data['Core Count'])
        base_reward += core_count * 2  # Adjust multiplier as needed
    elif 'Core Count' in cpu_data and cpu_data['Core Count'] is not None and cpu_data['Core Count'] < 6:
        base_reward -= 100
    
    # Clock speed reward (if available)
    if 'Performance Core Boost Clock' in cpu_data and cpu_data['Performance Core Boost Clock'] is not None:
        try:
            boost_clock_str = str(cpu_data['Performance Core Boost Clock'])
            boost_clock = safe_float_conversion(boost_clock_str.replace('GHz', '').strip())
            base_reward += boost_clock * 1.0  # Adjust multiplier as needed
        except (ValueError, TypeError, AttributeError):
            pass  # Skip if conversion fails
    
    # Add generation reward
    generation_reward = calculate_cpu_generation_reward(cpu_data)
    
    # Add tier bonus
    tier_bonus = calculate_cpu_tier_bonus(cpu_data)
    
    # Add suffix bonus
    suffix_bonus = calculate_cpu_suffix_bonus(cpu_data)
    
    # Add hybrid architecture bonus
    hybrid_bonus = calculate_cpu_hybrid_architecture_bonus(cpu_data)

    # Penalize Xeon CPUs
    xeon_penalty = 0.0
    cpu_name = cpu_data.get('Name') or cpu_data.get('Model')
    if cpu_name and isinstance(cpu_name, str) and 'xeon' in cpu_name.lower():
        xeon_penalty = -200  # Strong penalty to discourage Xeon selection

    # Combine all rewards
    total_reward = base_reward + generation_reward + tier_bonus + suffix_bonus + hybrid_bonus + xeon_penalty
    
    return total_reward

def calculate_enhanced_gpu_reward(gpu_data):
    """Calculate the enhanced GPU reward incorporating all factors."""
    # Handle None case
    if gpu_data is None:
        return 0.0
    
    # Base reward (can be from existing logic)
    base_reward = 0.0
    
    # VRAM reward (if available)
    if 'Memory' in gpu_data and gpu_data['Memory'] is not None:
        memory = safe_float_conversion(gpu_data['Memory'])
        base_reward += memory * 1.0  # Adjust multiplier as needed
    
    # Clock speed reward (if available)
    if 'Boost Clock' in gpu_data and gpu_data['Boost Clock'] is not None:
        try:
            boost_clock_str = str(gpu_data['Boost Clock'])
            boost_clock = safe_float_conversion(boost_clock_str.replace('MHz', '').strip()) / 1000  # Convert to GHz
            base_reward += boost_clock * 2.0  # Adjust multiplier as needed
        except (ValueError, TypeError, AttributeError):
            pass  # Skip if conversion fails
    
    # Apply memory type multiplier
    memory_type_multiplier = calculate_gpu_memory_type_multiplier(gpu_data)
    
    # Apply multiplier to the base reward
    total_reward = base_reward * memory_type_multiplier
    
    return total_reward

def calculate_total_reward(build_state):
    """Calculate the total reward for a PC build state."""
    # Handle None case
    if build_state is None:
        return 0.0
    
    total_reward = 0.0
    
    # Get CPU and GPU from build state
    cpu_selected = build_state.get('CPU')
    gpu_selected = build_state.get('GPU')
    
    # Calculate CPU reward if CPU is selected
    if cpu_selected is not None:
        cpu_reward = calculate_enhanced_cpu_reward(cpu_selected)
        total_reward += cpu_reward
    
    # Calculate GPU reward if GPU is selected
    if gpu_selected is not None:
        gpu_reward = calculate_enhanced_gpu_reward(gpu_selected)
        total_reward += gpu_reward
    
    # Add other component rewards and compatibility checks as needed
    # ...
    
    return total_reward

# Test function
def test_reward_calculation(cpu_data, gpu_data):
    """Test the reward calculation with sample data."""
    print("CPU Reward Components:")
    print(f"  Generation Reward: {calculate_cpu_generation_reward(cpu_data)}")
    print(f"  Tier Bonus: {calculate_cpu_tier_bonus(cpu_data)}")
    print(f"  Suffix Bonus: {calculate_cpu_suffix_bonus(cpu_data)}")
    print(f"  Hybrid Architecture Bonus: {calculate_cpu_hybrid_architecture_bonus(cpu_data)}")
    print(f"  Total CPU Reward: {calculate_enhanced_cpu_reward(cpu_data)}")
    
    print("\nGPU Reward Components:")
    print(f"  Memory Type Multiplier: {calculate_gpu_memory_type_multiplier(gpu_data)}")
    print(f"  Total GPU Reward: {calculate_enhanced_gpu_reward(gpu_data)}")
    
    # Create a mock build state
    build_state = {
        'CPU': cpu_data,
        'GPU': gpu_data
    }
    
    print(f"\nTotal Build Reward: {calculate_total_reward(build_state)}")
