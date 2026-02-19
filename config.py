import random
import numpy as np

# Random Seed Configuration
RANDOM_SEED = 42  # Set to None for non-reproducible results

# Graph Configuration
MIN_SIZE_OF_DIFFUSION = 20
#MIN_SIZE_OF_DIFFUSION_FACEBOOK = 10
#ZHAI_MAX_A_PRIME = 1000

# Simulation selection
RUN_SIMULATIONS_ON_REAL_GRAPHS = True
RUN_SIMULATIONS_ON_RANDOM_GRAPHS = False

# Method Selection
RUN_MCMC = True
RUN_NO_LOOP_METHOD = True
RUN_MAX_WEIGHT_ARBORESCENCE = True

# Evaluation Configuration
EVALUATE_EXACT_SOURCE = True
EVALUATE_TOP_3 = False
EVALUATE_DISTANCE = False 

# Logging Configuration
LOG_DIR = "logs"
ENABLE_DETAILED_LOGGING = True

def initialize_random_seed(seed=None):
    """Initialize random seed for reproducibility."""
    if seed is None:
        seed = RANDOM_SEED
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        return seed
    return None
