import time
from Real_Graphs import *
from Random_Graphs import *
from K_Sources_Real import *
from K_Sources_Random import *



def main():
    begin_time = time.time()
    
    # Initialize random seed for reproducibility
    #seed = initialize_random_seed(RANDOM_SEED)
    
    seed = 0
    # Seed=0 for regular running.
    # Change it to the iteration number for resuming an interrupted experiment.
    print(f"Random seed initialized: {seed}")

    run_random_graphs(base_seed=seed)
    run_real_graphs(base_seed=seed)
    #run_k_sources_all_methods_on_Random()
    #run_k_sources_all_methods_on_Real()

    total_time = time.time() - begin_time
    print(f"\nTotal time elapsed: {total_time:.2f} seconds")


if __name__ == '__main__':
    main()
