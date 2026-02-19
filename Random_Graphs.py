from datetime import datetime
import time
import random

# my models:
from Graph_Generator import *
from Independent_Cascade import *
from Markov_Chains import *
from Zhai_MCMC import infer_source_zhai_mcmc, rank_sources_zhai_mcmc
from logging_util import SimulationLogger

# ----------------------------- MCMC params (Zhai et al., 2015) -----------------------------
# The MCMC algorithm is significantly more expensive than the Markov-chain approach
ZHAI_K = 5_000
ZHAI_BURN_IN = 1_000
ZHAI_THINNING = 1
ZHAI_MAX_CANDIDATES_PER_SAMPLE = None


"""Random-graph experiment runner.
"""


def Append_to_file(file_name, text):
    print(file_name, ":", text)
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def run_random_graphs(base_seed):
    

    random_graphs = [
        ("G1", 500, 0.1, 0.0416),
        ("G2", 1000, 0.1, 0.0204),
        ("G3", 2000, 0.1, 0.0101),
        ("G4", 3000, 0.1, 0.0071),
        ("G5", 4000, 0.1, 0.0052),
        ("G6", 5000, 0.1, 0.0041),
        ("G7", 500, 0.0416, 0.1),
        ("G8", 1000, 0.02, 0.1),
        ("G9", 2000, 0.0101, 0.1),
        ("G10", 3000, 0.0067, 0.1),
        ("G11", 4000, 0.0052, 0.1),
        ("G12", 5000, 0.0041, 0.1),
        ("G13", 10000, 0.002, 0.1),
        ("G14", 15000, 0.0013, 0.1)
    ]

    

    min_size_of_diffusion = 20
    max_diffusions_per_graph = 1000
    print("starting")

    for (graph_name, n_nodes, p_edge, p_weight) in random_graphs:
        # Initialize success counters
        naive_num_of_successes = 0
        no_loop_num_of_successes = 0
        #self_loop_num_of_successes = 0
        max_arbo_num_of_successes = 0
        mcmc_num_of_successes = 0

        naive_top3_successes = 0
        no_loop_top3_successes = 0
        #self_loop_top3_successes = 0
        max_arbo_top3_successes = 0
        mcmc_top3_successes = 0

        naive_near_successes = 0
        no_loop_near_successes = 0
        #self_loop_near_successes = 0
        max_arbo_near_successes = 0
        mcmc_near_successes = 0

        num_of_too_small_diffusion = 0
        num_of_too_small_A_tag = 0
        num_of_total_diffusion_calculated = 0  # without small diffusion and small A'

        begin_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{graph_name}_{timestamp}"
        logger = SimulationLogger(exp_name)

        t = time.time()
        G = random_graph_generator(n_nodes, p_edge, p_weight, base_seed)
        tot = time.time() - t
        print("finished generating ", graph_name, "time:", tot)

        output_file_name = f"{exp_name}.txt"
        iteration_number = 0

        while num_of_total_diffusion_calculated < max_diffusions_per_graph:
            iteration_number += 1
            print(f"Starting new iteration: {iteration_number}")
            time_of_diffusion = time.time()
            

            random.seed(base_seed + iteration_number)
            source_node = random.choice(list(G.nodes()))
            #print("The source node is: ", source_node)
            #print("Running the ic model")
            infected_nodes = simulate_ic_model(G, source_node, max_iterations=len(G.nodes), seed=base_seed + iteration_number)

            if len(infected_nodes) < min_size_of_diffusion:
                print("Too small diffusion len(active set)= ", len(infected_nodes))
                #print("The infected nodes are: ", infected_nodes)
                num_of_too_small_diffusion += 1
                continue

            #print(f"Number of the infected nodes is: {len(infected_nodes)} and number of edges is: "
            #      f"{len(G.subgraph(infected_nodes).edges)}")
            infected_graph = create_induced_subgraph(G, infected_nodes)
            #print("induced subgraph was created")
            possible_sources = Atag_calc(infected_graph)

            if len(possible_sources) <= 1:  # if A' is smaller then 2
                print("too small A' ")
                num_of_too_small_A_tag += 1
                continue

            #print(f"Number of possible sources of infection: {len(possible_sources)}")
            #print(f"source_node in possible_sources: {source_node in possible_sources}")
            print("creating induced subgraph")
            induced_graph = create_induced_subgraph(G, possible_sources)
            print("true source node is: ", source_node)

            t = time.time()
            reversed_G = reverse_and_normalize_weights(induced_graph)
            print("finished naive method")
			#self_loops_G = apply_self_loop_method(induced_graph)
			#print("finished self loop method")
            no_loops_G = reverse_and_normalize_weights(induced_graph)
            print("finished no loop method")
            max_arbo_weights = Max_weight_arborescence(induced_graph)
            print("finished max weight arborescence method")



            # Run evaluations
            # ******************************* exact match *************************************************************
            exact_match_results = evaluate_exact_match(source_node, reversed_G, no_loops_G,
                                                       max_arbo_weights, induced_graph)
            tot = time.time() - t
            logger.log_simulation(
                random_seed=base_seed + iteration_number,
                source_node=source_node,
                size_A=len(infected_nodes),
                size_A_prime=len(possible_sources),
                method="naive",
                running_time=tot/2,
                success=exact_match_results[0],
                rank_of_true_source="n/a",
                additional_info=""
            )
            logger.log_simulation(
                random_seed=base_seed + iteration_number,
                source_node=source_node,
                size_A=len(infected_nodes),
                size_A_prime=len(possible_sources),
                method="no_loops",
                running_time=tot/2,
                success=exact_match_results[1],
                rank_of_true_source="n/a",
                additional_info=""
            )
            logger.log_simulation(
                random_seed=base_seed + iteration_number,
                source_node=source_node,
                size_A=len(infected_nodes),
                size_A_prime=len(possible_sources),
                method="max_weight_arborescence",
                running_time=tot/2,
                success=exact_match_results[2],
                rank_of_true_source="n/a",
                additional_info=""
            )
            
            t = time.time()
            tau = len(infected_nodes)
            zhai_pred = infer_source_zhai_mcmc(
                        G,
                        infected_nodes,
                        K=ZHAI_K,
                        tau_range=(tau, tau),
                        burn_in=ZHAI_BURN_IN,
                        sample_every=ZHAI_THINNING,
                        max_candidates_per_sample=ZHAI_MAX_CANDIDATES_PER_SAMPLE,
                        seed=base_seed + iteration_number
                    )
            tot = time.time() - t
            print("finished MCMC method")
            print("MCMC's prediction: ", zhai_pred)
            logger.log_simulation(
                random_seed=base_seed + iteration_number,
                source_node=source_node,
                size_A=len(infected_nodes),
                size_A_prime=len(possible_sources),
                method="mcmc",
                running_time=tot,
                success=int(zhai_pred == source_node),
                rank_of_true_source="n/a",
                additional_info=""
            )

            mcmc_num_of_successes += int(zhai_pred == source_node)
            naive_num_of_successes += exact_match_results[0]
            no_loop_num_of_successes += exact_match_results[1]
            #self_loop_num_of_successes += exact_match_results[2]
            max_arbo_num_of_successes += exact_match_results[2]

            #print("finished the exact match method")

            # ************************************** top 3 ************************************************************
            #top3_results = evaluate_top3(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights, induced_graph)
            #zhai_top3 = rank_sources_zhai_mcmc(
            #            G,
            #            infected_nodes,
            #            K=ZHAI_K,
            #            tau_range=(tau, tau),
            #            burn_in=ZHAI_BURN_IN,
            #            sample_every=ZHAI_THINNING,
            #            max_candidates_per_sample=ZHAI_MAX_CANDIDATES_PER_SAMPLE,
            #            seed=seed,
            #            top_k=3
            #        )
            #mcmc_top3_successes += int(source_node in zhai_top3)
            #naive_top3_successes += top3_results[0]
            #no_loop_top3_successes += top3_results[1]
            #self_loop_top3_successes += top3_results[2]
            #max_arbo_top3_successes += top3_results[3]
            #
            #print("finished the top 3 method")

            # ************************************ near source ********************************************************
            #near_results = evaluate_near_source(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights,
            #                                    induced_graph)
            #dist = nx.shortest_path_length(G.to_undirected(), source=source_node, target=zhai_pred)
            #mcmc_near_successes += int(dist <= 3)
            #naive_near_successes += near_results[0]
            #no_loop_near_successes += near_results[1]
            #self_loop_near_successes += near_results[2]
            #max_arbo_near_successes += near_results[3]
			#
            #print("finished the near source method")
            num_of_total_diffusion_calculated += 1
            print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")
            tot = time.time() - time_of_diffusion
            print("Time it took to calculate the diffusion is:", tot)

        total_time = time.time() - begin_time
        Append_to_file(output_file_name,
                       f"Results for graph {graph_name} (total good diffusions: {num_of_total_diffusion_calculated})\n")

        Append_to_file(output_file_name, "--- Evaluation 1: Exact match successes ---")
        Append_to_file(output_file_name, f"Naive successes: {naive_num_of_successes}")
        Append_to_file(output_file_name, f"No loop successes: {no_loop_num_of_successes}")
        #Append_to_file(output_file_name, f"Self loop successes: {self_loop_num_of_successes}")
        Append_to_file(output_file_name, f"Max arborescence successes: {max_arbo_num_of_successes}")
        Append_to_file(output_file_name, f"MCMC successes: {mcmc_num_of_successes}\n")

        """"
        Append_to_file(output_file_name, "--- Evaluation 2: Source in top-3 successes ---")
        Append_to_file(output_file_name, f"Naive top-3 successes: {naive_top3_successes}")
        Append_to_file(output_file_name, f"No loop top-3 successes: {no_loop_top3_successes}")
        Append_to_file(output_file_name, f"Self loop top-3 successes: {self_loop_top3_successes}")
        Append_to_file(output_file_name, f"Max arborescence top-3 successes: {max_arbo_top3_successes}")
        Append_to_file(output_file_name, f"MCMC top-3 successes: {mcmc_top3_successes}\n")

        Append_to_file(output_file_name, "--- Evaluation 3: Within 3-hop neighborhood successes ---")
        Append_to_file(output_file_name, f"Naive near-source successes: {naive_near_successes}")
        Append_to_file(output_file_name, f"No loop near-source successes: {no_loop_near_successes}")
        Append_to_file(output_file_name, f"Self loop near-source successes: {self_loop_near_successes}")
        Append_to_file(output_file_name, f"Max arborescence near-source successes: {max_arbo_near_successes}\n")
        Append_to_file(output_file_name, f"MCMC near-source successes: {mcmc_near_successes}")
        """

        Append_to_file(output_file_name, f"Number of too small diffusions: {num_of_too_small_diffusion}")
        Append_to_file(output_file_name, f"Number of too small A': {num_of_too_small_A_tag}")
        Append_to_file(output_file_name, f"Total time elapsed: {total_time:.2f} seconds")

        Append_to_file(output_file_name, "____________________________\n")

def evaluate_exact_match(source_node, reversed_G, no_loops_G, max_arbo_weights, induced_graph):
    # Find the most probable source for each method
    naive_node, _ = find_most_probable_source(reversed_G)
    print("naive's prediction: ", naive_node)
    no_loop_node, no_loop_max_prob = find_most_probable_source_no_loop(no_loops_G, induced_graph)
    print("no loops's prediction: ", no_loop_node)
    max_arbo_node = max(max_arbo_weights, key=max_arbo_weights.get)
    print("max arbo's prediction: ", max_arbo_node)
    max_arbo_node = max(max_arbo_weights, key=max_arbo_weights.get)
    #self_loop_node, _ = find_most_probable_source(self_loops_G)

    naive_success = int(source_node == naive_node)
    no_loop_success = int(source_node == no_loop_node)
    max_arbo_success = int(source_node == max_arbo_node)
    #self_loop_success = int(source_node == self_loop_node)

    return naive_success, no_loop_success, max_arbo_success

# Get top 3 most probable sources for each method
def evaluate_top3(source_node, reversed_G, no_loop_G, self_loops_G, max_arbo_weights, induced_graph):
    
    naive_top3 = list(find_top_three(reversed_G))
    no_loop_top3 = list(find_top_three_no_loops(no_loop_G, induced_graph))
    self_loop_top3 = list(find_top_three(self_loops_G))
    max_arbo_top3 = sorted(max_arbo_weights, key=max_arbo_weights.get, reverse=True)[:3]

    naive_top3_success = int(source_node in naive_top3)
    no_loop_top3_success = int(source_node in no_loop_top3)
    self_loop_top3_success = int(source_node in self_loop_top3)
    max_arbo_top3_success = int(source_node in max_arbo_top3)

    return naive_top3_success, no_loop_top3_success, self_loop_top3_success, max_arbo_top3_success

# Find if the most probable node is near up to 3 steps from the true source
def evaluate_near_source(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights, induced_graph):
    naive_near_success = int(is_most_probable_near_source(reversed_G, source_node))
    no_loop_near_success = int(is_most_probable_near_source_no_loop(no_loops_G, induced_graph, source_node))
    self_loop_near_success = int(is_most_probable_near_source(self_loops_G, source_node))
    max_arbo_near_success = int(is_most_probable_near_source_max_arbo(max_arbo_weights, induced_graph, source_node))

    return naive_near_success, no_loop_near_success, self_loop_near_success, max_arbo_near_success
