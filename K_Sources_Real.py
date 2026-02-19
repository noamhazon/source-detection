from collections import Counter

from Graph_Generator import *
from Independent_Cascade import *
from Markov_Chains import *
from K_Sources import *
import time
import random


def Append_to_file(file_name, text):
    print(file_name, ":", text)
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def run_k_sources_all_methods_on_Real():
    real_graphs = [
        # (r"real_graphs\advogato\out.advogato", "advogato", " ", 0.257),  # done
        # (r"real_graphs\digg-friends\out.digg-friends", "digg", " ", 0.694),  # finished running on the server
        (r"real_graphs\soc-Epinions1\out.soc-Epinions1", "epinion_trust", "\t", 0.299),  # need to run on server
        # (r"real_graphs\facebook-wosn-links\out.facebook-wosn-links", "facebook_friendships", " ", 0.157), # need to verify
        # (r"real_graphs\ego-gplus\out.ego-gplus", "google_plus", "\t", 1),  # done
        # (r"real_graphs\slashdot-threads\out.slashdot-threads", "slashdot", " ", 0.726),  # done
        # (r"real_graphs\ego-twitter\out.ego-twitter", "twitter", "\t", 1),  # finished running on the server
        # (r"real_graphs\youtube-links\out.youtube-links" , "youtube_links" , " " , 0.461) ,  # need to run on server
    ]

    min_size_of_diffusion = 20
    max_size_of_graph = 5000

    for (path, graph_name, sep_char, P_range) in real_graphs:  # (path, graph_name, sep_char, P_range)
        file_name = f"{graph_name}_k.txt"
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(f"Results for graph {graph_name} with multiple sources\n")
            f.write("===============================================\n\n")

        for k in [2,3,5]:
            Append_to_file(file_name, f"\n=== Results for K = {k} ===\n")

            Append_to_file(file_name, "\n--- Method 1: Top K Sources ---")
            Find_Top_K_Sources_Real(path, graph_name, sep_char, P_range, k, file_name)

            Append_to_file(file_name, "\n--- Method 2: Run K Times ---")
            Find_Source_K_Times(path, graph_name, sep_char, P_range, k, file_name)

            Append_to_file(file_name, "\n--- Method 3: K Strongest ---")
            Find_K_strongest(path, graph_name, sep_char, P_range, k, file_name)



# ****** Method 1 - Top K-Sources ******

def Find_Top_K_Sources_Real(path, graph_name, sep_char, P_range, k, file_name):
    begin_time = time.time()
    exact_total_no_loops = 0
    exact_total_max_arbo = 0
    recall_total_no_loops = 0
    recall_total_max_arbo = 0
    precision_total_no_loops = 0
    precision_total_max_arbo = 0
    distance_total_no_loops = 0
    distance_total_max_arbo = 0
    num_of_total_diffusion_calculated = 0  # without small diffusion and small A'
    print("Starting")

    t = time.time()
    G = read_graph_from_file_Prange(path, graph_name, sep_char, P_range)
    tot = time.time() - t
    print("finished reading ", graph_name, "time:", tot)


    num_of_too_small_diffusion = 0
    num_of_too_small_A_tag = 0
    min_size_of_diffusion = 20
    max_size_of_graph = 10000

    if graph_name in ["facebook_friendships", "facebook_nips", "twitter"]:  # added facebook_friendships
        min_size_of_diffusion = 10

    while num_of_total_diffusion_calculated < 1000:
        time_of_diffusion = time.time()
        G = reduce_graph_size(G, max_size_of_graph)

        source_nodes = random.sample(list(G.nodes()), k)
        print("The source nodes are: ", source_nodes)
        print("Running the ic model")
        infected_nodes = simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=len(G.nodes))

        if len(infected_nodes) < min_size_of_diffusion:  # if the diffusion is bigger then 20
            print("Too small diffusion len(active set)= ", len(infected_nodes))
            num_of_too_small_diffusion += 1
            continue  # check the next graph

        print(f"Number of the infected nodes is: {len(infected_nodes)}")
        infected_graph = create_induced_subgraph(G, infected_nodes)
        possible_sources = Atag_calc(infected_graph)

        if len(possible_sources) <= 1:  # if A' is smaller then 2
            num_of_too_small_A_tag += 1
            continue  # check the next diffusion

        print(f"Number of possible sources of infection: {len(possible_sources)}")
        induced_graph = create_induced_subgraph(G, possible_sources)

        no_loops_G = reverse_and_normalize_weights(induced_graph)
        Max_weight_arborescence_G = Max_weight_arborescence(induced_graph)

        if not verify_no_loops_transformation(induced_graph, no_loops_G):
            print(f"Did the no loops method work? false")

        no_loop_most_probable_nodes, no_loop_max_probs = find_K_most_probable_sources_no_loop(no_loops_G, induced_graph, k)
        top_k_arbo_nodes = sorted(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get, reverse=True)[:k]


        print(f"The real sources are nodes {source_nodes}")

        # *************************************** exact match *********************************************************
        exact_match_results = evaluate_exact_match(source_nodes, no_loop_most_probable_nodes, top_k_arbo_nodes)
        exact_total_no_loops += exact_match_results[0]
        exact_total_max_arbo += exact_match_results[1]

        # ***************************************** recall ************************************************************
        recall_results = evaluate_recall(source_nodes, no_loop_most_probable_nodes, top_k_arbo_nodes)
        recall_total_no_loops += recall_results[0]
        recall_total_max_arbo += recall_results[1]

        # *************************************** precision ***********************************************************
        precision_results = evaluate_precision(source_nodes, no_loop_most_probable_nodes, top_k_arbo_nodes)
        precision_total_no_loops += precision_results[0]
        precision_total_max_arbo += precision_results[1]

        # *************************************** distance ************************************************************
        distance_results = evaluate_by_distance(G, source_nodes, no_loop_most_probable_nodes, top_k_arbo_nodes)
        distance_total_no_loops += distance_results[0]
        distance_total_max_arbo += distance_results[1]

        num_of_total_diffusion_calculated += 1
        print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")
        tot = time.time() - time_of_diffusion
        print("time it took to calculate the diffusion is:", tot)

    total_time = time.time() - begin_time

    Append_to_file(file_name,f"Results for graph {graph_name} (total good diffusions: {num_of_total_diffusion_calculated})\n")
    Append_to_file(file_name, "--- Evaluation 1: Exact match successes ---")
    Append_to_file(file_name, f"No loop successes: {exact_total_no_loops}")
    Append_to_file(file_name, f"Max arborescence successes: {exact_total_max_arbo}\n")

    Append_to_file(file_name, "--- Evaluation 2: Recall percents ---")
    Append_to_file(file_name, f"No loop Recall: {recall_total_no_loops / 1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence Recall: {recall_total_max_arbo / 1000:.2f}%\n")

    Append_to_file(file_name, "--- Evaluation 3: Precision percents ---")
    Append_to_file(file_name, f"No loop precision: {precision_total_no_loops /1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence precision: {precision_total_max_arbo /1000:.2f}%\n")

    Append_to_file(file_name, "--- Evaluation 4: Distance-Based Precision percents ---")
    Append_to_file(file_name, f"No loop distance precision: {distance_total_no_loops /1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence distance precision: {distance_total_max_arbo /1000:.2f}%\n")

    Append_to_file(file_name, f"Number of too small diffusions: {num_of_too_small_diffusion}")
    Append_to_file(file_name, f"Number of too small A': {num_of_too_small_A_tag}")
    Append_to_file(file_name, f"Total time elapsed: {total_time:.2f} seconds")
    Append_to_file(file_name, "____________________________\n")


# ****** Method 2 - K Times ******
def Find_Source_K_Times(path, graph_name, sep_char, P_range, k, file_name):
    begin_time = time.time()
    exact_total_no_loops = 0
    exact_total_max_arbo = 0
    recall_total_no_loops = 0
    recall_total_max_arbo = 0
    precision_total_no_loops = 0
    precision_total_max_arbo = 0
    distance_total_no_loops = 0
    distance_total_max_arbo = 0
    num_of_total_diffusion_calculated = 0  # without small diffusion and small A'
    print("Starting")
    t = time.time()
    G = read_graph_from_file_Prange(path, graph_name, sep_char, P_range)
    tot = time.time() - t
    print("finished reading ", graph_name, "time:", tot)


    num_of_too_small_diffusion = 0
    num_of_too_small_A_tag = 0
    min_size_of_diffusion = 20
    max_size_of_graph = 10000

    while num_of_total_diffusion_calculated < 1000:
        time_of_diffusion = time.time()
        G = reduce_graph_size(G, max_size_of_graph)

        source_nodes = random.sample(list(G.nodes()), k)
        print("The source nodes are: ", source_nodes)
        print("Running the ic model")
        infected_nodes = simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=len(G.nodes))

        if len(infected_nodes) < min_size_of_diffusion:  # if the diffusion is bigger then 20
            print("Too small diffusion len(active set)= ", len(infected_nodes))
            num_of_too_small_diffusion += 1
            continue  # check the next diffusion

        print(f"Number of the infected nodes is: {len(infected_nodes)}")
        infected_graph = create_induced_subgraph(G, infected_nodes)
        possible_sources = Atag_calc(infected_graph)

        if len(possible_sources) <= 1:  # if A' is smaller then 2
            num_of_too_small_A_tag += 1
            continue  # check the next diffusion

        print(f"Number of possible sources of infection: {len(possible_sources)}")
        induced_graph = create_induced_subgraph(G, possible_sources)

        no_loops_G = reverse_and_normalize_weights(induced_graph)
        Max_weight_arborescence_G = Max_weight_arborescence(induced_graph)

        if not verify_no_loops_transformation(induced_graph, no_loops_G):
            print(f"Did the no loops method work? false")

        remaining_infected_nodes = set(infected_nodes)

        no_loop_k_most_probable_nodes = []

        for i in range(k):
            induced_graph = create_induced_subgraph(G, remaining_infected_nodes)
            possible_sources = Atag_calc(induced_graph)
            if len(possible_sources) <= 1:
                print("Too few possible sources after removal.")
                break

            induced_subgraph_sources = create_induced_subgraph(G, possible_sources)
            no_loops_G = reverse_and_normalize_weights(induced_subgraph_sources)

            no_loop_most_probable_node, _ = find_most_probable_source_no_loop(no_loops_G, induced_subgraph_sources)
            print(f"No loops probable source {no_loop_most_probable_node}")
            no_loop_k_most_probable_nodes.append(no_loop_most_probable_node)

            remaining_infected_nodes.remove(no_loop_most_probable_node)

        top_k_arbo_nodes = sorted(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get, reverse=True)[:k]
        print(f"Arbo probable sources {top_k_arbo_nodes}")

        # ************************************* exact match ***********************************************************
        exact_match_results = evaluate_exact_match(source_nodes, no_loop_k_most_probable_nodes, top_k_arbo_nodes)
        exact_total_no_loops += exact_match_results[0]
        exact_total_max_arbo += exact_match_results[1]

        # ************************************** recall ***************************************************************
        recall_results = evaluate_recall(source_nodes, no_loop_k_most_probable_nodes, top_k_arbo_nodes)
        recall_total_no_loops += recall_results[0]
        recall_total_max_arbo += recall_results[1]

        # *********************************** precision ***************************************************************
        precision_results = evaluate_precision(source_nodes, no_loop_k_most_probable_nodes, top_k_arbo_nodes)
        precision_total_no_loops += precision_results[0]
        precision_total_max_arbo += precision_results[1]

        # ************************************ distance ***************************************************************
        distance_results = evaluate_by_distance(G, source_nodes, no_loop_k_most_probable_nodes, top_k_arbo_nodes)
        distance_total_no_loops += distance_results[0]
        distance_total_max_arbo += distance_results[1]

        num_of_total_diffusion_calculated += 1
        print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")
        tot = time.time() - time_of_diffusion
        print("time it took to calculate the diffusion is:", tot)

    total_time = time.time() - begin_time

    Append_to_file(file_name,
                   f"Results for graph {graph_name} (total good diffusions: {num_of_total_diffusion_calculated})\n")
    Append_to_file(file_name, "--- Evaluation 1: Exact match successes ---")
    Append_to_file(file_name, f"No loop successes: {exact_total_no_loops}")
    Append_to_file(file_name, f"Max arborescence successes: {exact_total_max_arbo}\n")

    Append_to_file(file_name, "--- Evaluation 2: Recall percents ---")
    Append_to_file(file_name, f"No loop Recall: {recall_total_no_loops / 1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence Recall: {recall_total_max_arbo / 1000:.2f}%\n")

    Append_to_file(file_name, "--- Evaluation 3: Precision percents ---")
    Append_to_file(file_name, f"No loop precision: {precision_total_no_loops / 1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence precision: {precision_total_max_arbo / 1000:.2f}%\n")

    Append_to_file(file_name, "--- Evaluation 4: Distance-Based Precision percents ---")
    Append_to_file(file_name, f"No loop distance precision: {distance_total_no_loops / 1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence distance precision: {distance_total_max_arbo / 1000:.2f}%\n")

    Append_to_file(file_name, f"Number of too small diffusions: {num_of_too_small_diffusion}")
    Append_to_file(file_name, f"Number of too small A': {num_of_too_small_A_tag}")
    Append_to_file(file_name, f"Total time elapsed: {total_time:.2f} seconds")
    Append_to_file(file_name, "____________________________\n")


# ****** Method 3 - K strongest ******
def Find_K_strongest(path, graph_name, sep_char, P_range, k, file_name, ic_simulations=20, infected_threshold=0.3):
    begin_time = time.time()
    exact_total_no_loops = 0
    exact_total_max_arbo = 0
    recall_total_no_loops = 0
    recall_total_max_arbo = 0
    precision_total_no_loops = 0
    precision_total_max_arbo = 0
    distance_total_no_loops = 0
    distance_total_max_arbo = 0
    num_of_total_diffusion_calculated = 0
    print("Starting")

    t = time.time()
    G = read_graph_from_file_Prange(path, graph_name, sep_char, P_range)
    tot = time.time() - t
    print("finished reading ", graph_name, "time:", tot)

    min_size_of_diffusion = 20
    max_size_of_graph = 10000

    while num_of_total_diffusion_calculated < 1000:
        time_of_diffusion = time.time()
        G = reduce_graph_size(G, max_size_of_graph)

        source_nodes = random.sample(list(G.nodes()), k)
        print("Real source nodes:", source_nodes)

        infected_nodes = simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=len(G.nodes))
        if len(infected_nodes) < min_size_of_diffusion:
            continue

        remaining_candidates = set(infected_nodes)
        estimated_sources = []

        A_tag_full = Atag_calc(create_induced_subgraph(G, infected_nodes))
        if len(A_tag_full) <= 1:
            continue
        initial_induced_graph = create_induced_subgraph(G, A_tag_full)
        Max_weight_arborescence_G = Max_weight_arborescence(initial_induced_graph)
        top_k_arbo_nodes = sorted(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get, reverse=True)[:k]

        for i in range(k):
            if len(remaining_candidates) < 2:
                print("Too few remaining candidates")
                break

            A_tag = Atag_calc(create_induced_subgraph(G, remaining_candidates))
            if len(A_tag) <= 1:
                print("Too small A' after removal")
                break

            induced_G = create_induced_subgraph(G, A_tag)
            reversed_G = reverse_and_normalize_weights(induced_G)

            most_probable_node, _ = find_most_probable_source_no_loop(reversed_G, induced_G)
            print(f"Estimated source #{i + 1}: {most_probable_node}")
            estimated_sources.append(most_probable_node)

            infection_counts = Counter()
            for _ in range(ic_simulations):
                sim_infected = simulate_ic_model_on_k_sources(G, [most_probable_node], max_iterations=len(G.nodes))
                infection_counts.update(sim_infected)

            infected_to_remove = {node for node, count in infection_counts.items()
                                  if count / ic_simulations >= infected_threshold}
            print(f"Removing {len(infected_to_remove)} nodes infected from source {most_probable_node}")
            remaining_candidates -= infected_to_remove

        # *********************************** exact match *************************************************************
        exact_match_results = evaluate_exact_match(source_nodes, estimated_sources, top_k_arbo_nodes)
        exact_total_no_loops += exact_match_results[0]
        exact_total_max_arbo += exact_match_results[1]

        # ************************************** recall ***************************************************************
        recall_results = evaluate_recall(source_nodes, estimated_sources, top_k_arbo_nodes)
        recall_total_no_loops += recall_results[0]
        recall_total_max_arbo += recall_results[1]

        # ************************************* precision *************************************************************
        precision_results = evaluate_precision(source_nodes, estimated_sources, top_k_arbo_nodes)
        precision_total_no_loops += precision_results[0]
        precision_total_max_arbo += precision_results[1]

        # ************************************** distance *************************************************************
        distance_results = evaluate_by_distance(G, source_nodes, estimated_sources, top_k_arbo_nodes)
        distance_total_no_loops += distance_results[0]
        distance_total_max_arbo += distance_results[1]

        num_of_total_diffusion_calculated += 1
        print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")
        tot = time.time() - time_of_diffusion
        print("time it took to calculate the diffusion is:", tot)

    total_time = time.time() - begin_time

    Append_to_file(file_name,
                   f"Results for graph {graph_name} (total good diffusions: {num_of_total_diffusion_calculated})\n")
    Append_to_file(file_name, "--- Evaluation 1: Exact match successes ---")
    Append_to_file(file_name, f"No loop successes: {exact_total_no_loops}")
    Append_to_file(file_name, f"Max arborescence successes: {exact_total_max_arbo}\n")

    Append_to_file(file_name, "--- Evaluation 2: Recall percents ---")
    Append_to_file(file_name, f"No loop Recall: {recall_total_no_loops / 1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence Recall: {recall_total_max_arbo / 1000:.2f}%\n")

    Append_to_file(file_name, "--- Evaluation 3: Precision percents ---")
    Append_to_file(file_name, f"No loop precision: {precision_total_no_loops / 1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence precision: {precision_total_max_arbo / 1000:.2f}%\n")

    Append_to_file(file_name, "--- Evaluation 4: Distance-Based Precision percents ---")
    Append_to_file(file_name, f"No loop distance precision: {distance_total_no_loops / 1000:.2f}%")
    Append_to_file(file_name, f"Max arborescence distance precision: {distance_total_max_arbo / 1000:.2f}%\n")


    Append_to_file(file_name, f"Total time elapsed: {total_time:.2f} seconds")
    Append_to_file(file_name, "____________________________\n")


def evaluate_exact_match(source_nodes, no_loop_predicted_nodes, max_arbo_predicted_nodes):
    no_loop_success = int(all(node in no_loop_predicted_nodes for node in source_nodes))
    max_arbo_success = int(all(node in max_arbo_predicted_nodes for node in source_nodes))
    return no_loop_success, max_arbo_success


def evaluate_recall(source_nodes, no_loop_predicted_nodes, max_arbo_predicted_nodes):
    no_loop_recall = percent_exact_matches(source_nodes, no_loop_predicted_nodes)
    max_arbo_recall = percent_exact_matches(source_nodes, max_arbo_predicted_nodes)
    return no_loop_recall, max_arbo_recall


def evaluate_precision(source_nodes, no_loop_predicted_nodes, max_arbo_predicted_nodes):
    no_loop_precision = precision_of_estimation(source_nodes, no_loop_predicted_nodes)
    max_arbo_precision = precision_of_estimation(source_nodes, max_arbo_predicted_nodes)
    return no_loop_precision, max_arbo_precision


def evaluate_by_distance(G, source_nodes, no_loop_predicted_nodes, max_arbo_predicted_nodes):
    no_loop_dis = percent_sources_within_distance_k(G, source_nodes, no_loop_predicted_nodes, max_distance=3)
    max_arbo_dis = percent_sources_within_distance_k(G, source_nodes, max_arbo_predicted_nodes, max_distance=3)
    return no_loop_dis, max_arbo_dis


def reduce_graph_size(G: nx.DiGraph, maximum_size: int, max_edges: int = 30_000) -> nx.DiGraph:
    """
    Reduces the size of a directed graph while preserving connectivity and diffusion potential:
    1. Extracts the largest weakly connected component.
    2. Selects up to `maximum_size` nodes via BFS starting from a high out-degree node.
    3. If too many edges, randomly removes excess edges.
    """
    if G is None or maximum_size <= 0 or G.number_of_nodes() == 0:
        return nx.DiGraph()

    # Step 1: Get the largest weakly connected component
    if not nx.is_weakly_connected(G):
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_wcc).copy()

    if G.number_of_nodes() <= maximum_size and G.number_of_edges() <= max_edges:
        return G.copy()

    # Step 2: BFS from high out-degree node
    sorted_nodes = sorted(G.nodes, key=lambda n: G.out_degree(n), reverse=True)
    attempts = 10000
    best_nodes = set()

    for _ in range(attempts):
        start_node = random.choice(sorted_nodes[:len(sorted_nodes) // 5])  # top 20% by out-degree
        visited = set()
        queue = deque([start_node])
        selected_nodes = set()

        while queue and len(selected_nodes) < maximum_size:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            selected_nodes.add(current)
            for neighbor in G.successors(current):
                if neighbor not in visited and neighbor not in queue and len(selected_nodes) < maximum_size:
                    queue.append(neighbor)

        if len(selected_nodes) > len(best_nodes):
            best_nodes = selected_nodes
            if len(best_nodes) == maximum_size:
                break  # can't improve further

    # Step 3: Create subgraph and prune edges
    subgraph = G.subgraph(best_nodes).copy()
    if subgraph.number_of_edges() > max_edges:
        edges = list(subgraph.edges)
        random.shuffle(edges)
        subgraph.remove_edges_from(edges[max_edges:])

    return subgraph