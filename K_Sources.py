from Graph_Generator import *
from Independent_Cascade import *
from Markov_Chains import *
from collections import Counter


def Append_to_file(file_name, text):
    print(file_name, ":", text)
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(text + "\n")


# ****** Method 1 - Top K-Sources ******
def Find_Top_K_Sources_Random(graphs, k):
    num_of_total_diffusion_calculated = 0  # without small diffusion and small A'
    print("Starting")
    for tuple1 in graphs:
        G = random_graph_generator(tuple1[1], tuple1[2], tuple1[3])

        naive_num_of_successes = 0
        no_loop_num_of_successes = 0
        self_loop_num_of_successes = 0
        max_arbo_num_of_successes = 0
        num_of_too_small_diffusion = 0
        num_of_too_small_A_tag = 0
        min_size_of_diffusion = 20


        while num_of_total_diffusion_calculated < 1000:
            source_nodes = random.sample(list(G.nodes()), k)
            print("The source nodes are: ", source_nodes)
            print("Running the ic model")
            infected_nodes = simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=len(G.nodes))

            if len(infected_nodes) < min_size_of_diffusion:  # if the diffusion is bigger then 20
                print("Too small diffusion len(active set)= ", len(infected_nodes))
                print("The infected nodes are: ", infected_nodes)
                num_of_too_small_diffusion += 1
                continue  # check the next graph

            print(f"Number of the infected nodes is: {len(infected_nodes)}")
            infected_graph = create_induced_subgraph(G, infected_nodes)
            possible_sources = Atag_calc(infected_graph)

            if len(possible_sources) <= 1:  # if A' is smaller then 2
                num_of_too_small_A_tag += 1
                continue  # check the next graph

            print(f"Possible sources: {possible_sources}")
            print(f"Number of possible sources of infection: {len(possible_sources)}")
            induced_graph = create_induced_subgraph(G, possible_sources)

            reversed_G = reverse_and_normalize_weights(induced_graph)
            no_loops_G = reverse_and_normalize_weights(induced_graph)
            self_loops_G = apply_self_loop_method(induced_graph)
            Max_weight_arborescence_G = Max_weight_arborescence(induced_graph)

            if not verify_no_loops_transformation(induced_graph, no_loops_G):
                print(f"Did the no loops method work? false")
            #     continue # the algorithm didn't work move to the next graph
            #
            if not verify_self_loops_transformation(induced_graph, self_loops_G):
                print(f"Did the self loops method work? false")
            #     continue # the algorithm didn't work move to the next graph

            naive_most_probable_nodes, naive_max_probs = find_K_most_probable_sources(reversed_G, k)
            print(f"naive probable sources  {naive_most_probable_nodes}")
            no_loop_most_probable_nodes, no_loop_max_probs = find_K_most_probable_sources_no_loop(no_loops_G, induced_graph, k)
            print(f"No loops probable sources {no_loop_most_probable_nodes}")
            self_loop_most_probable_nodes, self_loop_max_probs = find_K_most_probable_sources(self_loops_G, k)
            print(f"Self loops probable sources {self_loop_most_probable_nodes}")
            top_k_arbo_nodes = sorted(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get, reverse=True)[:k]
            print(f"Arbo probable sources {top_k_arbo_nodes}")
            top_k_arbo_probs = [Max_weight_arborescence_G[node] for node in top_k_arbo_nodes]

            print(f"The real sources are nodes {source_nodes}")

            if all(node in naive_most_probable_nodes for node in source_nodes):
                naive_num_of_successes += 1

            if all(node in no_loop_most_probable_nodes for node in source_nodes):
                no_loop_num_of_successes += 1

            if all(node in self_loop_most_probable_nodes for node in source_nodes):
                self_loop_num_of_successes += 1

            if all(node in top_k_arbo_nodes for node in source_nodes):
                max_arbo_num_of_successes += 1

            num_of_total_diffusion_calculated += 1
            print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")

        print(f"The number of Successes in naive is: {naive_num_of_successes} ")
        print(f"The number of Successes in no loop is: {no_loop_num_of_successes} ")
        print(f"The number of Successes in self loop is: {self_loop_num_of_successes} ")
        print(f"The number of Max weight arborescence is: {max_arbo_num_of_successes}")
        print("number of 'good' diffusions:" + str(num_of_total_diffusion_calculated))

# ****** Method 2 - K Times ******

def Find_Source_K_Times(graphs, k):
    num_of_total_diffusion_calculated = 0  # without small diffusion and small A'
    print("Starting")
    for tuple1 in graphs:
        G = random_graph_generator(tuple1[1], tuple1[2], tuple1[3])

        naive_num_of_successes = 0
        no_loop_num_of_successes = 0
        self_loop_num_of_successes = 0
        max_arbo_num_of_successes = 0
        num_of_too_small_diffusion = 0
        num_of_too_small_A_tag = 0
        min_size_of_diffusion = 20

        # # output file:
        # output_file = tuple1[0] + ".txt"

        while num_of_total_diffusion_calculated < 1000:
            source_nodes = random.sample(list(G.nodes()), k)
            print("The source nodes are: ", source_nodes)
            print("Running the ic model")
            infected_nodes = simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=len(G.nodes))

            if len(infected_nodes) < min_size_of_diffusion:  # if the diffusion is bigger then 20
                print("Too small diffusion len(active set)= ", len(infected_nodes))
                print("The infected nodes are: ", infected_nodes)
                num_of_too_small_diffusion += 1
                continue  # check the next graph

            print(f"Number of the infected nodes is: {len(infected_nodes)}")
            infected_graph = create_induced_subgraph(G, infected_nodes)
            possible_sources = Atag_calc(infected_graph)

            if len(possible_sources) <= 1:  # if A' is smaller then 2
                num_of_too_small_A_tag += 1
                continue  # check the next graph

            print(f"Possible sources: {possible_sources}")
            print(f"Number of possible sources of infection: {len(possible_sources)}")
            induced_graph = create_induced_subgraph(G, possible_sources)

            # reversed_G = reverse_and_normalize_weights(induced_graph)
            no_loops_G = reverse_and_normalize_weights(induced_graph)
            # self_loops_G = apply_self_loop_method(induced_graph)
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



            if all(node in no_loop_k_most_probable_nodes for node in source_nodes):
                no_loop_num_of_successes += 1

            if all(node in top_k_arbo_nodes for node in source_nodes):
                max_arbo_num_of_successes += 1

            num_of_total_diffusion_calculated += 1
            print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")

        # print(f"The number of Successes in naive is: {naive_num_of_successes} ")
        print(f"The number of Successes in no loop is: {no_loop_num_of_successes} ")
        # print(f"The number of Successes in self loop is: {self_loop_num_of_successes} ")
        print(f"The number of Max weight arborescence is: {max_arbo_num_of_successes}")
        print("number of 'good' diffusions:" + str(num_of_total_diffusion_calculated))

# ****** Method 3 - K strongest ******

def Find_K_strongest(graphs, k, ic_simulations=20, infected_threshold=0.3):
    num_of_total_diffusion_calculated = 0
    exact_total = 0
    precision_total = 0
    nearby_total = 0
    print("Starting")

    for tuple1 in graphs:
        G = random_graph_generator(tuple1[1], tuple1[2], tuple1[3])

        num_of_successes = 0
        max_arbo_num_of_successes = 0
        min_size_of_diffusion = 20

        while num_of_total_diffusion_calculated < 1000:
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
            print("Top K Arbo nodes:", top_k_arbo_nodes)

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
                print(f"Estimated source #{i+1}: {most_probable_node}")
                estimated_sources.append(most_probable_node)

                infection_counts = Counter()
                for _ in range(ic_simulations):
                    sim_infected = simulate_ic_model_on_k_sources(G, [most_probable_node], max_iterations=len(G.nodes))
                    infection_counts.update(sim_infected)

                infected_to_remove = {node for node, count in infection_counts.items()
                                      if count / ic_simulations >= infected_threshold}
                print(f"Removing {len(infected_to_remove)} nodes infected from source {most_probable_node}")
                remaining_candidates -= infected_to_remove

            print("Estimated sources:", estimated_sources)
            percent_nearby = percent_sources_within_distance_k(G, source_nodes, estimated_sources, max_distance=3)
            nearby_total += percent_nearby
            exact_total += percent_exact_matches(source_nodes, estimated_sources)
            precision_total += precision_of_estimation(source_nodes, estimated_sources)

            if all(node in estimated_sources for node in source_nodes):
                num_of_successes += 1
                print("Success in no loops.")
            if all(node in top_k_arbo_nodes for node in source_nodes):
                max_arbo_num_of_successes += 1
                print("success in max arbo.")

            num_of_total_diffusion_calculated += 1
            print(f"Completed diffusion #{num_of_total_diffusion_calculated}\n")
        N = num_of_total_diffusion_calculated
        print(f"--- Results for graph {tuple1[0]} ---")
        print(f"Successes with K-Strongest method no loops: {num_of_successes} / 1000")
        print(f"Successes with Max Arborescence: {max_arbo_num_of_successes} / 1000")
        print(f"Avg % of exact matches: {100 * exact_total / N:.2f}%")
        print(f"Avg % of real sources with a close predicted source (≤3 hops): {100 * nearby_total / N:.2f}%")
        print(f"Avg precision of estimation: {100 * precision_total / N:.2f}%")
        print("-------------------------------------\n")