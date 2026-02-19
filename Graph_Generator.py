import networkx as nx
import random
import matplotlib.pyplot as plt


def random_graph_generator(numberOfNodes, ProbOfAnEdge, maxProbForDiffusion, seed):
    """
    Generates a random directed graph with independent edge weights.

    - numberOfNodes: the number of nodes in the graph.
    - ProbOfAnEdge: probability of an edge between any pair of nodes.
    - maxProbForDiffusion: maximum weight for the edges.

    :param numberOfNodes: int
    :param ProbOfAnEdge: float
    :param maxProbForDiffusion: float

    :return: nx.DiGraph object
    """
    # Create a random directed graph
    G = nx.fast_gnp_random_graph(numberOfNodes, ProbOfAnEdge, seed=seed, directed=True)

    # Assign random weights independently for each directed edge
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.random() * maxProbForDiffusion

    return G


def visualize_graph(G):
    """
    Visualizes the directed graph with correct edge weights.
    """
    pos = nx.spring_layout(G)  # Spring layout for positioning
    plt.figure(figsize=(12, 8))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

    # Draw directed edges
    nx.draw_networkx_edges(G, pos, width=1.0, edge_color='gray', arrows=True, connectionstyle="arc3,rad=0.2")

    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Display edge weights
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Graph Visualization with Correct Edge Weights")
    plt.axis('off')
    plt.show()


def visualize_subgraph(G, max_nodes=100):
    """
    Visualizes a subgraph of the graph with nodes, edges, and edge weights.
    Displays edge weights with 3 decimal places, regardless of the number of edges.
    """
    # Sample a subgraph with a maximum number of nodes
    subgraph_nodes = list(G.nodes())[:max_nodes]  # Select the first 'max_nodes' nodes
    subgraph = G.subgraph(subgraph_nodes)

    pos = nx.spring_layout(subgraph, seed=42)  # Using a fixed seed for reproducibility

    plt.figure(figsize=(12, 10))  # Increase figure size for better visibility
    nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.7, edge_color='gray')

    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold', font_color='black')

    # Get edge weights (probabilities) and format them with 3 decimal places
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')

    if not edge_labels:  # Debugging if there are no edge labels
        print("No edge weights found in the subgraph.")

    formatted_edge_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()}

    # Draw edge labels (with formatted probabilities) without checking the number of edges
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=formatted_edge_labels, font_size=8, font_color='red')

    plt.title("Graph Visualization (Subgraph)")
    plt.axis('off')
    plt.show()


def visualize_large_graph(G, max_nodes=1000, layout='spring'):
    """
    Visualizes a large graph by displaying only a subgraph with a maximum number of nodes.
    Uses either a spring layout or spectral layout for large graphs.
    """
    # Convert G.nodes() to a list before using random.sample()
    subgraph_nodes = random.sample(list(G.nodes()), min(max_nodes, len(G.nodes())))  # Randomly sample nodes
    subgraph = G.subgraph(subgraph_nodes)

    if layout == 'spring':
        pos = nx.spring_layout(subgraph, seed=42, k=0.15, iterations=20)
    elif layout == 'spectral':
        pos = nx.spectral_layout(subgraph)  # Faster layout for large graphs
    else:
        pos = nx.circular_layout(subgraph)  # Use circular layout for simplicity

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, width=0.5, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_color='black')

    plt.title(f"Graph Visualization (Subgraph - {len(subgraph.nodes())} nodes)")
    plt.axis('off')
    plt.show()


def read_graph_from_file_Prange(file_path, graph_name, separator, P_range):
    g = nx.DiGraph()
    with open(file_path, 'r') as my_file:
        for line in my_file.readlines():
            if line[0] != "%" and line[0] != "#":
                line1 = line.split(separator)
                if (graph_name == 'epinion_trust'):
                    pass
                v1 = int(line1[0].strip())
                v2 = int(line1[1].strip())
                weight1 = random.uniform(0, P_range)
                g.add_edge(v_of_edge=v1, u_of_edge=v2, weight=weight1)

    print("Finished reading", graph_name, "network. ", "number of nodes:", len(g.nodes), "number of edges:",
          len(g.edges))
    return g

def read_graph_from_file_norm(file_path, graph_name, seperator, norm_epsilon):
    g = nx.DiGraph()
    with open(file_path ,'r') as my_file:
        for line in my_file.readlines():
            if line[0]!="%" and line[0]!="#":
                line1 = line.split(seperator)
                v1 = int(line1[0].strip())
                v2 = int(line1[1].strip())
                weight1 = random.uniform(0.0, 2.0)
                g.add_edge(v_of_edge=v1,u_of_edge=v2,weight = weight1)

    #adgusting the edge-probabilities in order for the nodes expected sum-of-out-degree will be 1 +epsilon:
    for node1 in g.nodes:
        out_deg = g.out_degree(node1)
        for (node1, node2) in g.out_edges(node1):
            new_prob = (g.edges[node1,node2]["weight"] /out_deg)*(1+norm_epsilon)
            g.edges[node1,node2]["weight"] = min(new_prob,1.0)

    print("Finished readeing", graph_name ,"network","number of nodes:",len(g.nodes),"number of edges:",
          len(g.edges))
    return g