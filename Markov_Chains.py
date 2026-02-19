"""Markov-chain based baselines for cascade source detection.

This module contains the core routines used by the paper
*"Identifying the Source of Information Spread in Networks via Markov Chains"*.

Key update in this version
--------------------------
* `calc_stationary_distribution` now supports a **power-iteration** method (with
  Cesàro averaging) in addition to the original dense eigen-decomposition.
  This is both faster and more memory-friendly when the induced graph is larger.
"""

import heapq

import networkx as nx
import numpy as np


def reverse_and_normalize_weights(G):
    """
    Reverse a directed graph and normalize the weights of the edges.

    :param G: Directed graph (DiGraph) with 'weight' as an edge attribute.
    :return: A new reversed graph with normalized weights.
    """
    # Reverse the graph
    reversed_G = G.reverse(copy=True)

    # Populate the reversed graph and normalize weights
    for node in G.nodes:
        # Get all incoming edges to this node in the original graph
        incoming_edges = G.in_edges(node, data=True)
        total_weight = sum(attr.get('weight', 1) for _, _, attr in incoming_edges)

        # Avoid division by zero
        if total_weight > 0:
            for u, v, attr in incoming_edges:
                weight = attr.get('weight', 1)
                normalized_weight = weight / total_weight
                # Add the reversed edge with normalized weight
                reversed_G.add_edge(v, u, weight=normalized_weight)

    return reversed_G


def apply_self_loop_method(G):
    """
    Apply the self-loop method to a weighted directed graph.

    Parameters:
    G (networkx.DiGraph): Input weighted directed graph. Edges should have 'weight' attribute.

    Returns:
    networkx.DiGraph: Transformed graph with normalized weights and self-loops
    """

    transformed_G = G.reverse(copy=True)
    max_in = max(
        sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
        for node in G.nodes()
    )

    if max_in == 0:
        # raise ValueError("Graph has no weighted edges")
        return transformed_G

    for node in G.nodes():
        in_weight = sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
        self_loop_weight = (max_in - in_weight) / max_in

        if self_loop_weight > 0:
            transformed_G.add_edge(node, node, weight=self_loop_weight)

    for u, v, data in G.edges(data=True):
        normalized_weight = data['weight'] / max_in
        transformed_G.add_edge(v, u, weight=normalized_weight)

    return transformed_G


def verify_self_loops_transformation(G, transformed_G):  # check if needed!!!!!!
    """
    Verify that the transformation was done correctly by checking:
    1. All nodes have outgoing probabilities that sum to 1
    2. All original edges are reversed and normalized
    3. All nodes have self-loops if needed

    Parameters:
    G (networkx.DiGraph): Original graph
    transformed_G (networkx.DiGraph): Transformed graph

    Returns:
    bool: True if transformation is valid
    """
    for node in transformed_G.nodes():
        out_weights = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))
        if not np.isclose(out_weights, 1.0, atol= 0.001):
            print(f"Verification failed: Node {node} has outgoing sum {out_weights}, expected 1.")
            print(f"Outgoing edges for {node}: {list(transformed_G.out_edges(node, data=True))}")
            return False

    return True


def verify_no_loops_transformation(G, transformed_G):
    """
    Verify that the transformation was done correctly:
    1. All original edges are reversed and normalized.
    2. All outgoing probabilities sum to 1.
    """

    # Check if outgoing probabilities sum to 1
    for node in transformed_G.nodes():
        out_weights = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))
        if not np.isclose(out_weights, 1.0, atol=1e-3):
            print(f"Verification failed: Node {node} has outgoing sum {out_weights}, expected 1.")
            print(f"Outgoing edges for {node}: {list(transformed_G.out_edges(node, data=True))}")
            return False

    return True


def Max_weight_arborescence(G_orig:nx.DiGraph):
    # maximum weight arborescence (from the Italy paper: "contrasting the spread of
    # misinformation in online social networks" by Amoruso at. al. 2020)
    max_arbo = nx.maximum_spanning_arborescence(G_orig, attr='weight')
    max_weight_arbo_dict ={}
    # The root of the arborescence is the unique node that has no incoming edges. (i.e. has in-degree of 0)
    for node in max_arbo:
        if max_arbo.in_degree(node) == 0:
            max_weight_arbo_dict[node] = 1
        else:
            max_weight_arbo_dict[node] = 0
    node_dict = max_weight_arbo_dict
    return node_dict


def _stationary_distribution_eig(G: nx.DiGraph) -> dict:
    """Stationary distribution via dense eigen-decomposition.

    This matches the original implementation (but uses the 'weight' attribute explicitly).
    """
    mat = nx.to_numpy_array(G, weight="weight")
    # This is a paper baseline; keep the strict Markov check.
    assert checkMarkov(mat)

    evals, evecs = np.linalg.eig(mat.T)
    mask = np.isclose(evals, 1, atol=1e-3)
    if not mask.any():
        print("Error in computing the stationary distribution: no eigenvalue close to 1.")
        return {}

    evec1 = evecs[:, mask][:, 0].real
    stationary = evec1 / evec1.sum()
    stationary = stationary.real

    node_names = list(G.nodes())
    return {node_names[i]: float(stationary[i]) for i in range(len(node_names))}


def _stationary_distribution_power(
    G: nx.DiGraph,
    tol: float = 1e-12,
    max_iter: int = 20000,
) -> dict:
    """Stationary distribution via power iteration (with Cesàro averaging).

    For irreducible but periodic chains, raw power iteration can oscillate.
    Cesàro averaging (average of iterates) converges to the stationary distribution.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    idx = {node: i for i, node in enumerate(nodes)}
    pi = np.full(n, 1.0 / n)
    avg = np.zeros(n, dtype=float)

    # Pre-extract outgoing edges to avoid repeated attribute lookups.
    out_edges = {
        u: [(v, float(data.get("weight", 0.0))) for _, v, data in G.out_edges(u, data=True)]
        for u in nodes
    }

    t = 0
    for t in range(1, max_iter + 1):
        pi_next = np.zeros(n, dtype=float)
        for u in nodes:
            iu = idx[u]
            p_u = pi[iu]
            outs = out_edges[u]
            if not outs:
                # Dangling node: keep probability mass.
                pi_next[iu] += p_u
                continue
            for v, w in outs:
                pi_next[idx[v]] += p_u * w

        avg += pi_next

        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            pi = pi_next
            break
        pi = pi_next

    pi_hat = avg / max(t, 1)
    s = pi_hat.sum()
    if s > 0:
        pi_hat = pi_hat / s

    return {node: float(pi_hat[idx[node]]) for node in nodes}


def calc_stationary_distribution(
    G: nx.DiGraph,
    method: str = "auto",
    tol: float = 1e-12,
    max_iter: int = 20000,
):
    """Return the stationary distribution of a Markov chain graph.

    Parameters
    ----------
    method:
        "auto" (default): use eigen for small graphs, power iteration otherwise.
        "eig": dense eigen-decomposition.
        "power": power iteration (with Cesàro averaging).
    """
    if method not in {"auto", "eig", "power"}:
        raise ValueError("method must be one of: 'auto', 'eig', 'power'")

    n = G.number_of_nodes()
    if method == "auto":
        # Eigen is fine for small induced graphs; power is safer for bigger ones.
        method = "eig" if n <= 500 else "power"

    if method == "eig":
        return _stationary_distribution_eig(G)
    return _stationary_distribution_power(G, tol=tol, max_iter=max_iter)



def calc_normalized_stationary_distribution(G, G_orignal, num_steps=1):
    """
    returns the stationary distribution of a markov chain network.
    The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
     Stationary Distribution of a Markov Chain.)
    :param G: a nx.DiGraph that is a Markov chain
    :return: a dict with the pairs-> (node:probability) for every node in G.
    """
    # Reuse the (possibly faster) stationary distribution helper.
    stationary_distribution = calc_stationary_distribution(G, method="auto")
    if not stationary_distribution:
        return {}

    normalized_distribution = {}
    for i, node in enumerate(G.nodes()):
        win = sum(data['weight'] for _, _, data in G_orignal.in_edges(node, data=True))
        if win > 0:
            normalized_distribution[node] = stationary_distribution[node] / win

    return normalized_distribution


def find_most_probable_source(G,num_steps=1):
    """
    Find the most probable source in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    # Calculate the stationary distribution
    stationary_distribution = calc_stationary_distribution(G)

    # Find the node with the maximum stationary probability
    if not stationary_distribution:
        return -1, -1
    most_probable_node = max(stationary_distribution, key=stationary_distribution.get)
    max_prob = stationary_distribution[most_probable_node]



    return most_probable_node, max_prob


def find_most_probable_source_no_loop(G, G_original, num_steps=1):
    """
    Find the most probable source in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    normalized_stationary_distribution = {}
    stationary_distribution = calc_stationary_distribution(G)


    if not stationary_distribution:
        return -1, -1

    win_prob = find_Win_prob(G_original)
    for node in stationary_distribution:
        if win_prob[node] == 0:
            normalized_stationary_distribution[node] = 0  # handling division by 0
            continue
        normalized_stationary_distribution[node] = stationary_distribution[node] / win_prob[node]

    most_probable_node = max(normalized_stationary_distribution, key=normalized_stationary_distribution.get)
    max_prob = normalized_stationary_distribution[most_probable_node]
    return most_probable_node, max_prob


def find_Win_prob(G):
    win_probs = {}
    for node in G.nodes:
        # Get all incoming edges to this node in the original graph
        incoming_edges = G.in_edges(node, data=True)
        total_weight = sum(attr.get('weight', 1) for _, _, attr in incoming_edges)
        win_probs[node] = total_weight
    return win_probs


def find_top_three(G, num_steps=1):
    """
    Returns the top 3 nodes based on stationary distribution.
    """
    stationary_distribution = calc_stationary_distribution(G)

    if not stationary_distribution:
        return []

    # Just return the nodes, sorted by probability
    top_3_nodes = heapq.nlargest(3, stationary_distribution.items(), key=lambda x: x[1])
    return [node for node, _ in top_3_nodes]

def find_top_three_no_loops(G, G_original):
    """
    Returns the top 3 nodes based on normalized stationary distribution.
    """
    normalized_stationary_distribution = {}
    stationary_distribution = calc_stationary_distribution(G)

    if not stationary_distribution:
        return -1, -1

    win_prob = find_Win_prob(G_original)
    for node in stationary_distribution:
        if win_prob[node] == 0:
            normalized_stationary_distribution[node] = 0  # handling division by 0
            continue
        normalized_stationary_distribution[node] = stationary_distribution[node] / win_prob[node]

    top_3_nodes = heapq.nlargest(3, normalized_stationary_distribution, key=normalized_stationary_distribution.get)
    return list(top_3_nodes)


def is_most_probable_near_source(G, source_node):
    # Find if the most probable node is near up to 3 steps from the true source
    stationary_distribution = calc_stationary_distribution(G)

    if not stationary_distribution:
        return False

    most_probable_node = max(stationary_distribution, key=stationary_distribution.get)
    shortest_path_length = nx.shortest_path_length(G, source=source_node, target=most_probable_node)

    return shortest_path_length <= 3


def is_most_probable_near_source_no_loop(G, G_original, source_node):
    # Find if the most probable node is near up to 3 steps from the true source for no loop
    stationary_distribution = calc_normalized_stationary_distribution(G, G_original)

    if not stationary_distribution:
        return False

    most_probable_node = max(stationary_distribution, key=stationary_distribution.get)
    shortest_path_length = nx.shortest_path_length(G, source=source_node, target=most_probable_node)

    return shortest_path_length <= 3


def is_most_probable_near_source_max_arbo(Max_weight_arborescence_G, G, source_node):
    # Find if the most probable node is near up to 3 steps from the true source for max arbo
    most_probable_node = max(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get)
    shortest_path_length = nx.shortest_path_length(G, source=source_node, target=most_probable_node)

    return shortest_path_length <= 3


def checkMarkov(m):
    """
    Check if the given matrix is a valid Markov chain transition matrix,
    allowing a tolerance of 0.01 for row sums.

    Parameters:
    m (numpy.ndarray): The matrix to check.

    Returns:
    bool: True if each row sums approximately to 1 (±0.01), False otherwise.
    """
    return np.all(np.isclose(np.sum(m, axis=1), 1.0, atol=0.01))


# ********************************************* K-Sources Methods *****************************************************
def find_K_most_probable_sources(G,k, num_steps=1):
    """
    Find the k most probable sources in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    # Calculate the stationary distribution
    stationary_distribution = calc_stationary_distribution(G)

    # Find the nodes with the maximum stationary probability
    if not stationary_distribution:
        return -1, -1

    top_k_nodes = sorted(stationary_distribution, key=stationary_distribution.get, reverse=True)[:k]
    top_k_probs = [stationary_distribution[node] for node in top_k_nodes]

    return top_k_nodes, top_k_probs


def find_K_most_probable_sources_no_loop(G, G_original, k, num_steps=1):
    """
    Find the k most probable sources in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    normalized_stationary_distribution = {}
    stationary_distribution = calc_stationary_distribution(G)
    if not stationary_distribution:
        return -1, -1

    win_prob = find_Win_prob(G_original)
    for node in stationary_distribution:
        if win_prob[node] == 0:
            normalized_stationary_distribution[node] = 0  # handling division by 0
            continue
        normalized_stationary_distribution[node] = stationary_distribution[node] / win_prob[node]

    top_k_nodes = sorted(normalized_stationary_distribution, key=normalized_stationary_distribution.get, reverse=True)[ :k]
    top_k_probs = [normalized_stationary_distribution[node] for node in top_k_nodes]

    return top_k_nodes, top_k_probs


# ----------------------------------- Evaluation methods for k sources ------------------------------------------------


# Returns the percentage of success - from the real k sources, how many are in our prediction sources - Recall
def percent_exact_matches(real_sources, estimated_sources):
    return len(set(real_sources) & set(estimated_sources)) / len(real_sources)


# From our predicted sources how many of them are real ones
def precision_of_estimation(real_sources, estimated_sources):
    return len(set(real_sources) & set(estimated_sources)) / len(estimated_sources)


def count_sources_within_distance_k(G, real_sources, estimated_sources, max_distance=3):
    count = 0
    for real_node in real_sources:
        if any(
            nx.has_path(G, real_node, est_node) and
            nx.shortest_path_length(G, real_node, est_node) <= max_distance
            for est_node in estimated_sources
        ):
            count += 1
    return count


def percent_sources_within_distance_k(G, real_sources, estimated_sources, max_distance=3):
    return count_sources_within_distance_k(G, real_sources, estimated_sources, max_distance) / len(real_sources)


