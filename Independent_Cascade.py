"""Independent Cascade (IC) diffusion utilities.

This codebase uses the IC model to generate cascades and to build the induced
subgraphs needed by the source-detection algorithms.

Key functions
-------------
* ``simulate_ic_model``: simulate an IC diffusion and return the final
  infected/active set.
* ``Atag_calc``: compute the candidate source set A' for a diffusion outcome.

Implementation note
-------------------
``Atag_calc`` is implemented in linear time using SCC condensation
(Yael's version ran a BFS from every node).
"""

from __future__ import annotations

import random
from typing import Optional, Set

import matplotlib.pyplot as plt
import networkx as nx


def simulate_ic_model(
    G: nx.DiGraph,
    source_node,
    max_iterations: int = 500,
    seed: Optional[int] = None,
) -> Set:
    """
    Simulates the Independent Cascade (IC) model on the graph with a given source node.

    :param G: The graph on which the simulation is run (must have 'weight' attributes on edges).
    :param source_node: The initial node that will be infected.
    :param max_iterations: Maximum number of iterations before stopping the simulation.
    :param seed: Optional random seed for reproducibility.


    :return:
        A set of infected nodes at the end of the diffusion.
    """
    if seed is not None:
        random.seed(seed)

    # Validate edge weights
    assert all('weight' in G[u][v] for u, v in G.edges), "All edges must have a 'weight' attribute!"
    assert all(0 <= G[u][v]['weight'] <= 1 for u, v in G.edges), "Edge weights must be in [0, 1]!"

    # Step 1: Initialize infection state
    infected: Set = {source_node}  # Nodes that are infected
    new_infected: Set = {source_node}  # Nodes to attempt to infect in the next iteration

    # Step 2: Simulate the spread of the infection
    iterations = 0
    while new_infected and iterations < max_iterations:
        next_infected = set()
        for node in new_infected:
            for neighbor in G.neighbors(node):  # Only considers outgoing edges
                if neighbor not in infected:
                    # Infection probability
                    infection_probability = G[node][neighbor]['weight']
                    if random.random() < infection_probability:
                        next_infected.add(neighbor)					
        infected.update(next_infected)
        new_infected = next_infected
        iterations += 1

    return infected


def visualize_infection(G, infected_nodes):
    color_map = ['red' if node in infected_nodes else 'skyblue' for node in G.nodes]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=color_map)
    plt.show()


def Atag_calc(G: nx.DiGraph):
    """Compute the candidate source set A' for a directed graph induced on active nodes.

    A node is in A' iff it can reach *all* active nodes via directed paths.

    Implementation note
    -------------------
    This is the set of nodes in the **unique source SCC** in the condensation DAG.
    If the condensation has 0 or >1 source SCCs, then A' is empty.

    This runs in O(|V|+|E|) time.
    """
    if G.number_of_nodes() == 0:
        return []

    # Strongly connected components.
    sccs = list(nx.strongly_connected_components(G))
    if len(sccs) == 1:
        return list(G.nodes())

    node_to_scc = {}
    for cid, comp in enumerate(sccs):
        for v in comp:
            node_to_scc[v] = cid

    indeg = [0] * len(sccs)
    for u, v in G.edges():
        cu = node_to_scc[u]
        cv = node_to_scc[v]
        if cu != cv:
            indeg[cv] += 1

    source_comps = [i for i, d in enumerate(indeg) if d == 0]
    if len(source_comps) != 1:
        return []

    return list(sccs[source_comps[0]])


def create_induced_subgraph(G, nodes):
    """
    Creates the induced subgraph of G using the specified nodes.

    :param G: The original graph
    :param nodes: A set or list of nodes to include in the induced subgraph
    :return: The induced subgraph
    """
    return G.subgraph(nodes).copy()


# ************************************** K-Sources Methods ************************************************************
def simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=500, seed=None):
    """
    Simulates the Independent Cascade (IC) model on the graph with a given source node.

    :param G: The graph on which the simulation is run (must have 'weight' attributes on edges).
    :param source_nodes: The initial nodes that will be infected.
    :param max_iterations: Maximum number of iterations before stopping the simulation.
    :param seed: Optional random seed for reproducibility.

    :return: A set of infected nodes.
    """
    if seed is not None:
        random.seed(seed)

    # Validate edge weights
    assert all('weight' in G[u][v] for u, v in G.edges), "All edges must have a 'weight' attribute!"
    assert all(0 <= G[u][v]['weight'] <= 1 for u, v in G.edges), "Edge weights must be in [0, 1]!"

    # Step 1: Initialize infection state
    infected = set(source_nodes)  # Nodes that are infected
    new_infected = set(source_nodes)  # Nodes to attempt to infect in the next iteration

    # Step 2: Simulate the spread of the infection
    iterations = 0
    while new_infected and iterations < max_iterations:
        next_infected = set()
        for node in new_infected:
            for neighbor in G.neighbors(node):  # Only considers outgoing edges
                if neighbor not in infected:
                    # Infection probability
                    infection_probability = G[node][neighbor]['weight']
                    if random.random() < infection_probability:
                        next_infected.add(neighbor)
        infected.update(next_infected)
        new_infected = next_infected
        iterations += 1

    return infected
