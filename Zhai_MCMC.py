"""Implementation of Zhai et al. (2015) cascade source inference (Algorithm 3).

Paper: *"Cascade source inference in networks: a Markov chain Monte Carlo approach"*
X. Zhai, W. Wu, W. Xu (Computational Social Networks, 2015).

This module implements the **advanced source inference algorithm (Algorithm 3)** from
the paper, including the Metropolis local move (Algorithm 1).

Notes
-----
* The original algorithm is computationally expensive. The implementation here stays
  faithful to the paper, but exposes a few practical knobs: `burn_in`, `sample_every`
  (thinning), and `max_candidates_per_sample`.
* Candidate sources in a sample are computed using SCC condensation as suggested in
  the paper (linear time).
* The original algorithm stores three |A| x |A| tables (``accu``, ``count``,
  ``result``). This is O(|A|^2) memory. However, since we use the final-cascade setting in the experiments (snapshot time
  ``tau = |A|`` known), the algorithm simplifies: for any candidate source s we
  always have tau > G1,k(s) (since graph diameter is at most |A|-1), hence
  the per-sample contribution reduces to a constant factor and the score is
  proportional to the number of samples in which s can reach all active nodes.
  In that common case, this implementation uses an O(|A|)-memory fast path.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _edge_weight(G: nx.DiGraph, u, v) -> float:
    """Return edge weight as float (default 0.0 if missing)."""
    try:
        return float(G[u][v].get("weight", 0.0))
    except KeyError:
        return 0.0


def _candidate_sources_unique_source_scc(G: nx.DiGraph) -> List:
    """Return candidate source set C for a directed graph with vertex set A.

    C is the set of nodes that can reach all nodes in G.

    As described by Zhai et al., compute SCCs, build condensation DAG, then:
      * if exactly one SCC has in-degree 0, C is that SCC
      * else, C is empty.
    """
    if G.number_of_nodes() == 0:
        return []

    sccs = list(nx.strongly_connected_components(G))
    if len(sccs) == 1:
        return list(G.nodes())

    node_to_scc: Dict = {}
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


def _eccentricity_and_farthest_nodes(G: nx.DiGraph, source, active_set: Set) -> Tuple[int, List]:
    """Compute eccentricity ε_G(source) and list of farthest nodes.

    Eccentricity here is max directed shortest-path distance from `source` to nodes in `active_set`.
    Assumes `source` can reach all nodes in `active_set`.
    """
    # NetworkX BFS on directed graph
    dist = nx.single_source_shortest_path_length(G, source)

    # Ensure reachability to all active nodes.
    if len(dist) < len(active_set):
        # Not all nodes reachable; return sentinel.
        return math.inf, []

    ecc = max(dist.values()) if dist else 0
    farthest = [v for v, d in dist.items() if d == ecc]
    return ecc, farthest


def _compute_log_Wi(G: nx.DiGraph, active_nodes: Sequence) -> Tuple[np.ndarray, float, int, float]:
    """Compute log(W_i) for each active node i and derived quantities.

    Wi = ∏_{j∈V\A, (i,j)∈E} (1 - w_{i,j})

    Returns
    -------
    logWi : np.ndarray of shape (n,)
        log(Wi) or -inf if Wi=0.
    total_logW_finite : float
        Σ logWi over finite entries (i.e., product of non-zero Wi's).
    zero_count : int
        Number of Wi that are exactly 0.
    W : float
        W = ∏_i Wi (can be 0).
    """
    active_set = set(active_nodes)
    n = len(active_nodes)

    logWi = np.zeros(n, dtype=float)
    zero_mask = np.zeros(n, dtype=bool)

    for idx, u in enumerate(active_nodes):
        acc = 0.0
        for _, v, data in G.out_edges(u, data=True):
            if v in active_set:
                continue
            w = float(data.get("weight", 0.0))
            q = 1.0 - w
            if q <= 0.0:
                # Wi becomes 0
                zero_mask[idx] = True
                acc = -math.inf
                break
            acc += math.log(q)
        logWi[idx] = acc

    zero_count = int(zero_mask.sum())
    finite_mask = ~zero_mask
    total_logW_finite = float(logWi[finite_mask].sum()) if finite_mask.any() else 0.0
    W = 0.0 if zero_count > 0 else float(math.exp(total_logW_finite))
    return logWi, total_logW_finite, zero_count, W


def _w_from_farthest(
    farthest_nodes: Sequence,
    node_to_idx: Dict,
    logWi: np.ndarray,
    total_logW_finite: float,
    zero_count: int,
) -> float:
    """Compute the weight w used in Algorithm 3, line 12-14.

    In the paper:
      w = W; for j with d(s,j)==ε(s): w = w / Wj

    Since W = ∏_i Wi, this simplifies to w = ∏_{i not in farthest} Wi.

    We compute it in log-space, handling Wi=0 robustly:
      * if any zero Wi is outside `farthest`, w=0
      * otherwise, w = exp( Σ_{finite i not in farthest} logWi[i] ).
    """
    if not farthest_nodes:
        # Shouldn't happen in practice, but keep safe.
        # If farthest is empty, w is product over all nodes (including zeros) -> 0 if any zeros.
        if zero_count > 0:
            return 0.0
        return float(math.exp(total_logW_finite))

    finite_sum_farthest = 0.0
    zeros_in_farthest = 0
    for v in farthest_nodes:
        idx = node_to_idx[v]
        lv = logWi[idx]
        if math.isinf(lv) and lv < 0:
            zeros_in_farthest += 1
        else:
            finite_sum_farthest += float(lv)

    # Any zero outside farthest makes w=0.
    if zero_count - zeros_in_farthest > 0:
        return 0.0

    logw = total_logW_finite - finite_sum_farthest
    # exp underflow is fine; it means w is extremely small.
    return float(math.exp(logw)) if logw > -745 else 0.0


def _local_move_metropolis(
    Gk: nx.DiGraph,
    base_edges: Sequence[Tuple],
    weight_lookup: Dict[Tuple, float],
    rng: random.Random,
) -> None:
    """Algorithm 1: Local move (Metropolis) restricted to `base_edges`.

    The move toggles a random edge (i,j) from the *base* graph. The proposal is accepted
    with probability min(p,1) if the proposed graph remains in S1 (has a non-empty
    candidate set), and rejected otherwise.

    This function mutates `Gk` in-place.
    """
    (u, v) = base_edges[rng.randrange(len(base_edges))]
    w = weight_lookup[(u, v)]

    has_edge = Gk.has_edge(u, v)

    # Propose toggle
    if has_edge:
        # remove edge
        if w <= 0.0:
            ratio = float("inf")
        else:
            ratio = (1.0 - w) / w
        Gk.remove_edge(u, v)
    else:
        # add edge
        if w >= 1.0:
            ratio = float("inf")
        elif w <= 0.0:
            ratio = 0.0
        else:
            ratio = w / (1.0 - w)
        Gk.add_edge(u, v, weight=w)

    # Check if proposed graph is still in S1 (has a candidate set)
    C = _candidate_sources_unique_source_scc(Gk)
    if C:
        p = min(1.0, ratio)
    else:
        p = 0.0

    # Accept / reject
    if rng.random() >= p:
        # reject: revert toggle
        if has_edge:
            Gk.add_edge(u, v, weight=w)
        else:
            Gk.remove_edge(u, v)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ZhaiMCMCResult:
    """Return type for the Zhai Algorithm 3 implementation."""

    best_source: object
    scores: Dict[object, float]
    # For debugging/analysis
    tau_l: int
    tau_u: int
    K: int


def _is_final_cascade_tau(tau_range: Optional[Tuple[int, int]], active_size: int) -> bool:
    """Return True iff the caller is asking for the final-cascade setting.

    In Zhai et al.'s evaluation for the *final cascade*, they fix the snapshot time
    to ``tau = |A|`` and assume it is known. In our API this corresponds to
    ``tau_range=(|A|,|A|)``.
    """
    if tau_range is None:
        return False
    tau_l, tau_u = tau_range
    return int(tau_l) == active_size and int(tau_u) == active_size


def _infer_source_final_cascade_fastpath(
    G: nx.DiGraph,
    active_list: List,
    *,
    K: int,
    burn_in: int,
    sample_every: int,
    max_candidates_per_sample: Optional[int],
    seed: Optional[int],
    return_result: bool,
):
    """Final-cascade special case of Algorithm 3 using **O(|A|)** memory.

    When ``tau = |A|`` is known, for every candidate source we have ``tau > G1,k(s)``
    (see Zhai et al., discussion around Algorithm 3). In this case, the score for a
    node is proportional to the number of MCMC samples in which it can reach all
    nodes in the active set.

    Therefore, we only need a per-node counter and do not need the ``|A|x|A|`` tables
    used in the general Algorithm 3 implementation.
    """
    n = len(active_list)
    if n == 0:
        raise ValueError("active_nodes must be non-empty")
    if n == 1:
        only = active_list[0]
        res = ZhaiMCMCResult(only, {only: 1.0}, n, n, K)
        return res if return_result else only

    rng = random.Random(seed)

    # Build G1 induced on active nodes with internal edges (E1)
    G1 = G.subgraph(active_list).copy()
    E1 = list(G1.edges())
    if not E1:
        # No internal edges: reachability is impossible unless n==1 (handled above).
        best = active_list[0]
        scores = {v: 0.0 for v in active_list}
        scores[best] = 1.0
        res = ZhaiMCMCResult(best, scores, n, n, K)
        return res if return_result else best

    weight_lookup = {(u, v): _edge_weight(G1, u, v) for (u, v) in E1}

    # Initial sample G1,0 = G1
    Gk = G1.copy()

    # Burn-in
    for _ in range(burn_in):
        _local_move_metropolis(Gk, E1, weight_lookup, rng)

    # We only need counts per node (linear memory).
    counts: Dict[object, int] = {v: 0 for v in active_list}

    # Main sampling loop: count how often each node is in the candidate set.
    for _k in range(K):
        C = _candidate_sources_unique_source_scc(Gk)
        if C:
            if max_candidates_per_sample is not None and len(C) > max_candidates_per_sample:
                C = rng.sample(C, k=max_candidates_per_sample)
            for s in C:
                counts[s] += 1

        # Advance chain by `sample_every` local moves.
        for _ in range(sample_every):
            _local_move_metropolis(Gk, E1, weight_lookup, rng)

    # Convert counts to scores in [0,1] (monotone to counts, so argmax unchanged)
    scores = {v: (counts[v] / float(K)) for v in active_list}
    best_source = max(active_list, key=lambda v: scores[v])

    if return_result:
        return ZhaiMCMCResult(best_source, scores, n, n, K)
    return best_source


def infer_source_zhai_mcmc(
    G: nx.DiGraph,
    active_nodes: Iterable,
    *,
    K: int = 10_000,
    tau_range: Optional[Tuple[int, int]] = None,
    burn_in: int = 0,
    sample_every: int = 1,
    max_candidates_per_sample: Optional[int] = None,
    seed: Optional[int] = None,
    return_result: bool = False,
):
    """Infer cascade source using Zhai et al. Algorithm 3 (advanced MCMC).

    Parameters
    ----------
    G:
        Original network (directed) with edge attribute `weight` in [0,1].
    active_nodes:
        The observed active set A_τ (snapshot).
    K:
        Number of recorded MCMC samples.
    tau_range:
        Optional (tau_l, tau_u). If None, we use the paper's suggestion for unknown τ:
        tau_l = min eccentricity among candidate sources in the full induced G1,
        tau_u = |A|.
    burn_in:
        Number of MCMC moves before recording samples.
    sample_every:
        Thinning interval (number of MCMC moves between recorded samples).
    max_candidates_per_sample:
        Optional cap on |C| per sample (random subsampling) for speed.
    seed:
        RNG seed.
    return_result:
        If True, returns a :class:`ZhaiMCMCResult` with scores.

    Returns
    -------
    best_source or ZhaiMCMCResult
    """
    active_list = list(active_nodes)
    n = len(active_list)
    if n == 0:
        raise ValueError("active_nodes must be non-empty")
    if n == 1:
        return ZhaiMCMCResult(active_list[0], {active_list[0]: 1.0}, 1, 1, K) if return_result else active_list[0]

    if K <= 0:
        raise ValueError("K must be positive")
    if sample_every <= 0:
        raise ValueError("sample_every must be >= 1")
    if burn_in < 0:
        raise ValueError("burn_in must be >= 0")

    # -------------------- Final-cascade fast path --------------------
    # In your experiments (and in Zhai et al.'s first evaluation), the snapshot is
    # taken after termination and the starting time is set to tau = |A|.
    # In that setting we can avoid the O(|A|^2) tables and just count candidates.
    if _is_final_cascade_tau(tau_range, n):
        return _infer_source_final_cascade_fastpath(
            G,
            active_list,
            K=K,
            burn_in=burn_in,
            sample_every=sample_every,
            max_candidates_per_sample=max_candidates_per_sample,
            seed=seed,
            return_result=return_result,
        )

    rng = random.Random(seed)
    active_set = set(active_list)
    node_to_idx = {v: i for i, v in enumerate(active_list)}

    # Build G1 induced on active nodes with internal edges (E1)
    G1 = G.subgraph(active_list).copy()
    E1 = list(G1.edges())
    if not E1:
        # No internal edges: source is completely ambiguous; fall back to random.
        best = active_list[0]
        scores = {v: 0.0 for v in active_list}
        scores[best] = 1.0
        return ZhaiMCMCResult(best, scores, 1, n, K) if return_result else best

    weight_lookup = {(u, v): _edge_weight(G1, u, v) for (u, v) in E1}

    # Precompute W_i and W (line 5-7)
    logWi, total_logW_finite, zero_count, W = _compute_log_Wi(G, active_list)

    # Determine default tau range
    if tau_range is None:
        C0 = _candidate_sources_unique_source_scc(G1)
        if not C0:
            # If condensation has multiple sources, the snapshot is inconsistent with IC
            # under this subgraph-only view; still proceed with tau_l=1.
            tau_l = 1
        else:
            # tau_l = min eccentricity among candidates in full G1
            eccs = []
            for s in C0:
                ecc, _ = _eccentricity_and_farthest_nodes(G1, s, active_set)
                if math.isfinite(ecc):
                    eccs.append(int(ecc))
            tau_l = max(1, min(eccs)) if eccs else 1
        tau_u = n
    else:
        tau_l, tau_u = tau_range
        tau_l = max(1, int(tau_l))
        tau_u = min(n, int(tau_u))
        if tau_l > tau_u:
            raise ValueError("tau_range must satisfy tau_l <= tau_u")

    # accu and count tables: shape (n, n+1) so we can index by tau in [1..n]
    accu = np.zeros((n, n + 1), dtype=float)
    count = np.zeros((n, n + 1), dtype=np.int64)

    # Initial sample G1,0 = G1
    Gk = G1.copy()

    # Burn-in
    for _ in range(burn_in):
        _local_move_metropolis(Gk, E1, weight_lookup, rng)

    # Main sampling loop (Algorithm 3 line 8-18)
    for _k in range(K):
        C = _candidate_sources_unique_source_scc(Gk)
        if C:
            if max_candidates_per_sample is not None and len(C) > max_candidates_per_sample:
                C = rng.sample(C, k=max_candidates_per_sample)

            for s in C:
                ecc, farthest = _eccentricity_and_farthest_nodes(Gk, s, active_set)
                if not math.isfinite(ecc):
                    continue
                ecc = int(ecc)
                if ecc < 1:
                    ecc = 1
                if ecc > n:
                    # Should not happen, but keep within allocated table.
                    ecc = n

                w_val = _w_from_farthest(
                    farthest,
                    node_to_idx,
                    logWi,
                    total_logW_finite,
                    zero_count,
                )

                si = node_to_idx[s]
                accu[si, ecc] += w_val
                count[si, ecc] += 1

        # Advance chain by `sample_every` local moves.
        for _ in range(sample_every):
            _local_move_metropolis(Gk, E1, weight_lookup, rng)

    # Post-processing: compute result table (Algorithm 3 line 19-24)
    result = np.zeros((n, n + 1), dtype=float)
    for i in range(n):
        c = 0
        for tau in range(1, n + 1):
            result[i, tau] = accu[i, tau] + W * c
            c += count[i, tau]

    # Score each node by summing over the time range.
    scores_arr = result[:, tau_l : tau_u + 1].sum(axis=1)
    best_idx = int(np.argmax(scores_arr))
    best_source = active_list[best_idx]
    scores = {active_list[i]: float(scores_arr[i]) for i in range(n)}

    if return_result:
        return ZhaiMCMCResult(best_source, scores, tau_l, tau_u, K)
    return best_source


def rank_sources_zhai_mcmc(
    G: nx.DiGraph,
    active_nodes: Iterable,
    *,
    K: int = 10_000,
    tau_range: Optional[Tuple[int, int]] = None,
    burn_in: int = 0,
    sample_every: int = 1,
    max_candidates_per_sample: Optional[int] = None,
    seed: Optional[int] = None,
    top_k: Optional[int] = None,
) -> List:
    """Return nodes ranked by Zhai Algorithm 3 score (descending)."""
    res: ZhaiMCMCResult = infer_source_zhai_mcmc(
        G,
        active_nodes,
        K=K,
        tau_range=tau_range,
        burn_in=burn_in,
        sample_every=sample_every,
        max_candidates_per_sample=max_candidates_per_sample,
        seed=seed,
        return_result=True,
    )
    ranked = sorted(res.scores, key=res.scores.get, reverse=True)
    if top_k is not None:
        return ranked[: int(top_k)]
    return ranked
