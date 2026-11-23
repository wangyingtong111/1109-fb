import numpy as np

try:
    import jax.lax
except (ImportError, RuntimeError):
    JAX_IS_AVAILABLE = False
else:
    JAX_IS_AVAILABLE = True
    # if JAX is available, we need to initialize it with a dummy scores and capture
    # any output to avoid it from saying that gpu is not available
    _ = jax.lax.top_k(np.array([0] * 5), 1)


def _topk_numpy(query_scores, k, sorted):
    # Deterministic top-k selection: the score is the primary criterion and ties
    # are broken by the original document index (lower index wins). This makes
    # the result independent of k, so e.g. the top-2 of a query is always a
    # prefix of the top-3 for the same index.
    #
    # To stay fast on large collections we never fully sort the whole score
    # vector: np.partition finds the k-th largest score (the cutoff) in O(n),
    # then only the k selected items are ordered.
    n = query_scores.shape[0]
    if k > n:
        k = n
    if k == 0:
        return query_scores[:0].copy(), np.empty(0, dtype=np.intp)

    # Find the cutoff score: the k-th largest value. np.partition only
    # partially reorders the array, which is cheaper than a full argsort.
    kth_value = np.partition(query_scores, n - k)[n - k]

    # Every document scoring strictly above the cutoff is always selected.
    above_ind = np.flatnonzero(query_scores > kth_value)

    # The remaining slots are filled with documents tied exactly at the cutoff,
    # taking those with the smallest original document indices first so that
    # the selection is deterministic and consistent across different k.
    num_ties = k - above_ind.size
    tied_ind = np.flatnonzero(query_scores == kth_value)[:num_ties]

    ind = np.concatenate([above_ind, tied_ind])
    partitioned_scores = query_scores[ind]

    if sorted:
        # Sort only the k selected items: score descending is the primary key
        # and the document index (ascending) breaks ties. np.lexsort is stable
        # and uses the last key as primary, so the ordering is deterministic.
        sorted_trunc_ind = np.lexsort((ind, -partitioned_scores))
        ind = ind[sorted_trunc_ind]
        query_scores = partitioned_scores[sorted_trunc_ind]
    else:
        query_scores = partitioned_scores

    return query_scores, ind


def _topk_jax(query_scores, k):
    topk_scores, topk_indices = jax.lax.top_k(query_scores, k)
    topk_scores = np.asarray(topk_scores)
    topk_indices = np.asarray(topk_indices)

    return topk_scores, topk_indices


def topk(query_scores, k, backend="auto", sorted=True):
    """
    This function is used to retrieve the top-k results for a single query. It will only work
    on a 1-dimensional array of scores.
    """
    if backend == "auto":
        # if jax.lax is available, use it to speed up selection, otherwise use numpy
        backend = "jax" if JAX_IS_AVAILABLE else "numpy"
    
    if backend not in ["numpy", "jax"]:
        raise ValueError("Invalid backend. Please choose from 'numpy' or 'jax'.")
    elif backend == "jax":
        if not JAX_IS_AVAILABLE:
            raise ImportError("JAX is not available. Please install JAX with `pip install jax[cpu]` to use this backend.")
        return _topk_jax(query_scores, k)
    else:
        if k < 0:
            raise ValueError(f"k must be a non-negative integer, but got {k}.")
        return _topk_numpy(query_scores, k, sorted)
