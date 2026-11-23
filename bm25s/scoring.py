from collections import Counter
import math

import numpy as np
import os

def _faketqdm(*args, **kwargs):
    return args[0] if len(args) > 0 else None
try:
    if os.environ.get("DISABLE_TQDM", False):
        tqdm = _faketqdm
    else:
        from tqdm.auto import tqdm
except ImportError:
    tqdm = _faketqdm

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator to allow definition if numba is missing (code won't be used)
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

def _calculate_doc_freqs(
    corpus_tokens, unique_tokens, show_progress=True, leave_progress=False
) -> dict:
    """
    Document Frequency, aka DF, is the number of documents that contain a specific token.
    This function return a dictionary with the document frequency of each token, which is
    why it is called `doc_frequencies`.
    """
    unique_tokens = set(unique_tokens)

    # Now that we have all the unique tokens, we can count the number of
    # documents that contain each token
    doc_frequencies = {token: 0 for token in unique_tokens}

    for doc_tokens in tqdm(
        corpus_tokens,
        leave=leave_progress,
        disable=not show_progress,
        desc="BM25S Count Tokens",
    ):

        # get intersection of unique tokens and the tokens in the document
        shared_tokens = unique_tokens.intersection(doc_tokens)

        # for each token in the document, we increment the count of documents
        # This is a simple way to count the number of documents that contain each token
        for token in shared_tokens:
            doc_frequencies[token] += 1

    return doc_frequencies


def _calculate_doc_freqs_from_term_freqs(
    doc_term_freqs, unique_tokens, show_progress=True, leave_progress=False
):
    """
    Same as `_calculate_doc_freqs`, but over compressed documents.

    Each document is a ``(token_ids, term_frequencies, doc_length)`` tuple in which
    `token_ids` holds the distinct tokens of that document only. Alongside the document
    frequencies, this returns the total corpus length and the number of stored
    (document, token) entries; both are needed later and would otherwise require a
    second pass over the corpus.
    """
    unique_tokens = set(unique_tokens)
    doc_frequencies = {token: 0 for token in unique_tokens}
    total_doc_len = 0

    for token_ids, _, doc_len in tqdm(
        doc_term_freqs,
        leave=leave_progress,
        disable=not show_progress,
        desc="BM25S Count Tokens",
    ):
        total_doc_len += doc_len

        # Each distinct token of the document counts once, no matter its term frequency.
        for token in token_ids:
            if token in doc_frequencies:
                doc_frequencies[token] += 1

    n_entries = sum(doc_frequencies.values())
    return doc_frequencies, total_doc_len, n_entries


def _build_idf_array(
    doc_frequencies: dict,
    n_docs: int,
    compute_idf_fn: callable = None,
    dtype="float32",
) -> np.ndarray:
    n_vocab = len(doc_frequencies)
    idf_array = np.zeros(n_vocab, dtype=dtype)

    for token_id, df in doc_frequencies.items():
        if df != 0:
            idf_array[token_id] = compute_idf_fn(df, N=n_docs)

    return idf_array


def _build_nonoccurrence_array(
    doc_frequencies: dict,
    n_docs: int,
    compute_idf_fn: callable,
    calculate_tfc_fn: callable,
    l_d,
    l_avg,
    k1,
    b,
    delta,
    dtype="float32",
) -> np.ndarray:
    """
    The non-occurrence array is used to store the idf score for tokens that do not occur in the
    document. This is useful for BM25L and BM25+ variants, where we need to calculate the idf
    score for tokens that do not occur in the document, which will be used to calculate the
    final score.

    The nonoccurence array has length |V|, where V is the set of unique tokens in the corpus.

    The `compute_idf_fn` is the function to calculate the idf score for a token that does not occur
    in the document. The `calculate_tfc_fn` is the function to calculate the term frequency component
    of the BM25 score, which is used to calculate the final score for tokens that do not occur in the
    document.
    """
    n_vocab = len(doc_frequencies)
    nonoccurrence_array = np.zeros(n_vocab, dtype=dtype)

    for token_id, df in doc_frequencies.items():
        if df != 0:
            idf = compute_idf_fn(df, N=n_docs)
            tfc = calculate_tfc_fn(
                tf_array=0, l_d=l_d, l_avg=l_avg, k1=k1, b=b, delta=delta
            )
            nonoccurrence_array[token_id] = idf * tfc

    return nonoccurrence_array


def _score_tfc_robertson(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return tf_array / (k1 * ((1 - b) + b * l_d / l_avg) + tf_array)


def _score_tfc_lucene(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return _score_tfc_robertson(tf_array, l_d, l_avg, k1, b)


def _score_tfc_atire(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return (tf_array * (k1 + 1)) / (tf_array + k1 * (1 - b + b * l_d / l_avg))


def _score_tfc_bm25l(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    c_array = tf_array / (1 - b + b * l_d / l_avg)
    return ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)


def _score_tfc_bm25plus(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    num = (k1 + 1) * tf_array
    den = k1 * (1 - b + b * l_d / l_avg) + tf_array
    return (num / den) + delta


def _select_tfc_scorer(method) -> callable:
    if method == "robertson":
        return _score_tfc_robertson
    elif method == "lucene":
        return _score_tfc_lucene
    elif method == "atire":
        return _score_tfc_atire
    elif method == "bm25l":
        return _score_tfc_bm25l
    elif method == "bm25+":
        return _score_tfc_bm25plus
    else:
        error_msg = f"Invalid score_tfc value: {method}. Choose from 'robertson', 'lucene', 'atire'."
        raise ValueError(error_msg)


def _score_idf_robertson(df, N, allow_negative=False):
    """
    Computes the inverse document frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    inner = (N - df + 0.5) / (df + 0.5)
    if not allow_negative and inner < 1:
        inner = 1

    return math.log(inner)


def _score_idf_lucene(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def _score_idf_atire(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(N / df)


def _score_idf_bm25l(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / (df + 0.5))


def _score_idf_bm25plus(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / df)


def _select_idf_scorer(method) -> callable:
    if method == "robertson":
        return _score_idf_robertson
    elif method == "lucene":
        return _score_idf_lucene
    elif method == "atire":
        return _score_idf_atire
    elif method == "bm25l":
        return _score_idf_bm25l
    elif method == "bm25+":
        return _score_idf_bm25plus
    else:
        error_msg = f"Invalid score_idf_inner value: {method}. Choose from 'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'."
        raise ValueError(error_msg)


def _get_counts_from_token_ids(token_ids, dtype, int_dtype):
    token_counter = Counter(token_ids)
    voc_ind = np.array(list(token_counter.keys()), dtype=int_dtype)
    tf_array = np.array(list(token_counter.values()), dtype=dtype)

    return voc_ind, tf_array


def _iter_doc_term_freqs(corpus_token_ids, dtype="float32", int_dtype="int32"):
    """
    Lazily convert each document's token IDs into its compact term-frequency
    representation.

    Yields a ``(token_ids, term_frequencies, doc_length)`` tuple per document, where
    ``token_ids`` only contains the distinct tokens of the document. The original
    sequence of token IDs is never kept around, so the corpus can be consumed exactly
    once, even if it is a one-shot iterable.
    """
    for token_ids in corpus_token_ids:
        token_counter = Counter(token_ids)
        # `sum(...)` equals `len(token_ids)` but also works if the document is an
        # iterator that has already been consumed by `Counter`.
        doc_length = sum(token_counter.values())

        voc_ind = np.fromiter(
            token_counter.keys(), dtype=int_dtype, count=len(token_counter)
        )
        tf_array = np.fromiter(
            token_counter.values(), dtype=dtype, count=len(token_counter)
        )

        yield voc_ind, tf_array, doc_length


def _build_scores_and_indices_for_doc_term_freqs(
    doc_term_freqs,
    n_entries,
    idf_array,
    avg_doc_len,
    k1,
    b,
    delta,
    nonoccurrence_array,
    method="robertson",
    dtype="float32",
    int_dtype="int32",
    show_progress=True,
    leave_progress=False,
):
    """
    Build the flat score / document / vocabulary index arrays from a stream of
    per-document ``(token_ids, term_frequencies, doc_length)`` tuples.

    `n_entries` must be the exact number of (document, distinct token) pairs, which
    is also the total number of documents across all `doc_frequencies` entries.
    """
    # We create 3 arrays to store the scores, document indices, and vocabulary indices
    scores = np.empty(n_entries, dtype=dtype)
    doc_indices = np.empty(n_entries, dtype=int_dtype)
    voc_indices = np.empty(n_entries, dtype=int_dtype)

    calculate_tfc = _select_tfc_scorer(method)

    i = 0
    for doc_idx, (voc_ind_doc, tf_array, doc_len) in enumerate(
        tqdm(
            doc_term_freqs,
            desc="BM25S Compute Scores",
            disable=not show_progress,
            leave=leave_progress,
        )
    ):
        # Calculate the BM25 score for each token in the document
        tfc = calculate_tfc(
            tf_array=tf_array, l_d=doc_len, l_avg=avg_doc_len, k1=k1, b=b, delta=delta
        )
        idf = idf_array[voc_ind_doc]
        scores_doc = idf * tfc

        # If the method is uses a non-occurrence score array, then we need to subtract
        # the non-occurrence score from the scores
        if method in ("bm25l", "bm25+"):
            scores_doc -= nonoccurrence_array[voc_ind_doc]

        # Update the arrays with the new scores, document indices, and vocabulary indices
        doc_len = len(scores_doc)
        start, end = i, i + doc_len
        i = end

        doc_indices[start:end] = doc_idx
        voc_indices[start:end] = voc_ind_doc
        scores[start:end] = scores_doc

    return scores, doc_indices, voc_indices


def _build_scores_and_indices_for_matrix(
    corpus_token_ids,
    idf_array,
    avg_doc_len,
    doc_frequencies,
    k1,
    b,
    delta,
    nonoccurrence_array,
    method="robertson",
    dtype="float32",
    int_dtype="int32",
    show_progress=True,
    leave_progress=False,
):
    array_size = sum(doc_frequencies.values())

    return _build_scores_and_indices_for_doc_term_freqs(
        doc_term_freqs=_iter_doc_term_freqs(
            corpus_token_ids, dtype=dtype, int_dtype=int_dtype
        ),
        n_entries=array_size,
        idf_array=idf_array,
        avg_doc_len=avg_doc_len,
        k1=k1,
        b=b,
        delta=delta,
        nonoccurrence_array=nonoccurrence_array,
        method=method,
        dtype=dtype,
        int_dtype=int_dtype,
        show_progress=show_progress,
        leave_progress=leave_progress,
    )


def _compute_relevance_from_scores_legacy(
    data, indptr, indices, num_docs, query_tokens_ids, dtype
):
    """
    The legacy implementation of the `_compute_relevance_from_scores` function. This may
    be faster than the new implementation for some cases, but it cannot benefit from
    numba acceleration, as it uses python lists. This function is kept for reference
    and comparison purposes.
    """
    # First, we use the query_token_ids to select the relevant columns from the score_matrix
    query_tokens_ids = np.array(query_tokens_ids, dtype=int)
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]

    scores_lists = []
    indices_lists = []

    for i, (start, end) in enumerate(zip(indptr_starts, indptr_ends)):
        scores_lists.append(data[start:end])
        indices_lists.append(indices[start:end])

    # combine the lists into a single array

    scores = np.zeros(num_docs, dtype=dtype)
    if len(scores_lists) == 0:
        return scores

    scores_flat = np.concatenate(scores_lists)
    indices_flat = np.concatenate(indices_lists)
    np.add.at(scores, indices_flat, scores_flat)

    return scores

def _compute_relevance_from_scores_jit_ready(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    query_tokens_ids: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    This internal static function calculates the relevance scores for a given query,
    by using the BM25 scores that have been precomputed in the BM25 eager index.
    This version is ready for JIT compilation with numba, but is slow if not compiled.
    """
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]

    scores = np.zeros(num_docs, dtype=dtype)
    for i in range(len(query_tokens_ids)):
        start, end = indptr_starts[i], indptr_ends[i]
        # The following code is slower with numpy, but faster after JIT compilation
        for j in range(start, end):
            scores[indices[j]] += data[j]

    return scores


def _np_csc_jit_ready(data, rows, cols, shape):
    """
    Numba-compilable core implementation of CSC construction.
    Uses Counting Sort (linear time) instead of Argsort (linear-logarithmic time).
    
    This implementation assumes `rows` (doc_ids) are sorted in the input, which is 
    standard for BM25 construction. It performs a stable sort on `cols` to strictly 
    guarantee the final (col, row) sorted order required by CSC matrices.
    """
    n_cols = shape[1]
    n_items = len(data)

    # 1. Compute column counts (histogram)
    col_counts = np.bincount(cols, minlength=n_cols)
    
    # 2. Compute indptr and initialize write pointers (heads)
    # indptr: start/end of each column
    # heads: current write position for each column (mutable)
    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    heads = np.zeros(n_cols, dtype=np.int64)
    
    acc = 0
    for i in range(n_cols):
        heads[i] = acc
        acc += col_counts[i]
        indptr[i+1] = acc

    # 3. Perform Counting Sort to fill data and indices
    # Since 'heads' tracks the write position, we can fill the output arrays in one pass.
    sorted_data = np.zeros_like(data)
    sorted_indices = np.zeros_like(rows)

    for i in range(n_items):
        col = cols[i]
        pos = heads[col]
        sorted_data[pos] = data[i]
        sorted_indices[pos] = rows[i]
        heads[col] += 1
    
    return sorted_data, sorted_indices, indptr

def _np_csc_python(data, rows, cols, shape):
    """
    Pure NumPy implementation of CSC construction.
    """
    n_cols = shape[1]

    # 1. Compute indptr
    col_counts = np.bincount(cols, minlength=n_cols)

    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    np.cumsum(col_counts, out=indptr[1:])

    # 2. Sort the data
    packed_indices = (cols.astype(np.int64) << 32) | rows.astype(np.int64)
    sorter = np.argsort(packed_indices)

    # 3. Reorder arrays based on the sort
    sorted_data = data[sorter]
    sorted_indices = rows[sorter]
    
    return sorted_data, sorted_indices, indptr
