import itertools
import weakref
from collections import Counter
from typing import Any

import pytest

from opt_einsum import contract, contract_expression, contract_path, get_symbol, shared_intermediates
from opt_einsum.backends import to_cupy, to_torch
from opt_einsum.contract import _einsum
from opt_einsum.parser import parse_einsum_input
from opt_einsum.sharing import count_cached_ops, currently_sharing, get_sharing_cache
from opt_einsum.testing import build_views
from opt_einsum.typing import BackendType

pytest.importorskip("numpy")

try:
    import numpy as np  # type: ignore

    numpy_if_found = "numpy"
except ImportError:
    numpy_if_found = pytest.param("numpy", marks=[pytest.mark.skip(reason="NumPy not installed.")])  # type: ignore

try:
    import cupy  # noqa

    cupy_if_found = "cupy"
except ImportError:
    cupy_if_found = pytest.param("cupy", marks=[pytest.mark.skip(reason="CuPy not installed.")])  # type: ignore

try:
    import torch  # type: ignore # noqa

    torch_if_found = "torch"
except ImportError:
    torch_if_found = pytest.param("torch", marks=[pytest.mark.skip(reason="PyTorch not installed.")])  # type: ignore

backends = [numpy_if_found, torch_if_found, cupy_if_found]
equations = [
    "ab,bc->ca",
    "abc,bcd,dea",
    "abc,def->fedcba",
    "abc,bcd,df->fa",
    # test 'prefer einsum' ops
    "ijk,ikj",
    "i,j->ij",
    "ijk,k->ij",
    "AB,BC->CA",
]
to_backend = {
    "numpy": lambda x: x,
    "torch": to_torch,
    "cupy": to_cupy,
}


@pytest.mark.parametrize("eq", equations)
@pytest.mark.parametrize("backend", backends)
def test_sharing_value(eq: str, backend: BackendType) -> None:
    views = build_views(eq)
    shapes = [v.shape for v in views]
    expr = contract_expression(eq, *shapes)

    expected = expr(*views, backend=backend)
    with shared_intermediates():
        actual = expr(*views, backend=backend)

    assert (actual == expected).all()


@pytest.mark.parametrize("optimize_kwargs", [{"optimize": False}, {"use_blas": False}])
def test_sharing_dtype_out_identity_and_repair(optimize_kwargs: dict) -> None:
    """dtype/casting must be honored while sharing, and writing into ``out``
    must not be skipped on a cache hit: repeating the exact same call after
    manually corrupting ``out`` must restore the correct result and return the
    very same output array.
    """
    # the int8 reduction overflows (true sum 200 vs int8 wrap-around -56), so
    # the accumulation dtype materially changes the result
    x = np.full(100, 2, dtype=np.int8)
    eq = "a->"
    true_sum = int(x.astype(np.int64).sum())

    expected_i64 = contract(eq, x, dtype=np.int64, **optimize_kwargs)
    expected_f64 = contract(eq, x, dtype=np.float64, **optimize_kwargs)

    with shared_intermediates():
        # first contract the same inputs in int64, then request float64 into
        # an existing output buffer -- the cached int64 result must not leak
        result_i64 = contract(eq, x, dtype=np.int64, **optimize_kwargs)
        assert result_i64.dtype == np.int64
        assert int(result_i64) == true_sum

        out = np.zeros((), dtype=np.float64)
        result_f64 = contract(eq, x, out=out, dtype=np.float64, **optimize_kwargs)
        assert result_f64 is out
        assert out.dtype == np.float64
        assert float(out) == float(expected_f64)
        assert np.array_equal(result_i64, expected_i64)

        # corrupt the output, then issue the identical call again
        out.fill(np.nan)
        repeated = contract(eq, x, out=out, dtype=np.float64, **optimize_kwargs)
        assert repeated is out
        assert out.dtype == np.float64
        assert float(out) == float(expected_f64)


def test_sharing_dtype_and_casting_key_separation() -> None:
    """Changing dtype or casting produces a distinct computation and must not
    hit a non-equivalent cached result, while semantically identical options
    and explicit defaults must keep sharing intermediates.
    """
    x = np.full(100, 2, dtype=np.int8)
    eq = "a->"

    with shared_intermediates() as cache:
        r_i64 = contract(eq, x, optimize=False, dtype=np.int64)
        r_f64 = contract(eq, x, optimize=False, dtype=np.float64)
        r_f64_dtype_obj = contract(eq, x, optimize=False, dtype=np.dtype("float64"))
        r_f64_explicit_defaults = contract(
            eq, x, optimize=False, dtype=np.float64, casting="safe", order="K"
        )

        assert r_i64.dtype == np.int64 and int(r_i64) == 200
        assert r_f64.dtype == np.float64 and float(r_f64) == 200.0
        assert float(r_f64_dtype_obj) == 200.0
        assert float(r_f64_explicit_defaults) == 200.0

        einsum_keys = [k for k in cache if k[0] == "einsum"]
        option_keys = {k[-1] for k in einsum_keys}
        # int64 vs float64 are distinct entries ...
        assert (("dtype", np.dtype("int64")),) in option_keys
        assert (("dtype", np.dtype("float64")),) in option_keys
        # ... whereas np.float64 / np.dtype('float64') and explicit defaults
        # must not fragment the cache
        assert option_keys == {
            (("dtype", np.dtype("int64")),),
            (("dtype", np.dtype("float64")),),
        }
        # ``out`` must never appear in a cache key
        assert all(all(item[0] != "out" for item in k[-1]) for k in einsum_keys)

    with shared_intermediates() as cache:
        out = np.zeros((), dtype=np.int8)
        # a different casting mode changes the computation, hence the key
        result = contract(eq, x, out=out, optimize=False, dtype=np.int8, casting="unsafe")
        assert result is out
        assert np.array_equal(
            out, np.einsum(eq, x, dtype=np.int8, casting="unsafe")
        )
        option_keys = {k[-1] for k in cache if k[0] == "einsum"}
        assert (
            ("casting", "unsafe"),
            ("dtype", np.dtype("int8")),
        ) in option_keys


def test_sharing_out_does_not_poison_intermediates() -> None:
    """Equivalent equations without ``out`` keep sharing intermediates with
    contractions that used ``out``, and cached tensors stay independent of the
    user-supplied output buffer.
    """
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 6, size=(3, 4)).astype(np.int8)
    b = rng.integers(-5, 6, size=(4, 5)).astype(np.int8)
    c = rng.integers(-5, 6, size=(5, 6)).astype(np.int8)
    eq = "ab,bc,cd->ad"

    expected = contract(eq, a, b, c, use_blas=False, dtype=np.int64)

    with shared_intermediates() as cache:
        out = np.zeros((3, 6), dtype=np.int64)
        result = contract(eq, a, b, c, out=out, use_blas=False, dtype=np.int64)
        assert result is out
        assert np.array_equal(out, expected)

        cached_ops_after_out = count_cached_ops(cache)["einsum"]

        # the equivalent call without ``out`` must hit the same intermediates
        shared = contract(eq, a, b, c, use_blas=False, dtype=np.int64)
        assert count_cached_ops(cache)["einsum"] == cached_ops_after_out
        assert shared.dtype == np.int64
        assert np.array_equal(shared, expected)
        assert shared is not out

        # mutating the user output must not alter the cached intermediate
        out[:] = -1
        assert np.array_equal(shared, expected)

        # a repaired repeated call still returns the same output identity
        repeated = contract(eq, a, b, c, out=out, use_blas=False, dtype=np.int64)
        assert repeated is out
        assert np.array_equal(out, expected)


@pytest.mark.parametrize("backend", backends)
def test_complete_sharing(backend: BackendType) -> None:
    eq = "ab,bc,cd->"
    views = build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))

    print("-" * 40)
    print("Without sharing:")
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)

    print("-" * 40)
    print("With sharing:")
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expr(*views, backend=backend)
        actual = count_cached_ops(cache)

    print("-" * 40)
    print(f"Without sharing: {expected} expressions")
    print(f"With sharing: {actual} expressions")
    assert actual == expected


@pytest.mark.parametrize("backend", backends)
def test_sharing_reused_cache(backend: BackendType) -> None:
    eq = "ab,bc,cd->"
    views = build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))

    print("-" * 40)
    print("Without sharing:")
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)

    print("-" * 40)
    print("With sharing:")
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
    with shared_intermediates(cache):
        expr(*views, backend=backend)
        actual = count_cached_ops(cache)

    print("-" * 40)
    print(f"Without sharing: {expected} expressions")
    print(f"With sharing: {actual} expressions")
    assert actual == expected


@pytest.mark.parametrize("backend", backends)
def test_no_sharing_separate_cache(backend: BackendType) -> None:
    eq = "ab,bc,cd->"
    views = build_views(eq)
    expr = contract_expression(eq, *(v.shape for v in views))

    print("-" * 40)
    print("Without sharing:")
    with shared_intermediates() as cache:
        expr(*views, backend=backend)
        expected = count_cached_ops(cache)
        expected.update(count_cached_ops(cache))  # we expect double

    print("-" * 40)
    print("With sharing:")
    with shared_intermediates() as cache1:
        expr(*views, backend=backend)
        actual = count_cached_ops(cache1)
    with shared_intermediates() as cache2:
        expr(*views, backend=backend)
        actual.update(count_cached_ops(cache2))

    print("-" * 40)
    print(f"Without sharing: {expected} expressions")
    print(f"With sharing: {actual} expressions")
    assert actual == expected


@pytest.mark.parametrize("backend", backends)
def test_sharing_nesting(backend: BackendType) -> None:
    eqs = ["ab,bc,cd->a", "ab,bc,cd->b", "ab,bc,cd->c", "ab,bc,cd->c"]
    views = build_views(eqs[0])
    shapes = [v.shape for v in views]
    refs: Any = weakref.WeakValueDictionary()

    def method1(views):
        with shared_intermediates():
            w = contract_expression(eqs[0], *shapes)(*views, backend=backend)
            x = contract_expression(eqs[2], *shapes)(*views, backend=backend)
            result = contract_expression("a,b->", w.shape, x.shape)(w, x, backend=backend)
            refs["w"] = w
            refs["x"] = x
            del w, x
            assert "w" in refs
            assert "x" in refs
        assert "w" not in refs, "cache leakage"
        assert "x" not in refs, "cache leakage"
        return result

    def method2(views):
        with shared_intermediates():
            y = contract_expression(eqs[2], *shapes)(*views, backend=backend)
            z = contract_expression(eqs[3], *shapes)(*views, backend=backend)
            refs["y"] = y
            refs["z"] = z
            result = contract_expression("c,d->", y.shape, z.shape)(y, z, backend=backend)
            result = result + method1(views)  # nest method1 in method2
            del y, z
            assert "y" in refs
            assert "z" in refs
        assert "y" not in refs
        assert "z" not in refs

    method1(views)
    method2(views)


@pytest.mark.parametrize("eq", equations)
@pytest.mark.parametrize("backend", backends)
def test_sharing_modulo_commutativity(eq: str, backend: BackendType) -> None:
    ops = tuple(to_backend[backend](x) for x in build_views(eq))
    inputs, output, _ = parse_einsum_input([eq] + list(ops))
    inputs_list = inputs.split(",")

    print("-" * 40)
    print("Without sharing:")
    with shared_intermediates() as cache:
        _einsum(eq, *ops, backend=backend)
        expected = count_cached_ops(cache)

    print("-" * 40)
    print("With sharing:")
    with shared_intermediates() as cache:
        for permuted in itertools.permutations(zip(inputs_list, ops)):
            permuted_inputs = [p[0] for p in permuted]
            permuted_ops = [p[1] for p in permuted]
            permuted_eq = "{}->{}".format(",".join(permuted_inputs), output)
            _einsum(permuted_eq, *permuted_ops, backend=backend)
        actual = count_cached_ops(cache)

    print("-" * 40)
    print(f"Without sharing: {expected} expressions")
    print(f"With sharing: {actual} expressions")
    assert actual == expected


@pytest.mark.parametrize("backend", backends)
def test_partial_sharing(backend: BackendType) -> None:
    eq = "ab,bc,de->"
    x, y, z1 = build_views(eq)  # type: ignore
    z2 = 2.0 * z1 - 1.0
    expr = contract_expression(eq, x.shape, y.shape, z1.shape)

    print("-" * 40)
    print("Without sharing:")
    num_exprs_nosharing: Any = Counter()
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))
    with shared_intermediates() as cache:
        expr(x, y, z2, backend=backend)
        num_exprs_nosharing.update(count_cached_ops(cache))

    print("-" * 40)
    print("With sharing:")
    with shared_intermediates() as cache:
        expr(x, y, z1, backend=backend)
        expr(x, y, z2, backend=backend)
        num_exprs_sharing = count_cached_ops(cache)

    print("-" * 40)
    print(f"Without sharing: {num_exprs_nosharing} expressions")
    print(f"With sharing: {num_exprs_sharing} expressions")
    assert num_exprs_nosharing["einsum"] > num_exprs_sharing["einsum"]


@pytest.mark.parametrize("backend", backends)
def test_sharing_with_constants(backend: BackendType) -> None:
    inputs = "ij,jk,kl"
    outputs = "ijkl"
    equations = [f"{inputs}->{output}" for output in outputs]
    shapes = (2, 3), (3, 4), (4, 5)
    constants = {0, 2}
    ops = [np.random.rand(*shp) if i in constants else shp for i, shp in enumerate(shapes)]
    var = np.random.rand(*shapes[1])

    expected = [contract_expression(eq, *shapes)(ops[0], var, ops[2]) for eq in equations]

    with shared_intermediates():
        actual = [contract_expression(eq, *ops, constants=constants)(var) for eq in equations]

    for dim, expected_dim, actual_dim in zip(outputs, expected, actual):
        assert np.allclose(expected_dim, actual_dim), f"error at {dim}"


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("backend", backends)
def test_chain(size: int, backend: BackendType) -> None:
    xs = [np.random.rand(2, 2) for _ in range(size)]
    shapes = [x.shape for x in xs]
    alphabet = "".join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i : i + 2] for i in range(size)]
    inputs = ",".join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            eq = f"{inputs}->{target}"
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *shapes)
            expr(*xs, backend=backend)
        print("-" * 40)


@pytest.mark.parametrize("size", [3, 4, 5, 10])
@pytest.mark.parametrize("backend", backends)
def test_chain_2(size: int, backend: BackendType) -> None:
    xs = [np.random.rand(2, 2) for _ in range(size)]
    shapes = [x.shape for x in xs]
    alphabet = "".join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i : i + 2] for i in range(size)]
    inputs = ",".join(names)

    with shared_intermediates():
        print(inputs)
        for i in range(size):
            target = alphabet[i : i + 2]
            eq = f"{inputs}->{target}"
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *shapes)
            expr(*xs, backend=backend)
        print("-" * 40)


def _compute_cost(cache):
    counts = count_cached_ops(cache)
    return counts["einsum"] + counts["tensordot"]


@pytest.mark.parametrize("backend", backends)
def test_chain_2_growth(backend: BackendType) -> None:
    sizes = list(range(1, 21))
    costs = []
    for size in sizes:
        xs = [np.random.rand(2, 2) for _ in range(size)]
        alphabet = "".join(get_symbol(i) for i in range(size + 1))
        names = [alphabet[i : i + 2] for i in range(size)]
        inputs = ",".join(names)

        with shared_intermediates() as cache:
            for i in range(size):
                target = alphabet[i : i + 2]
                eq = f"{inputs}->{target}"
                expr = contract_expression(eq, *(x.shape for x in xs))
                expr(*xs, backend=backend)
            costs.append(_compute_cost(cache))

    print(f"sizes = {repr(sizes)}")
    print(f"costs = {repr(costs)}")
    for size, cost in zip(sizes, costs):
        print(f"{size}\t{cost}")


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("backend", backends)
def test_chain_sharing(size: int, backend: BackendType) -> None:
    xs = [np.random.rand(2, 2) for _ in range(size)]
    alphabet = "".join(get_symbol(i) for i in range(size + 1))
    names = [alphabet[i : i + 2] for i in range(size)]
    inputs = ",".join(names)

    num_exprs_nosharing = 0
    for i in range(size + 1):
        with shared_intermediates() as cache:
            target = alphabet[i]
            eq = f"{inputs}->{target}"
            expr = contract_expression(eq, *tuple(x.shape for x in xs))
            expr(*xs, backend=backend)
            num_exprs_nosharing += _compute_cost(cache)

    with shared_intermediates() as cache:
        print(inputs)
        for i in range(size + 1):
            target = alphabet[i]
            eq = f"{inputs}->{target}"
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *[x.shape for x in xs])
            expr(*xs, backend=backend)
        num_exprs_sharing = _compute_cost(cache)

    print("-" * 40)
    print(f"Without sharing: {num_exprs_nosharing} expressions")
    print(f"With sharing: {num_exprs_sharing} expressions")
    assert num_exprs_nosharing > num_exprs_sharing


def test_multithreaded_sharing() -> None:
    from multiprocessing.pool import ThreadPool

    def fn():
        x, y, z = build_views("ab,bc,cd")

        with shared_intermediates():
            contract("ab,bc,cd->a", x, y, z)
            contract("ab,bc,cd->b", x, y, z)

            return len(get_sharing_cache())

    expected = fn()
    pool = ThreadPool(8)
    fs = [pool.apply_async(fn) for _ in range(16)]
    assert not currently_sharing()
    assert [f.get() for f in fs] == [expected] * 16
    pool.close()
