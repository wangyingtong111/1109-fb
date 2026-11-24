"""Main init function for opt_einsum."""

from opt_einsum import blas, helpers, path_random, paths
from opt_einsum._version import __version__
from opt_einsum.contract import contract, contract_expression, contract_path
from opt_einsum.parser import get_symbol
from opt_einsum.path_random import RandomGreedy
from opt_einsum.paths import BranchBound, DynamicProgramming, FixedNodeStep
from opt_einsum.sharing import shared_intermediates
from opt_einsum.typing import FixedNodePathType

__all__ = [
    "__version__",
    "blas",
    "helpers",
    "path_random",
    "paths",
    "contract",
    "contract_expression",
    "contract_path",
    "get_symbol",
    "RandomGreedy",
    "BranchBound",
    "DynamicProgramming",
    "FixedNodeStep",
    "FixedNodePathType",
    "shared_intermediates",
]


paths.register_path_fn("random-greedy", path_random.random_greedy)
paths.register_path_fn("random-greedy-128", path_random.random_greedy_128)
