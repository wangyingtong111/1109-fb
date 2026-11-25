# Reusing Paths

If you expect to use a particular contraction repeatedly, it can make things simpler and more efficient not to compute the path each time.
Instead, supplying [`opt_einsum.contract_expression`](../api_reference.md#opt_einsumcontract_expression) with the contraction string and the shapes of the tensors generates a [`opt_einsum.ContractExpression`](../api_reference.md#opt_einsumcontractcontractexpression) which can then be repeatedly called with any matching set of arrays.
For example:

```python
my_expr = oe.contract_expression("abc,cd,dbe->ea", (2, 3, 4), (4, 5), (5, 3, 6))
print(my_expr)
#> <ContractExpression('abc,cd,dbe->ea')>
#>   1.  'dbe,cd->bce' [GEMM]
#>   2.  'bce,abc->ea' [GEMM]
```

The `ContractExpression` can be called with 3 arrays that match the original shapes without having to recompute the path:

```python
x, y, z = (np.random.rand(*s) for s in [(2, 3, 4), (4, 5), (5, 3, 6)])
my_expr(x, y, z)
#> array([[ 3.08331541,  4.13708916],
#>        [ 2.92793729,  4.57945185],
#>        [ 3.55679457,  5.56304115],
#>        [ 2.6208398 ,  4.39024187],
#>        [ 3.66736543,  5.41450334],
#>        [ 3.67772272,  5.46727192]])
```

Note that few checks are performed when calling the expression, and while it will work for a set of arrays with the same ranks as the original shapes but differing sizes, it might no longer be optimal.


## Shape Contracts and Dynamic Batch Axes

A batched matrix multiply is usually reused with a *varying* batch size, while
the contraction and channel dimensions stay fixed. Passing `dynamic_axes` to
[`opt_einsum.contract_expression`](../api_reference.md#opt_einsumcontract_expression)
compiles that intent into a shape contract:

```python
expr = oe.contract_expression(
    "...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), dynamic_axes=["..."]
)

expr(np.ones((5, 2, 3)), np.ones((1, 3, 4))).shape
#> (5, 2, 4)
```

The explicit labels (`i`, `j`, `k` here) are locked to the sizes they had in the
build shapes, while the broadcast batch label that `...` expands to may change
freely. This is checked from the shapes alone, so an invalid call is rejected
*before* the backend is entered and before an `out` buffer is written:

```python
expr(np.ones((5, 2, 6)), np.ones((1, 6, 4)))
#> ValueError: ... Operand 0 has size 6 for static label 'j' ...
```

The batch label is the one that `...` expands to, so it may sit at different axis
positions in different operands: in `"i...j,...jk->i...k"` it is axis 1 of the
first operand but axis 0 of the second, and only the sizes of the *same* label
are compared. Two variable batch sizes that are both non-1 and differ are
likewise rejected up front:

```python
expr = oe.contract_expression("...ij,...jk->...ik", (1, 2, 3), (1, 3, 4), dynamic_axes=True)
expr(np.ones((5, 2, 3)), np.ones((3, 3, 4)))
#> ValueError: ... Dynamic label 'c' is 5 in operand 0 but 3 in operand 1 ...
```

`dynamic_axes` accepts:

- `True` or `"auto"` to infer the variable labels (ellipsis batch labels, plus
  the label shared by every input and the output for matmul-like equations such
  as `b` in `"bij,bjk->bik"`),
- `False` to lock every label,
- a label or an iterable of labels (optionally including `Ellipsis`/`"..."`) to
  declare exactly which labels may vary,
- `{operand_index: axis_index}` to declare labels by axis position.

`shape_contract=True` is a shortcut for the inferred declaration, and
`shape_contract=False` turns the contract off completely. A declared contract is
enforced on every call; without a declaration plain calls stay permissive (as
before) but are still validated before an `out` buffer is written. Constant
tensors keep working: a constant whose batch size is `1` is evaluated once and
its cache is reused across different legal batch sizes.


## Specifying Constants

Often one generates contraction expressions where some of the tensor arguments
will remain *constant* across many calls.
[`opt_einsum.contract_expression`](../api_reference.md#opt_einsumcontract_expression) allows you to specify the indices of
these constant arguments, allowing `opt_einsum` to build and then reuse as
many constant contractions as possible.

Take for example the equation:

```python
eq = "ij,jk,kl,lm,mn->ni"
```

where we know that *only* the first and last tensors will vary between calls.
We can specify this by marking the middle three as constant - we then need to
supply the actual arrays rather than just the shapes to
[`opt_einsum.contract_expression`](../api_reference.md#opt_einsumcontract_expression):

```python
#           A       B       C       D       E
shapes = [(9, 5), (5, 5), (5, 5), (5, 5), (5, 8)]

# mark the middle three arrays as constant
constants = [1, 2, 3]

# generate the constant arrays
B, C, D = [np.random.randn(*shapes[i]) for i in constants]

# supplied ops are now mix of shapes and arrays
ops = (9, 5), B, C, D, (5, 8)

expr = oe.contract_expression(eq, *ops, constants=constants)
expr
#> <ContractExpression('ij,[jk,kl,lm],mn->ni', constants=[1, 2, 3])>
```

The expression now only takes the remaining two arrays as arguments (the
tensors with `'ij'` and `'mn'` indices), and will store as many reusable
constant contractions as possible.

.. code:: python

```python
A1, E1 = np.random.rand(*shapes[0]), np.random.rand(*shapes[-1])
out1 = expr(A1, E1)
out1.shap
#> (8, 9)

A2, E2 = np.random.rand(*shapes[0]), np.random.rand(*shapes[-1])
out2 = expr(A2, E2)
out2.shape
#> (8, 9)

np.allclose(out1, out2)
#> False

print(expr)
#> <ContractExpression('ij,[jk,kl,lm],mn->ni', constants=[1, 2, 3])>
#>   1.  'jm,mn->jn' [GEMM]
#>   2.  'jn,ij->ni' [GEMM]
```

Where we can see that the expression now only has to perform
two contractions to compute the output.

!!! note
    The constant part of an expression is lazily generated upon the first call
    (specific to each backend), though it can also be explicitly built by calling
    [`opt_einsum.contract.ContractExpression.evaluate_constants`](../api_reference.md#opt_einsumcontractcontractexpression).

We can confirm the advantage of using expressions and constants by timing the
following scenarios, first setting
`A = np.random.rand(*shapes[0])` and `E = np.random.rand(*shapes[-1])`.

### Contract from scratch

```python
%timeit oe.contract(eq, A, B, C, D, E)
#> 239 µs ± 5.06 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

### Contraction with an expression but no constants

```python
expr_no_consts = oe.contract_expression(eq, *shapes)
%timeit expr_no_consts(A, B, C, D, E)
#> 76.7 µs ± 2.47 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

### Contraction with an expression and constants marked

```python
%timeit expr(A, E)
#> 40.8 µs ± 1.22 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

Although this gives us a rough idea, of course the efficiency savings are
hugely dependent on the size of the contraction and number of possible constant
contractions.

We also note that even if there are *no* constant contractions to perform, it
can be very advantageous to specify constant tensors for particular backends.
For instance, if a GPU backend is used, the constant tensors will be kept on
the device rather than being transferred each time.
