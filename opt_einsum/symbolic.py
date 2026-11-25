"""Symbolic dimension expressions for shape-only contraction planning.

The objects in this module are intentionally tiny: they preserve a dimension
name or a broadcasting constraint until concrete bindings are available.
They do not attempt to be a general computer-algebra system.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "Broadcast",
    "Dimension",
    "DimensionConflictError",
    "MaxExpr",
    "Ratio",
    "Sum",
    "Symbol",
    "SymbolicDimension",
    "SymbolicExpression",
    "SymbolicInt",
    "SymbolicPlanningError",
    "SymbolicSize",
    "UnresolvedSymbolError",
    "UnresolvedSymbolicDimensionError",
    "add",
    "bind_plan",
    "broadcast_dimensions",
    "coerce_dimension",
    "multiply",
    "normalize_bindings",
    "symbol",
    "symbols",
    "symbolic_max",
]


class SymbolicPlanningError(ValueError):
    """Base class for symbolic shape planning errors."""


class UnresolvedSymbolError(SymbolicPlanningError, TypeError):
    """An operation needs concrete values for unbound symbols."""


# Descriptive alias used by callers who prefer the explicit name.
UnresolvedSymbolicDimensionError = UnresolvedSymbolError


class DimensionConflictError(SymbolicPlanningError):
    """Concrete or bound dimensions for one einsum label are incompatible."""


def normalize_bindings(bindings: Mapping[Any, Any] | None = None, **kwargs: Any) -> dict[str, int]:
    """Normalize symbolic bindings to ``{name: non_negative_int}``."""
    merged: dict[Any, Any] = {}
    if bindings is not None:
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping of symbol names to integer sizes")
        merged.update(bindings)
    merged.update(kwargs)

    normalized: dict[str, int] = {}
    for key, value in merged.items():
        name = key.name if isinstance(key, Symbol) else str(key)
        try:
            size = operator.index(value)
        except TypeError:
            try:
                size = int(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Binding for {name!r} must be an integer size, got {value!r}") from exc
        if size < 0:
            raise ValueError(f"Binding for {name!r} must be non-negative, got {size}")
        normalized[name] = size
    return normalized


def collect_bindings(
    bindings: Mapping[Any, Any] | Any = None,
    value: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Normalize the flexible binding arguments used by ``bind`` and ``subs``."""
    raw_bindings: dict[Any, Any] = {}
    if isinstance(bindings, Mapping):
        raw_bindings.update(bindings)
    elif bindings is not None and value is not None:
        raw_bindings[bindings] = value
    elif bindings is not None:
        raw_bindings.update(dict(bindings))
    if extra:
        raw_bindings.update(extra)
    return normalize_bindings(raw_bindings)


def coerce_dimension(value: Any, bindings: Mapping[str, Any] | None = None) -> int | SymbolicExpression:
    """Coerce a shape entry to an int or symbolic expression.

    Non-numeric strings are treated as named symbols.  Numeric strings retain
    their historical integer interpretation.
    """
    if isinstance(value, SymbolicExpression):
        if bindings is not None and value.free_symbols <= set(bindings):
            return value.bind(bindings)
        return value

    if isinstance(value, str):
        text = value.strip()
        try:
            value = int(text)
        except ValueError:
            expr: SymbolicExpression = Symbol(text)
            if bindings is not None and expr.free_symbols <= set(bindings):
                return expr.bind(bindings)
            return expr

    try:
        size = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"Shape dimensions must be integers or named symbols, got {value!r}") from exc

    if size < 0:
        raise ValueError(f"Shape dimensions must be non-negative, got {size}")
    return size


class _CompatibilityText(str):
    """String whose equality treats ``×``/``*`` and spacing as equivalent."""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return NotImplemented
        return _text_key(self) == _text_key(other)

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        return hash(_text_key(self))


class SymbolicExpression(str):
    """Base class for small symbolic size expressions.

    Expressions are also strings containing their readable Unicode form.  This
    makes reports/JSON serialization straightforward while still allowing
    callers to bind and evaluate the retained symbols.
    """

    __slots__ = ()

    def __new__(cls, *args: Any, **_kwargs: Any) -> "SymbolicExpression":
        # ``SymbolicDimension("B")`` is a convenient alias for ``Symbol("B")``.
        if cls is SymbolicExpression and args and isinstance(args[0], str):
            return Symbol(args[0])
        text = args[0] if args and isinstance(args[0], str) else ""
        return str.__new__(cls, text)

    @property
    def free_symbols(self) -> frozenset[str]:
        return frozenset()

    @property
    def symbol_names(self) -> tuple[str, ...]:
        """Free symbol names in stable order."""
        return tuple(sorted(self.free_symbols))

    def _evaluate(self, bindings: Mapping[str, int]) -> int | float:
        raise NotImplementedError

    def bind(self, bindings: Mapping[Any, Any] | Any = None, value: Any = None, **kwargs: Any) -> int | float:
        """Bind all symbols and return a concrete number.

        Accepts a mapping (``bind({"B": 5})``), keywords (``bind(B=5)``), or a
        name/value pair (``bind("B", 5)``).
        """
        values = collect_bindings(bindings, value, kwargs)
        try:
            return self._evaluate(values)
        except KeyError as exc:
            name = exc.args[0] if exc.args else "<unknown>"
            raise UnresolvedSymbolError(
                f"Symbol {name!r} is unbound; cannot evaluate {self.display_text()}. "
                "Provide it in bindings=... or as a keyword argument."
            ) from exc

    def _substitute(self, values: Mapping[str, int]) -> Any:
        return self

    def subs(self, bindings: Mapping[Any, Any] | Any = None, value: Any = None, **kwargs: Any) -> Any:
        """Partially or fully substitute bindings and simplify constraints."""
        values = collect_bindings(bindings, value, kwargs)
        return self._substitute(values)

    def __call__(self, *args: Any, **kwargs: Any) -> int | float:
        return self.bind(*args, **kwargs)

    evaluate = bind
    eval = bind
    resolve = bind
    resolve_bindings = bind
    materialize = bind
    substitute = subs

    def display_text(self, *, unicode: bool = True, compact: bool = True) -> str:
        raise NotImplementedError

    def ascii_text(self) -> str:
        return self.display_text(unicode=False)

    def _comparison_error(self) -> UnresolvedSymbolError:
        symbols = ", ".join(sorted(self.free_symbols)) or "<unknown>"
        return UnresolvedSymbolError(
            f"Cannot compare an unbound symbolic size ({self.display_text()}); free symbol(s): {symbols}. "
            "Bind the dimensions first or use an explicit fixed path without an unresolved budget/priority comparison."
        )

    def __int__(self) -> int:
        value = self.bind()
        return int(value)

    def __index__(self) -> int:
        return int(self)

    def __float__(self) -> float:
        return float(self.bind())

    def __bool__(self) -> bool:
        # Symbolic expressions are valid objects; do not pretend to know zero-ness.
        return True

    def _compare(self, other: Any, operation: str) -> bool:
        try:
            value = self.bind()
        except UnresolvedSymbolError as exc:
            raise self._comparison_error() from exc
        result = getattr(value, operation)(other)
        if result is NotImplemented:
            return NotImplemented
        return bool(result)

    def __lt__(self, other: Any) -> bool:
        return self._compare(other, "__lt__")

    def __le__(self, other: Any) -> bool:
        return self._compare(other, "__le__")

    def __gt__(self, other: Any) -> bool:
        return self._compare(other, "__gt__")

    def __ge__(self, other: Any) -> bool:
        return self._compare(other, "__ge__")

    def __mul__(self, other: Any) -> SymbolicExpression:
        return multiply(self, other)

    __rmul__ = __mul__

    def __add__(self, other: Any) -> SymbolicExpression:
        return add(self, other)

    __radd__ = __add__

    def __truediv__(self, other: Any) -> Ratio:
        return Ratio(self, other)

    def __rtruediv__(self, other: Any) -> Ratio:
        return Ratio(other, self)

    def _key(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(_text_key(self.display_text()))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return _text_key(self.display_text()) == _text_key(other)
        if isinstance(other, SymbolicExpression):
            return self._key() == other._key()
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            try:
                return self.bind() == other
            except UnresolvedSymbolError:
                return False
        return NotImplemented

    def __repr__(self) -> str:
        return self.display_text(unicode=True)

    def __str__(self) -> str:
        return _CompatibilityText(self.display_text(unicode=True))

    def __format__(self, format_spec: str) -> str:
        # Numeric specs such as ".3e" cannot be applied until bound.  Showing
        # the expression is more useful than raising from PathInfo printing.
        if not format_spec:
            return self.display_text()
        try:
            return format(self.bind(), format_spec)
        except UnresolvedSymbolError:
            return self.display_text()


# A descriptive base alias.
SymbolicDimension = SymbolicExpression


def _text_key(text: str) -> str:
    return "".join(str(text).replace("×", "*").split()).lower()


class Symbol(SymbolicExpression):
    """A named, bindable dimension."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = str(name)

    @property
    def free_symbols(self) -> frozenset[str]:
        return frozenset({self.name})

    def _evaluate(self, bindings: Mapping[str, int]) -> int:
        if self.name not in bindings:
            raise KeyError(self.name)
        return int(bindings[self.name])

    def _substitute(self, values: Mapping[str, int]) -> Any:
        return values[self.name] if self.name in values else self

    def display_text(self, *, unicode: bool = True, compact: bool = True) -> str:
        return self.name

    def _key(self) -> tuple[Any, ...]:
        return ("symbol", self.name)


def symbol(name: str) -> Symbol:
    """Create a named symbolic dimension."""
    return Symbol(name)


def symbols(names: str | Iterable[str]) -> tuple[Symbol, ...]:
    """Create several symbols from whitespace/comma-separated names or an iterable."""
    if isinstance(names, str):
        normalized = names.replace(",", " ")
        items = normalized.split()
    else:
        items = list(names)
    return tuple(Symbol(str(name).strip()) for name in items if str(name).strip())


# Concise/domain aliases used by different naming conventions.
Dimension = Symbol
SymbolicInt = SymbolicExpression
UnboundSymbolicDimension = UnresolvedSymbolError


class Broadcast(SymbolicExpression):
    """A NumPy/einsum broadcasting constraint for one label."""

    __slots__ = ("dimensions", "label")

    def __new__(cls, *dimensions: Any, label: str | None = None) -> "Broadcast":
        coerced = tuple(coerce_dimension(dim) for dim in dimensions)
        inner = ",".join(_display(dim, unicode=True, compact=True) for dim in coerced)
        return str.__new__(cls, f"broadcast({inner})")

    def __init__(self, *dimensions: Any, label: str | None = None):
        self.dimensions = tuple(coerce_dimension(d) for d in dimensions)
        self.label = label

    @property
    def free_symbols(self) -> frozenset[str]:
        result: set[str] = set()
        for dim in self.dimensions:
            if isinstance(dim, SymbolicExpression):
                result.update(dim.free_symbols)
        return frozenset(result)

    def _evaluate(self, bindings: Mapping[str, int]) -> int:
        values = [dim.bind(bindings) if isinstance(dim, SymbolicExpression) else int(dim) for dim in self.dimensions]
        return broadcast_values(values, self.label)

    def _substitute(self, values: Mapping[str, int]) -> Any:
        substituted = [dim._substitute(values) if isinstance(dim, SymbolicExpression) else dim for dim in self.dimensions]
        return broadcast_dimensions(substituted, label=self.label)

    def display_text(self, *, unicode: bool = True, compact: bool = True) -> str:
        sep = "," if compact else ", "
        inner = sep.join(_display(d, unicode=unicode, compact=compact) for d in self.dimensions)
        return f"broadcast({inner})"

    def _key(self) -> tuple[Any, ...]:
        return ("broadcast", tuple(_expr_key(d) for d in self.dimensions))


def _expr_key(value: Any) -> Any:
    if isinstance(value, SymbolicExpression):
        return value._key()
    return ("const", int(value))


def _display(value: Any, *, unicode: bool, compact: bool) -> str:
    if isinstance(value, SymbolicExpression):
        return value.display_text(unicode=unicode, compact=compact)
    return str(value)


def broadcast_values(values: list[int], label: str | None = None) -> int:
    """Return the broadcast result, retaining zero when paired with one."""
    non_unit = [value for value in values if value != 1]
    if not non_unit:
        return 1

    result = non_unit[0]
    if any(value != result for value in non_unit[1:]):
        label_text = repr(label) if label is not None else "<dimension>"
        raise DimensionConflictError(
            f"Size of label {label_text} does not match previous terms: incompatible sizes {list(values)!r}. "
            "Dimensions must be equal, or one side may be 1; size 1 broadcasts to the other "
            "size, so 0 with 1 resolves to 0 rather than max(0, 1). Label conflict / 标签冲突."
        )
    return int(result)


def broadcast_dimensions(dimensions: list[Any] | tuple[Any, ...], label: str | None = None) -> Any:
    """Normalize a label's dimensions, preserving unresolved constraints."""
    coerced = [coerce_dimension(dim) for dim in dimensions]
    unique = list(dict.fromkeys(coerced))

    non_unit = [dim for dim in unique if dim != 1]
    if not non_unit:
        return 1
    if len(non_unit) == 1:
        return non_unit[0]

    concrete = [dim for dim in non_unit if not isinstance(dim, SymbolicExpression)]
    if len(concrete) >= 2:
        broadcast_values([int(dim) for dim in concrete], label)

    return Broadcast(*non_unit, label=label)


class Product(SymbolicExpression):
    """A product of a constant coefficient and symbolic factors."""

    __slots__ = ("coefficient", "factors")

    def __new__(cls, coefficient: int = 1, factors: tuple[SymbolicExpression, ...] = ()) -> "Product":
        text = _product_display(coefficient, factors, unicode=True, compact=True)
        return str.__new__(cls, text)

    def __init__(self, coefficient: int = 1, factors: tuple[SymbolicExpression, ...] = ()):
        self.coefficient = int(coefficient)
        self.factors = tuple(factors)

    @property
    def coef(self) -> int:
        """Alias for the constant product coefficient."""
        return self.coefficient

    @property
    def coeff(self) -> int:
        """Alias for the constant product coefficient."""
        return self.coefficient

    @property
    def free_symbols(self) -> frozenset[str]:
        result: set[str] = set()
        for factor in self.factors:
            result.update(factor.free_symbols)
        return frozenset(result)

    def _evaluate(self, bindings: Mapping[str, int]) -> int:
        value = self.coefficient
        for factor in self.factors:
            value *= factor.bind(bindings)
        return int(value)

    def _substitute(self, values: Mapping[str, int]) -> Any:
        substituted = [
            factor._substitute(values) if isinstance(factor, SymbolicExpression) else factor
            for factor in self.factors
        ]
        return multiply(self.coefficient, *substituted)

    def display_text(self, *, unicode: bool = True, compact: bool = True) -> str:
        return _product_display(self.coefficient, self.factors, unicode=unicode, compact=compact)

    def _key(self) -> tuple[Any, ...]:
        return ("product", self.coefficient, tuple(factor._key() for factor in self.factors))


def _product_display(
    coefficient: int,
    factors: tuple[Any, ...],
    *,
    unicode: bool,
    compact: bool,
) -> str:
    multiplication = "×" if unicode else "*"
    parts: list[str] = []
    if coefficient != 1 or not factors:
        parts.append(str(coefficient))
    parts.extend(_display(factor, unicode=unicode, compact=compact) for factor in factors)
    return multiplication.join(parts)


# Domain-relevant alias.
SymbolicSize = Product


def multiply(*values: Any) -> Any:
    """Multiply values while folding constants and flattening products."""
    coefficient = 1
    factors: list[SymbolicExpression] = []

    def visit(value: Any) -> None:
        nonlocal coefficient
        if isinstance(value, Product):
            coefficient *= value.coefficient
            factors.extend(value.factors)
        elif isinstance(value, SymbolicExpression):
            factors.append(value)
        else:
            coefficient *= operator.index(value)

    for value in values:
        visit(value)

    if coefficient == 0:
        return 0
    if not factors:
        return coefficient
    if coefficient == 1 and len(factors) == 1:
        return factors[0]
    return Product(coefficient, tuple(factors))


class Sum(SymbolicExpression):
    """A sum of concrete and symbolic terms."""

    __slots__ = ("terms",)

    def __new__(cls, terms: tuple[Any, ...] = ()) -> "Sum":
        text = "+".join(_display(term, unicode=True, compact=True) for term in terms)
        return str.__new__(cls, text)

    def __init__(self, terms: tuple[Any, ...] = ()):
        self.terms = tuple(terms)

    @property
    def free_symbols(self) -> frozenset[str]:
        result: set[str] = set()
        for term in self.terms:
            if isinstance(term, SymbolicExpression):
                result.update(term.free_symbols)
        return frozenset(result)

    def _evaluate(self, bindings: Mapping[str, int]) -> float:
        total = 0
        for term in self.terms:
            total += term.bind(bindings) if isinstance(term, SymbolicExpression) else term
        return total

    def _substitute(self, values: Mapping[str, int]) -> Any:
        substituted = [
            term._substitute(values) if isinstance(term, SymbolicExpression) else term for term in self.terms
        ]
        return add(*substituted)

    def display_text(self, *, unicode: bool = True, compact: bool = True) -> str:
        return "+".join(_display(term, unicode=unicode, compact=compact) for term in self.terms)

    def _key(self) -> tuple[Any, ...]:
        return ("sum", tuple(_expr_key(term) for term in self.terms))


def add(*values: Any) -> Any:
    constant = 0
    terms: list[Any] = []

    def visit(value: Any) -> None:
        nonlocal constant
        if isinstance(value, Sum):
            for term in value.terms:
                visit(term)
        elif isinstance(value, SymbolicExpression):
            terms.append(value)
        else:
            constant += operator.index(value)

    for value in values:
        visit(value)

    if constant:
        terms.append(constant)
    if not terms:
        return 0
    if len(terms) == 1 and not (isinstance(terms[0], int) and terms[0] == 0):
        return terms[0]
    return Sum(tuple(terms))


class NaryExpression(SymbolicExpression):
    """Shared behavior for max-like expressions."""

    __slots__ = ("name", "terms")

    def __init__(self, name: str, terms: tuple[Any, ...]):
        self.name = name
        self.terms = tuple(terms)

    @property
    def free_symbols(self) -> frozenset[str]:
        result: set[str] = set()
        for term in self.terms:
            if isinstance(term, SymbolicExpression):
                result.update(term.free_symbols)
        return frozenset(result)

    def _evaluated_values(self, bindings: Mapping[str, int]) -> list[float]:
        return [term.bind(bindings) if isinstance(term, SymbolicExpression) else term for term in self.terms]

    def _substitute(self, values: Mapping[str, int]) -> Any:
        substituted = [
            term._substitute(values) if isinstance(term, SymbolicExpression) else term for term in self.terms
        ]
        if self.name == "max":
            return symbolic_max(*substituted)
        new = NaryExpression(self.name, tuple(substituted))
        return new

    def display_text(self, *, unicode: bool = True, compact: bool = True) -> str:
        sep = "," if compact else ", "
        return f"{self.name}({sep.join(_display(term, unicode=unicode, compact=compact) for term in self.terms)})"

    def _key(self) -> tuple[Any, ...]:
        return (self.name, tuple(_expr_key(term) for term in self.terms))


class MaxExpr(NaryExpression):
    """A deferred max comparison."""

    def __new__(cls, *terms: Any) -> "MaxExpr":
        inner = ",".join(_display(term, unicode=True, compact=True) for term in terms)
        return str.__new__(cls, f"max({inner})")

    def __init__(self, *terms: Any):
        super().__init__("max", tuple(terms))

    def _evaluate(self, bindings: Mapping[str, int]) -> float:
        return max(self._evaluated_values(bindings))


def symbolic_max(*values: Any) -> Any:
    """Create a max expression, fully evaluating it when possible."""
    terms: list[Any] = []
    for value in values:
        if isinstance(value, MaxExpr):
            terms.extend(value.terms)
        else:
            terms.append(value)
    if not terms:
        return 1
    if len(terms) == 1:
        return terms[0]
    if all(not isinstance(term, SymbolicExpression) for term in terms):
        return max(operator.index(term) for term in terms)
    return MaxExpr(*terms)


class Ratio(SymbolicExpression):
    """A deferred ratio, used for symbolic speedup display."""

    __slots__ = ("numerator", "denominator")

    def __new__(cls, numerator: Any, denominator: Any) -> "Ratio":
        left = _display(numerator, unicode=True, compact=True)
        right = _display(denominator, unicode=True, compact=True)
        return str.__new__(cls, f"{left}/{right}")

    def __init__(self, numerator: Any, denominator: Any):
        self.numerator = numerator
        self.denominator = denominator

    @property
    def free_symbols(self) -> frozenset[str]:
        result: set[str] = set()
        for term in (self.numerator, self.denominator):
            if isinstance(term, SymbolicExpression):
                result.update(term.free_symbols)
        return frozenset(result)

    def _evaluate(self, bindings: Mapping[str, int]) -> float:
        numerator = self.numerator.bind(bindings) if isinstance(self.numerator, SymbolicExpression) else self.numerator
        denominator = (
            self.denominator.bind(bindings)
            if isinstance(self.denominator, SymbolicExpression)
            else self.denominator
        )
        return numerator / max(denominator, 1)

    def _substitute(self, values: Mapping[str, int]) -> Any:
        numerator = (
            self.numerator._substitute(values)
            if isinstance(self.numerator, SymbolicExpression)
            else self.numerator
        )
        denominator = (
            self.denominator._substitute(values)
            if isinstance(self.denominator, SymbolicExpression)
            else self.denominator
        )
        if not isinstance(numerator, SymbolicExpression) and not isinstance(denominator, SymbolicExpression):
            return numerator / max(denominator, 1)
        return Ratio(numerator, denominator)

    def display_text(self, *, unicode: bool = True, compact: bool = True) -> str:
        left = _display(self.numerator, unicode=unicode, compact=compact)
        right = _display(self.denominator, unicode=unicode, compact=compact)
        return f"{left}/{right}"

    def _key(self) -> tuple[Any, ...]:
        return ("ratio", _expr_key(self.numerator), _expr_key(self.denominator))


def bind_plan(plan: Any, bindings: Mapping[Any, Any] | Any = None, value: Any = None, **kwargs: Any) -> Any:
    """Bind a :class:`~opt_einsum.contract.PathInfo` or symbolic expression."""
    if hasattr(plan, "bind"):
        return plan.bind(bindings, value, **kwargs) if value is not None else plan.bind(bindings, **kwargs)
    if isinstance(plan, SymbolicExpression):
        return plan.bind(bindings, value, **kwargs) if value is not None else plan.bind(bindings, **kwargs)
    raise TypeError("bind_plan expects a PathInfo or SymbolicExpression object")
