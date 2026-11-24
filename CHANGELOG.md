# Changelog

List of new feature changes (excluding most bug fixes and optimizations)

## Unreleased

* Add `Matrix#solution_space` (and `Matrix::SolutionSpace`) for the exact
  solution space of `A*x = b` with rectangular, rank deficient coefficient
  matrices and several right hand side columns at once.  Each column is
  reported solvable or not, a particular solution is returned for solvable
  columns, and one null space basis is shared by all of them.

## v0.4.0

* Add `Matrix#rotate_entries` [#19]

## v0.3.1 / Ruby 3.0

* Frozen `Matrix` are Ractor-shareable.

## v0.3.0

* Add `Matrix#adjoint` [#14]

## v0.2.0 / Ruby 2.7

* Add Matrix#abs [ruby/ruby#2199]

## v0.1.0 / Ruby 2.6

* Add `Matrix#antisymmetric?` / `#skew_symmetric?`
* Add `Matrix#map!` / `#collect!`
* Add `Matrix#[]=`
* Add `Vector#map!` / `#collect!`
* Add `Vector#[]=`

## Ruby 2.5

* Add `Matrix.combine` and `Matrix#combine`
* `Matrix#hadamard_product` and `Matrix#entrywise_product`
