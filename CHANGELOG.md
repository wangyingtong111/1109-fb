# Changelog

List of new feature changes (excluding most bug fixes and optimizations)

## Unreleased

* Add `Matrix#qr` / `Matrix::QRDecomposition`, a rank-revealing QR
  decomposition with column pivoting, and `Matrix#least_squares` (aliased
  `Matrix#lstsq`) for rectangular, possibly rank deficient least squares
  systems; a single decomposition can be reused for several right hand sides
  and the numerical rank can be controlled through an explicit tolerance.

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
