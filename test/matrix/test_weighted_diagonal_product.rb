# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestWeightedDiagonalProduct < Test::Unit::TestCase
  def setup
    @a = Matrix[[1, 2, 0], [0, 2, 1]]
    @b = Matrix[[3, 0], [2, 5], [0, 1]]
  end

  def test_diagonal_without_off_diagonal_entries
    assert_equal([7, 11], @a.diagonal_product(@b))
    assert_equal([7, 11], Matrix.diagonal_product(@a, @b))
  end

  def test_weighted_diagonal_product
    result = @a.weighted_diagonal_product(@b, [2, 0])

    assert_equal([[7, 11], 14], result)
    assert_equal([7, 11], result.diagonal)
    assert_equal(14, result.weighted_sum)
    assert_equal(14, Matrix.weighted_diagonal_product(@a, @b, [2, 0])[1])
    assert_equal(14, @a.weighted_diagonal_sum(@b, [2, 0]))
  end

  def test_rectangular_product_keeps_all_diagonal_values_and_exact_scalars
    a = Matrix[[Rational(1, 3), 0], [0, Rational(1, 5)], [1, 2]]
    b = Matrix[[3, 0], [0, 2]]

    diagonal, total = a.weighted_diagonal_product(b, [0, Rational(1, 2)])

    assert_equal([1, Rational(2, 5)], diagonal)
    assert_equal(Rational(1, 5), total)
  end

  def test_empty_axes_have_minimum_dimension_diagonal
    a = Matrix.empty(2, 0)
    b = Matrix.empty(0, 3)

    diagonal, total = a.weighted_diagonal_product(b, [1, 2])

    assert_equal([0, 0], diagonal)
    assert_equal(0, total)
  end

  def test_wrong_weight_count_raises_dimension_mismatch
    error = assert_raise(Matrix::ErrDimensionMismatch) do
      @a.weighted_diagonal_product(@b, [1])
    end
    assert_match(/expected 2.*got 1/, error.message)
  end

  def test_incompatible_factors_keep_multiplication_validation
    assert_raise(Matrix::ErrDimensionMismatch) do
      @b.diagonal_product(@b)
    end
  end

  def test_original_multiplication_and_trace_are_unchanged
    product = @a * @b

    assert_equal(Matrix[[7, 10], [4, 11]], product)
    assert_equal(18, product.trace)
  end
end
