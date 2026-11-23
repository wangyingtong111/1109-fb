# frozen_string_literal: false
class Matrix
  # Adapted from JAMA: http://math.nist.gov/javanumerics/jama/
  # and LAPACK (xGEQRF / xGEQP3 / xLARFG / xORMQR).

  #
  # For an m-by-n matrix +A+ (m >= n), the QR decomposition with column
  # pivoting produces an orthogonal m-by-m matrix +Q+, an upper triangular
  # m-by-n matrix +R+ and a permutation matrix +P+ so that
  #
  #   A*P = Q*R
  #
  # Column pivoting ensures that the diagonal entries of +R+ are ordered by
  # non-increasing magnitude:
  #
  #   |r_00| >= |r_11| >= ... >= |r_(k-1,k-1)|
  #
  # which makes the decomposition numerically rank revealing.  This allows a
  # least squares solution of an overdetermined (rectangular, possibly rank
  # deficient) system +A*x ~= b+ to be computed with an explicit tolerance,
  # while reusing a single decomposition for any number of right hand sides.
  #
  # Like JAMA, this code does not support Complex matrix inputs.
  #
  class QRDecomposition

    include Matrix::ConversionHelper

    # Constructs the QR decomposition with column pivoting for the matrix +a+.
    #
    def initialize(a)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      unless a.real?
        raise TypeError, "QR decomposition is not supported for Complex matrices"
      end

      @row_count    = a.row_count
      @column_count = a.column_count
      # Working copy of the matrix.  It stores:
      # * the strict upper triangle (including the diagonal) of +R+
      # * the Householder vectors below the diagonal
      @qr     = a.to_a.map(&:dup)
      @pivots = Array.new(@column_count) {|j| j}
      @rdiag  = Array.new(@column_count, 0.0) # diagonal entries of R

      householder_qr
    end

    # Returns the orthogonal factor +Q+, an m-by-m orthogonal matrix.
    #
    def q
      # Q = H_0 * H_1 * ... * H_(n-1); build it by applying each reflector
      # to the identity matrix from the left, in reverse order.
      q = Array.new(@row_count) {|i| Array.new(@row_count) {|j| i == j ? 1.0 : 0.0}}
      (@column_count - 1).downto(0) do |k|
        v = Array.new(@row_count, 0.0)
        v[k] = 1.0
        (k + 1).upto(@row_count - 1) {|i| v[i] = @qr[i][k]}
        apply_householder!(q, k, @tau[k], v)
      end
      Matrix.send(:new, q, @row_count)
    end
    alias_method :orthogonal_matrix, :q

    # Returns the upper triangular factor +R+, an m-by-n matrix.
    #
    def r
      rows = Array.new(@row_count) {|i|
        Array.new(@column_count) {|j|
          if i <= j && i < @column_count
            i == j ? @rdiag[j] : @qr[i][j]
          else
            0.0
          end
        }
      }
      Matrix.send(:new, rows, @column_count)
    end
    alias_method :upper_triangular, :r

    # Returns the column pivot indices.  Column +pivots[j]+ of the original
    # matrix was used as the j-th column of the factored matrix.
    #
    attr_reader :pivots

    # Returns the permutation matrix +P+ (so that +A*P = Q*R+).
    #
    def p
      rows = Array.new(@column_count) {Array.new(@column_count, 0)}
      @pivots.each_with_index {|col, j| rows[col][j] = 1}
      Matrix.send(:new, rows, @column_count)
    end
    alias_method :permutation, :p

    # Returns [+q+, +r+, +p+].
    #
    def to_ary
      [q, r, p]
    end
    alias_method :to_a, :to_ary

    #
    # Returns the numerical rank of the factored matrix, using +tolerance+ to
    # decide which diagonal entries of +R+ are effectively zero.
    #
    # A column is counted as independent while its pivot (the corresponding
    # diagonal entry of +R+) has a magnitude strictly greater than
    # +tolerance+.  When +tolerance+ is not given (or +nil+), it defaults to
    #
    #   column_count * eps * |r_00|
    #
    # where +eps+ is the machine epsilon for Float.  An explicitly given
    # tolerance makes nearly collinear columns distinguishable as intended:
    # small but resolvable pivots count toward the rank, while pivots below
    # the tolerance are treated as rank deficiency.
    #
    def rank(tolerance = nil)
      tol = tolerance || default_tolerance
      count = 0
      @rdiag.each do |diag|
        break if diag.abs <= tol
        count += 1
      end
      count
    end

    # Returns +true+ if the factored matrix is rank deficient according to
    # +tolerance+ (see #rank).
    #
    def rank_deficient?(tolerance = nil)
      rank(tolerance) < @column_count
    end
    alias_method :singular?, :rank_deficient?

    #
    # Returns a least squares solution +x+ of
    #
    #   A*x ~= b
    #
    # i.e. the vector (or matrix, for several right hand sides) minimizing
    # the squared residual.  +b+ can be a Vector, an Array or a Matrix; the
    # shape of the result mirrors the shape of +b+ (a Vector/Array yields a
    # Vector, a Matrix yields a Matrix with the same number of columns).
    #
    # The decomposition is computed once and may be reused for any number of
    # right hand sides:
    #
    #   qr = a.qr
    #   x1 = qr.solve(b1)
    #   x2 = qr.solve(b2)
    #
    # +tolerance+ controls the numerical rank determination as in #rank.
    # Rank deficient and underdetermined inputs do not raise: a basic minimum
    # residual solution is returned, with the unresolved unknowns reported as
    # zero.  Raises Matrix::ErrDimensionMismatch when the number of rows of
    # +b+ does not match the number of rows of +A+.
    #
    def solve(b, tolerance = nil)
      if b.is_a?(Matrix)
        if b.row_count != @row_count
          raise Matrix::ErrDimensionMismatch,
            "expected right hand side with #{@row_count} rows, got #{b.row_count}"
        end
        rhs = b.to_a.map(&:dup)
        column_count_b = b.column_count
        matrix_result = true
      else
        rhs_array = convert_to_array(b)
        if rhs_array.size != @row_count
          raise Matrix::ErrDimensionMismatch,
            "expected right hand side with #{@row_count} entries, got #{rhs_array.size}"
        end
        rhs = rhs_array.map {|e| [e.to_f]}
        column_count_b = 1
        matrix_result = false
      end
      rhs.each {|row| row.map!(&:to_f)}

      # Column pivoting only reorders the unknowns, so the right hand side
      # needs no row permutation.  Apply Q^T to the right hand side(s).
      apply_q_transpose!(rhs)

      k = rank(tolerance)

      # Solve R11 * y = (Q^T b) for the first k rows, by back substitution.
      # R is stored with its diagonal in @rdiag and strict upper part in @qr.
      y = Array.new(k) {Array.new(column_count_b, 0.0)}
      (k - 1).downto(0) do |i|
        column_count_b.times do |j|
          s = rhs[i][j]
          (i + 1).upto(k - 1) do |col|
            s -= @qr[i][col] * y[col][j]
          end
          y[i][j] = s / @rdiag[i]
        end
      end

      # Undo the column permutation; unknowns beyond the rank stay zero.
      x = Array.new(@column_count) {Array.new(column_count_b, 0.0)}
      k.times do |i|
        column_count_b.times {|j| x[@pivots[i]][j] = y[i][j]}
      end

      if matrix_result
        Matrix.send(:new, x, column_count_b)
      else
        Vector.elements(x.map {|row| row[0]}, false)
      end
    end
    alias_method :least_squares, :solve
    alias_method :lstsq, :solve

    private

    # Householder QR with column pivoting (LAPACK style).
    def householder_qr
      @tau = Array.new(@column_count, 0.0)
      column_norms = Array.new(@column_count) {|j| column_norm(j)}

      @column_count.times do |k|
        # Pivot: move the column with the largest remaining norm into place.
        pivot = (k ... @column_count).max_by {|j| column_norms[j]}
        if pivot != k
          swap_columns(k, pivot)
          column_norms[k], column_norms[pivot] = column_norms[pivot], column_norms[k]
        end

        # Build the Householder reflector H_k that zeros out column k below
        # the diagonal (LAPACK dlarfg convention).
        x_norm = 0.0
        (k ... @row_count).each {|i| x_norm = Math.hypot(x_norm, @qr[i][k].to_f)}

        if x_norm == 0.0
          @rdiag[k] = 0.0
          @tau[k]   = 0.0
          next
        end

        alpha = (@qr[k][k].to_f >= 0) ? -x_norm : x_norm
        entry_k = @qr[k][k].to_f
        # LAPACK dlarfg: H = I - tau v v^T, with v[k] == 1.
        tau = (alpha - entry_k) / alpha
        # Normalize the Householder vector so that its first component is 1.
        scale = entry_k - alpha # never zero when x_norm != 0
        (k + 1).upto(@row_count - 1) do |i|
          @qr[i][k] = (@qr[i][k].to_f / scale).to_f
        end
        @tau[k]   = tau
        @rdiag[k] = alpha

        # Apply H_k to the trailing submatrix: A <- (I - tau v v^T) A,
        # where v has 1 in position k and the stored entries below.
        (k + 1).upto(@column_count - 1) do |j|
          dot = @qr[k][j].to_f
          (k + 1).upto(@row_count - 1) {|i| dot += @qr[i][k] * @qr[i][j].to_f}
          @qr[k][j] = (@qr[k][j].to_f - tau * dot).to_f
          (k + 1).upto(@row_count - 1) do |i|
            @qr[i][j] = (@qr[i][j].to_f - tau * @qr[i][k] * dot).to_f
          end
        end

        # Update remaining column norms (the norm lost at this step).
        (k + 1).upto(@column_count - 1) do |j|
          if column_norms[j] != 0.0
            # Recompute from scratch for numerical reliability; cheap and robust.
            column_norms[j] = trailing_column_norm(j, k + 1)
          end
        end
      end
    end

    def column_norm(j)
      trailing_column_norm(j, 0)
    end

    def trailing_column_norm(j, first_row)
      norm = 0.0
      first_row.upto(@row_count - 1) {|i| norm = Math.hypot(norm, @qr[i][j].to_f)}
      norm
    end

    def swap_columns(a, b)
      @row_count.times do |i|
        @qr[i][a], @qr[i][b] = @qr[i][b], @qr[i][a]
      end
      @pivots[a], @pivots[b] = @pivots[b], @pivots[a]
    end

    # Apply Q^T in place to an array of column-vectors stored row-major:
    # rhs has @row_count rows and an arbitrary number of columns.
    def apply_q_transpose!(rhs)
      @column_count.times do |k|
        next if @tau[k] == 0.0
        apply_householder!(rhs, k, @tau[k], householder_vector(k))
      end
    end

    def householder_vector(k)
      v = Array.new(@row_count, 0.0)
      v[k] = 1.0
      (k + 1).upto(@row_count - 1) {|i| v[i] = @qr[i][k]}
      v
    end

    # In place application of (I - tau v v^T) to every column of +matrix+,
    # where v[0...k] are implicitly zero.
    def apply_householder!(matrix, k, tau, v)
      return if tau == 0.0
      column_count_b = matrix[0].size
      return if column_count_b.zero?
      column_count_b.times do |j|
        dot = 0.0
        k.upto(@row_count - 1) {|i| dot += v[i] * matrix[i][j]}
        k.upto(@row_count - 1) {|i| matrix[i][j] -= tau * v[i] * dot}
      end
    end

    def default_tolerance
      @column_count * Float::EPSILON * (@rdiag.first || 0.0).abs
    end
  end
end
