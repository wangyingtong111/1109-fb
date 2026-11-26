# frozen_string_literal: false
class Matrix
  # Adapted from JAMA: http://math.nist.gov/javanumerics/jama/

  #
  # For an m-by-n matrix A with m >= n, the LU decomposition is an m-by-n
  # unit lower triangular matrix L, an n-by-n upper triangular matrix U,
  # and a m-by-m permutation matrix P so that L*U = P*A.
  # If m < n, then L is m-by-m and U is m-by-n.
  #
  # The LUP decomposition with pivoting always exists, even if the matrix is
  # singular, so the constructor will never fail.  The primary use of the
  # LU decomposition is in the solution of square systems of simultaneous
  # linear equations.  This will fail if singular? returns true.
  #

  class LUPDecomposition
    # Returns the lower triangular factor +L+

    include Matrix::ConversionHelper

    def l
      Matrix.build(@row_count, [@column_count, @row_count].min) do |i, j|
        if (i > j)
          @lu[i][j]
        elsif (i == j)
          1
        else
          0
        end
      end
    end

    # Returns the upper triangular factor +U+

    def u
      Matrix.build([@column_count, @row_count].min, @column_count) do |i, j|
        if (i <= j)
          @lu[i][j]
        else
          0
        end
      end
    end

    # Returns the permutation matrix +P+

    def p
      rows = Array.new(@row_count){Array.new(@row_count, 0)}
      @pivots.each_with_index{|p, i| rows[i][p] = 1}
      Matrix.send :new, rows, @row_count
    end

    # Returns +L+, +U+, +P+ in an array

    def to_ary
      [l, u, p]
    end
    alias_method :to_a, :to_ary

    # Returns the pivoting indices

    attr_reader :pivots

    # Returns +true+ if +U+, and hence +A+, is singular.

    def singular?
      @column_count.times do |j|
        if (@lu[j][j] == 0)
          return true
        end
      end
      false
    end

    # Returns the determinant of +A+, calculated efficiently
    # from the factorization.

    def det
      if (@row_count != @column_count)
        raise Matrix::ErrDimensionMismatch
      end
      d = @pivot_sign
      @column_count.times do |j|
        d *= @lu[j][j]
      end
      d
    end
    alias_method :determinant, :det

    # Returns the pair <tt>[log(|det(A)|), phase]</tt> of +A+, calculated
    # from the factorization without ever forming the determinant itself.
    #
    # The logarithms of the pivots are added one by one, so a determinant
    # whose magnitude overflows or underflows a Float still gets a finite
    # logarithm.  The phase is +1+ or +-1+ when the determinant is real,
    # and a unit Complex number when it is complex; in both cases it
    # includes the sign of the row permutations performed by the
    # factorization.  The phase of a singular matrix is +0+, matching the
    # negative infinite logarithm of its zero determinant.
    #
    #   Matrix.diagonal(-1e200, 1e200).lup.slogdet
    #   #  => [921.0340371976183, -1]

    def slogdet
      if (@row_count != @column_count)
        raise Matrix::ErrDimensionMismatch
      end
      log_abs = 0.0
      phase = @pivot_sign
      @column_count.times do |j|
        pivot = @lu[j][j]
        return [-Float::INFINITY, 0] if pivot == 0
        pivot_log, pivot_phase = pivot_log_and_phase(pivot)
        log_abs += pivot_log
        phase *= pivot_phase
      end
      [log_abs, phase]
    end

    # Returns +m+ so that <tt>A*m = b</tt>,
    # or equivalently so that <tt>L*U*m = P*b</tt>
    # +b+ can be a Matrix or a Vector

    def solve b
      if (singular?)
        raise Matrix::ErrNotRegular, "Matrix is singular."
      end
      if b.is_a? Matrix
        if (b.row_count != @row_count)
          raise Matrix::ErrDimensionMismatch
        end

        # Copy right hand side with pivoting
        nx = b.column_count
        m = @pivots.map{|row| b.row(row).to_a}

        # Solve L*Y = P*b
        @column_count.times do |k|
          (k+1).upto(@column_count-1) do |i|
            nx.times do |j|
              m[i][j] -= m[k][j]*@lu[i][k]
            end
          end
        end
        # Solve U*m = Y
        (@column_count-1).downto(0) do |k|
          nx.times do |j|
            m[k][j] = m[k][j].quo(@lu[k][k])
          end
          k.times do |i|
            nx.times do |j|
              m[i][j] -= m[k][j]*@lu[i][k]
            end
          end
        end
        Matrix.send :new, m, nx
      else # same algorithm, specialized for simpler case of a vector
        b = convert_to_array(b)
        if (b.size != @row_count)
          raise Matrix::ErrDimensionMismatch
        end

        # Copy right hand side with pivoting
        m = b.values_at(*@pivots)

        # Solve L*Y = P*b
        @column_count.times do |k|
          (k+1).upto(@column_count-1) do |i|
            m[i] -= m[k]*@lu[i][k]
          end
        end
        # Solve U*m = Y
        (@column_count-1).downto(0) do |k|
          m[k] = m[k].quo(@lu[k][k])
          k.times do |i|
            m[i] -= m[k]*@lu[i][k]
          end
        end
        Vector.elements(m, false)
      end
    end

    def initialize a
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      # Use a "left-looking", dot-product, Crout/Doolittle algorithm.
      @lu = a.to_a
      @row_count = a.row_count
      @column_count = a.column_count
      @pivots = Array.new(@row_count)
      @row_count.times do |i|
         @pivots[i] = i
      end
      @pivot_sign = 1
      lu_col_j = Array.new(@row_count)

      # Outer loop.

      @column_count.times do |j|

        # Make a copy of the j-th column to localize references.

        @row_count.times do |i|
          lu_col_j[i] = @lu[i][j]
        end

        # Apply previous transformations.

        @row_count.times do |i|
          lu_row_i = @lu[i]

          # Most of the time is spent in the following dot product.

          kmax = [i, j].min
          s = 0
          kmax.times do |k|
            s += lu_row_i[k]*lu_col_j[k]
          end

          lu_row_i[j] = lu_col_j[i] -= s
        end

        # Find pivot and exchange if necessary.

        p = j
        (j+1).upto(@row_count-1) do |i|
          if (lu_col_j[i].abs > lu_col_j[p].abs)
            p = i
          end
        end
        if (p != j)
          @column_count.times do |k|
            t = @lu[p][k]; @lu[p][k] = @lu[j][k]; @lu[j][k] = t
          end
          k = @pivots[p]; @pivots[p] = @pivots[j]; @pivots[j] = k
          @pivot_sign = -@pivot_sign
        end

        # Compute multipliers.

        if (j < @row_count && @lu[j][j] != 0)
          (j+1).upto(@row_count-1) do |i|
            @lu[i][j] = @lu[i][j].quo(@lu[j][j])
          end
        end
      end
    end

    private

    # Returns <tt>[log(|value|), phase]</tt> for a non-zero pivot.
    # The magnitude is split off before any multiplication, so a pivot
    # whose modulus overflows a Float (as with a large Complex entry)
    # still yields a finite logarithm and an exact unit phase.

    def pivot_log_and_phase(value)
      if value.is_a?(Complex)
        real, imag = value.real, value.imag
        # A complex value with a zero imaginary part is still real-valued.
        return pivot_log_and_phase(real) if imag == 0
        scale = [real.abs, imag.abs].max
        return [0.0, 1] if scale == 0
        if scale.infinite?
          return [log_of(scale), Complex(real, imag).arg]
        end
        scaled_real, scaled_imag = real.quo(scale), imag.quo(scale)
        magnitude = Math.hypot(scaled_real.to_f, scaled_imag.to_f)
        phase = Complex((scaled_real.to_f / magnitude), (scaled_imag.to_f / magnitude))
        [log_of(scale) + Math.log(magnitude), phase]
      else
        [log_of(value.abs), value < 0 ? -1 : 1]
      end
    end

    # Returns the natural logarithm of a non-negative Numeric without
    # converting it through an overflowing Float first.

    def log_of(value)
      if defined?(BigDecimal) && value.is_a?(BigDecimal)
        _sign, digits, base, exponent = value.abs.split
        return Math.log(digits.to_i) + (exponent - digits.length) * Math.log(base)
      end
      case value
      when Integer
        Math.log(value)
      when Rational
        Math.log(value.numerator) - Math.log(value.denominator)
      else
        Math.log(value.to_f)
      end
    end
  end
end
