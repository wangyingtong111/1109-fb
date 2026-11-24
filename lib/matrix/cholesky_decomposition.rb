# frozen_string_literal: false
class Matrix
  #
  # Pivoted (and optionally incomplete) Cholesky factorization of a Hermitian
  # positive semidefinite matrix.
  #
  # For an +n+ by +n+ Hermitian positive semidefinite matrix +A+ the
  # factorization is built column by column, always picking the unused axis
  # with the largest remaining residual diagonal ("column pivoting").  A zero
  # on the diagonal therefore does not hide a later positive pivot:
  #
  #   m = Matrix.diagonal(0, 4, 1)
  #   c = m.cholesky
  #   c.rank                            # => 2
  #   c.pivots                          # => [1, 2]
  #   c.order                           # => [1, 2, 0]
  #   c.factor * c.factor.adjoint == m  # => true
  #
  # Let +P+ be the permutation matrix of #order (the retained pivots first,
  # followed by the unused axes), +L+ the +n+ by +rank+ lower triangular
  # factor and +U = L.adjoint+.  Then
  #
  #   P * A * P.adjoint == L * U
  #   factor = P.adjoint * L
  #   factor * factor.adjoint == A
  #
  # The last identity uses the adjoint, so it also holds for complex Hermitian
  # matrices.  When the matrix is rank deficient, +L * U+ has a trailing zero
  # block and #factor is the corresponding low-rank square root.
  #
  # +P+ is built from #order, which is #pivots (the retained axes in the order
  # they were picked) followed by the unused axes.  #pivots itself has one
  # entry per retained factor column, so its size equals #rank.
  #
  # == Relative tolerance and early stopping
  #
  # An optional relative tolerance may be given; the factorization stops as
  # soon as the largest remaining residual is at most
  # <tt>rel_tolerance * scale</tt>, where +scale+ is the largest absolute
  # diagonal entry of the original matrix.  A residual that is merely
  # numerically close to zero is truncated (#truncated?), while a clearly
  # negative direction raises +Matrix::ErrNotPositiveSemidefinite+.
  #
  # The stopped factorization keeps its state, so the program can continue it
  # later with #resume or redo it from scratch with #recompute:
  #
  #   c = m.cholesky(1e-12)
  #   c.truncated?          # => true when the factorization stopped early
  #   c.tolerance_limited?  # => true when a relative tolerance stopped it
  #   c.negligible?         # => true when the leftover residual is (near) zero
  #   c.residual            # largest remaining residual at the stopping point
  #   d = c.resume(1e-20)   # continue with a smaller tolerance, +c+ unchanged
  #   e = c.recompute       # or factorize again from scratch
  #
  # Without an explicit tolerance a run keeps going until the largest residual
  # is indistinguishable from rounding noise (but it does not divide by a
  # value that is only noise).  Exact types (Integer, Rational and Complex of
  # those) are factorized exactly: perfect squares give exact roots and exact
  # zero residuals are recognized without any tolerance.
  #
  class CholeskyDecomposition

    include Matrix::ConversionHelper

    # The retained pivots, in the order the columns were picked.  Its size is
    # the effective #rank of the matrix.
    attr_reader :pivots

    # The original matrix.
    attr_reader :matrix

    # The relative tolerance requested, or +nil+ when none was given.
    attr_reader :rel_tolerance

    # The absolute tolerance, <tt>rel_tolerance * scale</tt>, or +0+.
    attr_reader :tolerance

    # Largest absolute diagonal entry of the original matrix, as a Float.
    attr_reader :scale

    # Creates the pivoted Cholesky factorization of +a+.
    #
    # +rel_tolerance+ is an optional relative stopping tolerance; +nil+ (the
    # default) only stops when the residual is indistinguishable from zero.
    # When +check_hermitian+ is true the matrix is verified to be Hermitian
    # first and +Matrix::ErrNotHermitian+ is raised otherwise.  With the check
    # disabled the matrix is assumed Hermitian and read from its lower
    # triangle, like LAPACK's +uplo = 'L'+.
    def initialize(a, rel_tolerance = nil, check_hermitian: true)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch unless a.square?
      unless rel_tolerance.nil? || (rel_tolerance.is_a?(Numeric) && rel_tolerance >= 0)
        raise ArgumentError, "relative tolerance must be a non-negative number"
      end
      if rel_tolerance.is_a?(Float) && !rel_tolerance.finite?
        raise ArgumentError, "relative tolerance must be finite"
      end
      @matrix = a
      @n = a.row_count
      @assume_hermitian = !check_hermitian
      @exact = self.class.send(:exact_entries?, a)
      @scale = self.class.send(:scale_of, a)
      hermitian_tolerance = @exact ? 0 : @scale * Float::EPSILON * [@n, 1].max
      if check_hermitian &&
         !self.class.send(:hermitian_within?, a, hermitian_tolerance)
        raise Matrix::ErrNotHermitian
      end
      @residual_diagonal = Array.new(@n) { |i| source_entry(i, i).real }
      @active = (0...@n).to_a
      @loads = Array.new(@n) { [] }
      @pivots = []
      @pivot_diagonal = []
      @truncated = false
      @residual = nil
      @noisy = false
      factorize(rel_tolerance)
    end

    # Returns +true+ when every axis was used, i.e. the matrix had full rank.
    def complete?
      @pivots.size == @n
    end
    alias_method :full_rank?, :complete?

    # Returns +true+ when the factorization stopped before using every axis.
    def truncated?
      @truncated
    end
    alias_method :stopped?, :truncated?

    # Returns +true+ when the stopped factorization lost nothing, i.e. the
    # largest leftover residual is (numerically) exactly zero.  This is how a
    # genuinely rank deficient matrix differs from one that was cut short by a
    # relative tolerance (#tolerance_limited?).
    def negligible?
      return false unless truncated?
      return true if @scale.zero?
      @residual.abs <= noise_floor
    end
    alias_method :exact_rank?, :negligible?

    # Returns +true+ when the factorization stopped because the remaining
    # residual was already below the requested relative tolerance, but not
    # negligible: the rank is a numerical decision, and #resume with a smaller
    # tolerance may retain more axes.
    def tolerance_limited?
      truncated? && !negligible?
    end

    # Returns +true+ when the matrix is singular, i.e. its effective #rank is
    # smaller than its dimension.
    def singular?
      @pivots.size < @n
    end

    # A pivoted Cholesky factorization is only constructed for positive
    # semidefinite matrices; a clearly negative direction raises
    # +Matrix::ErrNotPositiveSemidefinite+ instead.  This always returns +true+
    # and is provided for symmetry with the other decompositions.
    def positive_semidefinite?
      true
    end

    # Returns the effective rank, i.e. the number of retained pivots.
    def rank
      @pivots.size
    end

    # Returns the largest remaining residual at the stopping point, or +nil+
    # when every axis was used.
    def residual
      @pivots.size == @n ? nil : @residual
    end

    # Returns the largest remaining residual relative to #scale, or +nil+ when
    # every axis was used.  A truncated factorization satisfies
    # <tt>relative_residual <= rel_tolerance</tt> unless rounding noise made it
    # stop earlier.
    def relative_residual
      value = residual
      return nil if value.nil?
      return 0.0 if @scale.zero?
      (value.abs / @scale).to_f
    end

    # Returns the residual diagonal in axis order.  Used axes are zero, unused
    # axes keep their remaining residual.
    def residuals
      @residual_diagonal.dup
    end

    # Returns the axes that were not used as pivots, in their original order.
    def remaining
      (0...@n).to_a - @pivots
    end

    # Returns the full permutation: retained pivots first, then the remaining
    # axes.  The permutation matrix #p follows this order, so that
    # <tt>(p * x)[i] == x[order[i]]</tt>.
    def order
      @pivots + remaining
    end
    alias_method :permutation, :order

    # Returns the permutation matrix +P+ so that
    # <tt>p * a * p.adjoint == l * u</tt>.
    def p
      rows = Array.new(@n) { Array.new(@n, 0) }
      order.each_with_index { |axis, i| rows[i][axis] = 1 }
      Matrix.send :new, rows, @n
    end

    # Returns the +n+ by +rank+ lower triangular factor +L+ in permuted order,
    # so that <tt>p * a * p.adjoint == l * u</tt>.
    def l
      loads = factor.to_a
      rows = order.map { |axis| loads[axis] }
      Matrix.send :new, rows, @pivots.size
    end

    # Returns the +rank+ by +n+ upper triangular factor +U+, the adjoint of #l.
    def u
      l.adjoint
    end

    # Returns the +n+ by +rank+ factor +F+ in the original axis order, so that
    # <tt>factor * factor.adjoint == a</tt> when the matrix is positive
    # semidefinite and the factorization ran to completion.
    def factor
      k = @pivots.size
      rows = Array.new(@n) { |i| padded_row(i, k) }
      Matrix.send :new, rows, k
    end
    alias_method :low_rank_factor, :factor

    # Returns +L+, +U+, +P+ in an array.
    def to_ary
      [l, u, p]
    end
    alias_method :to_a, :to_ary

    # Returns the determinant, from the product of the pivot residuals.  A
    # singular matrix has determinant 0.
    def det
      return 0 unless complete?
      @pivot_diagonal.inject(1) { |product, value| product * value }
    end
    alias_method :determinant, :det

    # Returns +x+ so that <tt>a * x = b</tt>, using the factor.  Raises
    # +Matrix::ErrNotRegular+ if the matrix is singular.  +b+ can be a Matrix
    # or a Vector.
    def solve(b)
      raise Matrix::ErrNotRegular, "Matrix is singular." unless complete?
      n = @n
      permutation = order
      rows = l.to_a
      if b.is_a?(Matrix)
        raise Matrix::ErrDimensionMismatch if b.row_count != n
        nx = b.column_count
        m = permutation.map { |axis| b.row(axis).to_a }
        n.times do |k|
          nx.times { |j| m[k][j] = m[k][j].quo(rows[k][k]) }
          (k+1).upto(n-1) do |i|
            coefficient = rows[i][k]
            nx.times { |j| m[i][j] -= coefficient * m[k][j] }
          end
        end
        (n-1).downto(0) do |k|
          conjugate = rows[k][k].conjugate
          nx.times { |j| m[k][j] = m[k][j].quo(conjugate) }
          k.times do |i|
            coefficient = rows[k][i].conjugate
            nx.times { |j| m[i][j] -= coefficient * m[k][j] }
          end
        end
        solution = Array.new(n)
        permutation.each_with_index { |axis, q| solution[axis] = m[q] }
        Matrix.send :new, solution, nx
      else
        b = convert_to_array(b)
        raise Matrix::ErrDimensionMismatch if b.size != n
        m = b.values_at(*permutation)
        n.times do |k|
          m[k] = m[k].quo(rows[k][k])
          (k+1).upto(n-1) do |i|
            m[i] -= rows[i][k] * m[k]
          end
        end
        (n-1).downto(0) do |k|
          m[k] = m[k].quo(rows[k][k].conjugate)
          k.times do |i|
            m[i] -= rows[k][i].conjugate * m[k]
          end
        end
        solution = Array.new(n)
        permutation.each_with_index { |axis, q| solution[axis] = m[q] }
        Vector.elements(solution, false)
      end
    end

    # Returns a new decomposition continuing from where this one stopped.
    #
    # +rel_tolerance+ is the new relative tolerance; the default +0+ only
    # stops at an exactly zero residual.  The receiver is not modified.
    def resume(rel_tolerance = 0)
      unless rel_tolerance.is_a?(Numeric) && rel_tolerance >= 0
        raise ArgumentError, "relative tolerance must be a non-negative number"
      end
      if rel_tolerance.is_a?(Float) && !rel_tolerance.finite?
        raise ArgumentError, "relative tolerance must be finite"
      end
      self.class.send(:continue_from, self, rel_tolerance)
    end

    # Returns a new decomposition of the same matrix, factorized from scratch
    # with the given relative tolerance.
    def recompute(rel_tolerance = nil)
      self.class.new(@matrix, rel_tolerance, check_hermitian: !@assume_hermitian)
    end

    private

    # Entry +i,j+ of the Hermitian matrix being factorized.  When the caller
    # disabled the Hermitian check only the lower triangle is read.
    def source_entry(i, j)
      return @matrix[i, j] unless @assume_hermitian
      i >= j ? @matrix[i, j] : @matrix[j, i].conjugate
    end

    # Row +i+ of +F+, zero padded to +column_count+ entries.  An axis stops
    # accumulating entries once it has been used as a pivot, so the stored row
    # can be shorter.
    def padded_row(i, column_count)
      stored = @loads[i]
      return stored.dup if stored.size >= column_count
      stored + Array.new(column_count - stored.size, 0)
    end

    def factorize(rel_tolerance)
      @rel_tolerance = rel_tolerance
      @tolerance = rel_tolerance.nil? ? 0 : rel_tolerance * @scale
      loop do
        if @active.empty?
          @residual = nil
          break
        end
        pivot = @active.max_by { |i| @residual_diagonal[i] }
        diagonal = @residual_diagonal[pivot]
        if diagonal < -noise_floor
          raise Matrix::ErrNotPositiveSemidefinite,
                "negative pivot #{diagonal} on axis #{pivot}"
        elsif diagonal <= stopping_threshold
          verify_positive_semidefinite!
          @truncated = true
          @residual = @active.map { |i| @residual_diagonal[i] }.max_by(&:abs)
          break
        end
        column = @pivots.size
        root = self.class.send(:square_root_of, diagonal)
        @noisy ||= self.class.send(:inexact?, root)
        @active.each do |i|
          sum = source_entry(i, pivot)
          column.times { |j| sum -= @loads[i][j] * @loads[pivot][j].conjugate }
          load = i == pivot ? root : sum.quo(root)
          @noisy ||= self.class.send(:inexact?, load)
          @loads[i] << load
        end
        others = @active - [pivot]
        others.each do |i|
          value = @loads[i][column]
          @noisy ||= self.class.send(:inexact?, value)
          @residual_diagonal[i] = @residual_diagonal[i] - (value * value.conjugate).real
        end
        @residual_diagonal[pivot] = 0
        @pivots << pivot
        @pivot_diagonal << diagonal
        @active = others
      end
      self
    end

    # Residuals below this are indistinguishable from rounding noise; they are
    # truncated rather than reported as negative directions.  Exact arithmetic
    # has no noise, so anything strictly negative is a definite negative
    # direction.  The threshold does not depend on the requested relative
    # tolerance: a clearly negative direction is always reported.
    def noise_floor
      return 0 if @exact && !@noisy
      @scale * Float::EPSILON * [@n, 1].max
    end

    # The largest residual that still counts as zero: the rounding noise floor
    # plus the requested relative tolerance.  A run without an explicit
    # tolerance therefore stops at a numerically zero residual instead of
    # dividing by it.
    def stopping_threshold
      [@tolerance.to_f, noise_floor].max
    end

    # Necessary conditions for the residual block to be positive semidefinite:
    # every remaining diagonal is non-negative and
    # <tt>|S[i,j]|**2 <= S[i,i]*S[j,j]</tt> for every pair.  A violation proves
    # a negative direction even when the diagonals vanished, e.g.
    # <tt>Matrix[[0, i], [-i, 0]]</tt>.
    #
    # The threshold is the rounding noise floor, not the requested relative
    # tolerance: a clearly negative direction is reported even when the caller
    # asked to stop early.
    def verify_positive_semidefinite!
      unused = @active
      threshold = noise_floor
      unused.each do |i|
        if @residual_diagonal[i] < -threshold
          raise Matrix::ErrNotPositiveSemidefinite,
                "negative pivot #{@residual_diagonal[i]} on axis #{i}"
        end
      end
      return if unused.size < 2
      column = @pivots.size
      slack = if @exact && !@noisy
                0
              else
                @scale * @scale * Float::EPSILON * [@n, 1].max
              end
      unused.each_with_index do |i, index|
        (index+1).upto(unused.size-1) do |other|
          j = unused[other]
          value = source_entry(i, j)
          column.times { |t| value -= @loads[i][t] * @loads[j][t].conjugate }
          magnitude = value.abs
          bound = @residual_diagonal[i].abs * @residual_diagonal[j].abs
          if magnitude * magnitude > bound + slack
            raise Matrix::ErrNotPositiveSemidefinite,
                  "negative direction between axes #{i} and #{j}"
          end
        end
      end
    end

    class << self
      private

      def continue_from(source, rel_tolerance)
        copy = allocate
        %i[@matrix @n @exact @scale @noisy @assume_hermitian].each do |name|
          copy.instance_variable_set(name, source.instance_variable_get(name))
        end
        %i[@residual_diagonal @active @pivots @pivot_diagonal].each do |name|
          copy.instance_variable_set(name, source.instance_variable_get(name).dup)
        end
        copy.instance_variable_set(:@loads, source.instance_variable_get(:@loads).map(&:dup))
        copy.instance_variable_set(:@truncated, false)
        copy.instance_variable_set(:@residual, nil)
        copy.send(:factorize, rel_tolerance)
        copy
      end

      def exact_entries?(a)
        a.each do |e|
          return false if e.is_a?(Float)
          if e.is_a?(Complex)
            return false if e.real.is_a?(Float) || e.imaginary.is_a?(Float)
          end
        end
        true
      end

      def inexact?(value)
        return true if value.is_a?(Float)
        if value.is_a?(Complex)
          return true if value.real.is_a?(Float) || value.imaginary.is_a?(Float)
        end
        false
      end

      # Largest absolute diagonal entry.  The diagonal of a Hermitian matrix is
      # real, so reading it directly is safe regardless of the triangle used.
      def scale_of(a)
        size = a.row_count
        return 0.0 if size.zero?
        Array.new(size) { |i| a[i, i].real.abs.to_f }.max
      end

      def hermitian_within?(a, tolerance)
        size = a.row_count
        size.times do |i|
          size.times do |j|
            difference = a[i, j] - a[j, i].conjugate
            return false if difference.abs > tolerance
          end
        end
        true
      end

      # Square root that stays exact for perfect squares of exact types.
      def square_root_of(value)
        case value
        when Integer
          raise Matrix::ErrNotPositiveSemidefinite, "negative pivot #{value}" if value < 0
          root = Integer.sqrt(value)
          root * root == value ? root : Math.sqrt(value)
        when Rational
          raise Matrix::ErrNotPositiveSemidefinite, "negative pivot #{value}" if value < 0
          numerator = square_root_of(value.numerator)
          denominator = square_root_of(value.denominator)
          if numerator.is_a?(Integer) && denominator.is_a?(Integer) && !denominator.zero?
            Rational(numerator, denominator)
          else
            Math.sqrt(value)
          end
        else
          Math.sqrt(value)
        end
      end
    end
  end
end

