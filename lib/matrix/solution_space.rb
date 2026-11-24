# encoding: utf-8
# frozen_string_literal: false
class Matrix
  # Exact solution space of <tt>A*x = b</tt> for one or more right hand sides.
  #
  # Unlike Matrix::LUPDecomposition#solve, the coefficient matrix may be
  # rectangular and rank deficient, and a single call may carry several
  # target columns.  Every right hand side is treated independently: one
  # column may be consistent while another is not, even though the same
  # row operations are applied to all of them at once.
  #
  # All arithmetic stays in the entries' own type: divisions use
  # Integer#quo / Rational#quo, so with Integer or Rational coefficients
  # the particular solutions and the null space basis are exact.
  #
  #   a = Matrix[[1, 2, 3], [2, 4, 6]]
  #   b = Matrix.columns([[1, 2], [1, 3]])   # two target columns
  #   ss = a.solution_space(b)
  #   ss.rank                # => 1
  #   ss.consistent?(0)      # => true
  #   ss.particular(0)       # => Vector[(1/1), (0/1), (0/1)]
  #   a * ss.particular(0)   # => Vector[(1/1), (2/1)]  (== first column)
  #   ss.null_space.size     # => 2
  #   ss.consistent?(1)      # => false
  #   ss.particular(1)       # => nil
  #
  # The null space depends only on +A+, hence it is computed once and shared
  # by every consistent column.
  class SolutionSpace
    include ExceptionForMatrix

    # A single consistent right hand side: one particular solution plus the
    # null space shared by all columns.
    class Solution
      # The particular solution for this right hand side, as a Vector.
      attr_reader :particular
      # Array of Vector forming a basis of the null space of +A+.
      attr_reader :null_space
      # Columns of +A+ that are not pivots, i.e. the free variables of the
      # particular solution (all set to 0 there).
      attr_reader :free_column_indices

      def initialize(particular, null_space, free_column_indices)
        @particular = particular
        @null_space = null_space
        @free_column_indices = free_column_indices
      end

      # Dimension of the null space, i.e. the number of free variables.
      def nullity
        @null_space.size
      end

      # Returns +true+ if the right hand side was zero, in which case the
      # particular solution is the zero vector.
      def homogeneous?
        @particular.all?(&:zero?)
      end

      # Recomputes one member of the solution space:
      #   particular + c0*basis0 + c1*basis1 + ...
      #
      # The number of coefficients must equal the nullity.
      def at(*coefficients)
        unless coefficients.size == nullity
          raise ArgumentError, "expected #{nullity} coefficients, got #{coefficients.size}"
        end
        coordinates = @particular.to_a
        @null_space.each_with_index do |basis, i|
          c = coefficients[i]
          basis.each_with_index do |e, j|
            next if e.zero?
            coordinates[j] += c * e
          end
        end
        Vector.elements(coordinates)
      end

      # Returns +true+ if the particular solution and every basis vector
      # recompute exactly under +matrix+.
      def verify(matrix, rhs)
        (matrix * @particular) == rhs &&
          @null_space.all? { |basis| (matrix * basis).zero? }
      end

      def inspect
        "#<Matrix::SolutionSpace::Solution particular=#{@particular.inspect} nullity=#{nullity}>"
      end
    end

    include Enumerable

    # The coefficient matrix.
    attr_reader :matrix
    # Number of right hand side columns.
    attr_reader :column_count
    # Rank of the coefficient matrix.
    attr_reader :rank
    # Column indices of +A+ containing a pivot, in increasing order.
    attr_reader :pivot_column_indices
    # Column indices of +A+ that are free variables, in increasing order.
    attr_reader :free_column_indices
    # Array of Vector forming a basis of the null space of +A+.
    attr_reader :null_space
    # Array of Vector (respectively +nil+ for an inconsistent column),
    # one entry per right hand side column.
    attr_reader :particulars

    # +rhs+ may be:
    # * +nil+: no target column, useful to obtain the null space / degrees of
    #   freedom only,
    # * a Vector or an Array of row_count numbers: a single target column,
    # * a Matrix with +row_count+ rows: one target column per matrix column.
    def initialize(matrix, rhs = nil)
      raise TypeError, "Expected Matrix but got #{matrix.class}" unless matrix.is_a?(Matrix)
      @matrix = matrix
      @row_count = matrix.row_count
      @variable_count = matrix.column_count
      @rhs = normalize_rhs(rhs)
      @column_count = @rhs ? @rhs.column_count : 0
      reduce!
    end

    # Dimension of the null space, i.e. column_count - rank of +matrix+.
    # This is the number of degrees of freedom, and it depends only on the
    # coefficient matrix: it is well defined for an empty matrix too
    # (<tt>0 x n</tt> has nullity +n+, <tt>m x 0</tt> has nullity 0).
    def nullity
      @null_space.size
    end
    alias degrees_of_freedom nullity
    alias free_variable_count nullity

    # Returns +true+ if the right hand side column +index+ is solvable.
    def consistent?(index)
      @consistency.fetch(normalize_index(index))
    end
    alias solvable? consistent?

    # Returns +true+ if the right hand side column +index+ is not solvable.
    def inconsistent?(index)
      !consistent?(index)
    end

    # Returns the exact particular solution of column +index+ as a Vector,
    # or +nil+ if that column is inconsistent.
    def particular(index)
      @particulars.fetch(normalize_index(index))
    end
    alias particular_solution particular

    # Returns a Solution (particular solution plus shared null space) for
    # column +index+, or +nil+ if that column is inconsistent.
    def solution(index)
      return nil unless consistent?(index)
      @solutions[normalize_index(index)] ||= Solution.new(
        particular(index), @null_space, @free_column_indices
      )
    end

    # Returns the right hand side column +index+ as a Vector.
    def rhs_column(index)
      raise IndexError, "no right hand side was given" unless @rhs
      i = normalize_index(index)
      Vector.elements(Array.new(@row_count) { |row| @rhs[row, i] })
    end

    # Returns the indices of the solvable right hand side columns.
    def consistent_column_indices
      (0...@column_count).select { |i| @consistency[i] }
    end

    # Returns the indices of the unsolvable right hand side columns.
    def inconsistent_column_indices
      (0...@column_count).reject { |i| @consistency[i] }
    end

    # Returns +true+ if there is at least one right hand side column and all
    # of them are solvable.
    def all_consistent?
      !@consistency.empty? && @consistency.all?
    end

    # Returns +true+ if there is at least one right hand side column and none
    # of them is solvable.
    def none_consistent?
      !@consistency.empty? && @consistency.none?
    end

    # Returns a Matrix whose columns are the null space basis vectors,
    # i.e. a +variable_count+ x +nullity+ matrix.
    def null_space_matrix
      Matrix.build(@variable_count, nullity) { |i, j| @null_space[j][i] }
    end

    # Recomputes column +index+ exactly: A * particular == rhs and A * basis == 0.
    # Returns +false+ for an inconsistent column.
    def verify(index)
      sol = solution(index)
      return false unless sol
      sol.verify(@matrix, rhs_column(index))
    end

    # Iterates over one Solution per right hand side column (+nil+ for an
    # inconsistent column).
    def each
      return to_enum(:each) { @column_count } unless block_given?
      @column_count.times { |i| yield solution(i) }
      self
    end

    def to_h
      {
        particular: @particulars,
        null_space: @null_space,
        consistent: consistent_column_indices,
        inconsistent: inconsistent_column_indices,
      }
    end

    def inspect
      "#<Matrix::SolutionSpace rank=#{rank} nullity=#{nullity} " \
        "consistent=#{consistent_column_indices.inspect} " \
        "inconsistent=#{inconsistent_column_indices.inspect}>"
    end
    alias to_s inspect

    private

    def normalize_index(index)
      i = Integer(index)
      i += @column_count if i < 0
      unless 0 <= i && i < @column_count
        raise IndexError, "right hand side column #{index} is outside of 0...#{@column_count}"
      end
      i
    end

    def normalize_rhs(rhs)
      case rhs
      when nil
        nil
      when Vector
        raise ErrDimensionMismatch unless rhs.size == @row_count
        Matrix.column_vector(rhs)
      when Matrix
        raise ErrDimensionMismatch unless rhs.row_count == @row_count
        rhs
      when Array
        raise ErrDimensionMismatch unless rhs.size == @row_count
        Matrix.column_vector(rhs)
      else
        begin
          converted = rhs.to_matrix
        rescue StandardError
          raise TypeError, "can't convert #{rhs.class} into a Matrix or a Vector"
        end
        raise TypeError, "can't convert #{rhs.class} into a Matrix or a Vector" unless converted.is_a?(Matrix)
        normalize_rhs(converted)
      end
    end

    # Gauss-Jordan elimination on [A | b].  Every row operation is applied to
    # all right hand side columns at once.  Exact types stay exact thanks to
    # Numeric#quo.
    def reduce!
      a = @matrix.to_a.map(&:dup)
      b = @rhs && @rhs.to_a
      m = @row_count
      n = @variable_count
      k = @column_count

      @pivot_column_indices = []
      pivot_row = 0

      n.times do |col|
        break if pivot_row >= m
        candidates = (pivot_row...m).select { |i| a[i][col] != 0 }
        # An all-zero column is free: skip it, later columns may still pivot.
        next if candidates.empty?
        pivot = candidates.max_by { |i| a[i][col].abs }
        if pivot != pivot_row
          a[pivot_row], a[pivot] = a[pivot], a[pivot_row]
          b[pivot_row], b[pivot] = b[pivot], b[pivot_row] if b
        end

        akk = a[pivot_row][col]
        col.upto(n - 1) do |j|
          a[pivot_row][j] = a[pivot_row][j].quo(akk)
        end
        if b
          k.times { |j| b[pivot_row][j] = b[pivot_row][j].quo(akk) }
        end

        m.times do |i|
          next if i == pivot_row
          factor = a[i][col]
          next if factor == 0
          a[i][col] = 0
          (col + 1).upto(n - 1) do |j|
            e = a[pivot_row][j]
            next if e == 0
            a[i][j] = a[i][j] - factor * e
          end
          if b
            k.times do |j|
              e = b[pivot_row][j]
              next if e == 0
              b[i][j] = b[i][j] - factor * e
            end
          end
        end

        @pivot_column_indices << col
        pivot_row += 1
      end

      @rank = pivot_row
      @free_column_indices = (0...n).to_a - @pivot_column_indices

      # A right hand side is solvable iff every row below the rank has a
      # zero entry in that column of the reduced right hand side.
      @consistency = Array.new(k) do |j|
        (pivot_row...m).all? { |i| b[i][j] == 0 }
      end

      @particulars = Array.new(k) do |j|
        next nil unless @consistency[j]
        coordinates = Array.new(n, 0)
        @pivot_column_indices.each_with_index do |pc, r|
          coordinates[pc] = b[r][j]
        end
        Vector.elements(coordinates)
      end

      @solutions = Array.new(k)
      @null_space = @free_column_indices.map do |fc|
        coordinates = Array.new(n, 0)
        coordinates[fc] = 1
        @pivot_column_indices.each_with_index do |pc, r|
          coordinates[pc] = -a[r][fc]
        end
        Vector.elements(coordinates)
      end
    end
  end
end