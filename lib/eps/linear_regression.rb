module Eps
  class LinearRegression < BaseEstimator
    def coefficients
      @evaluator.coefficients
    end

    def r2
      @r2 ||= (sst - sse) / sst
    end

    def adjusted_r2
      @adjusted_r2 ||= (mst - mse) / mst
    end

    # prediction intervals for individual observations
    def predict_intervals(data, confidence: 0.95)
      _predict_intervals(data, confidence, :prediction)
    end

    # confidence intervals for the mean response
    def predict_mean_intervals(data, confidence: 0.95)
      _predict_intervals(data, confidence, :mean)
    end

    private

    # https://people.richland.edu/james/ictcm/2004/multiple.html
    def _summary(extended: false)
      coefficients = @coefficients
      str = String.new("")
      len = [coefficients.keys.map(&:size).max, 15].max
      if extended
        str += "%-#{len}s %12s %12s %12s %12s\n" % ["", "coef", "stderr", "t", "p"]
      else
        str += "%-#{len}s %12s %12s\n" % ["", "coef", "p"]
      end
      coefficients.each do |k, v|
        if extended
          str += "%-#{len}s %12.2f %12.2f %12.2f %12.3f\n" % [display_field(k), v, std_err[k], t_value[k], p_value[k]]
        else
          str += "%-#{len}s %12.2f %12.3f\n" % [display_field(k), v, p_value[k]]
        end
      end
      str += "\n"
      str += "r2: %.3f\n" % [r2] if extended
      str += "adjusted r2: %.3f\n" % [adjusted_r2]
      str
    end

    def _train(intercept: true, gsl: nil)
      raise "Target must be numeric" if @target_type != "numeric"
      check_missing_value(@train_set)
      check_missing_value(@validation_set) if @validation_set

      data = prep_x(@train_set)

      if data.size < data.columns.size + 2
        raise "Number of data points must be at least two more than number of features"
      end

      x = data.map_rows(&:to_a)

      gsl = defined?(GSLR) if gsl.nil?

      @intercept = intercept
      @gsl = gsl

      if intercept && !gsl
        data.size.times do |i|
          x[i].unshift(1)
        end
      end

      v3 =
        if gsl
          model = GSLR::OLS.new(intercept: intercept)
          model.fit(x, data.label, weight: data.weight)

          @covariance = model.covariance

          coefficients = model.coefficients.dup
          coefficients.unshift(model.intercept) if intercept
          coefficients
        else
          x = Matrix.rows(x)
          y = Matrix.column_vector(data.label)

          # weighted OLS
          # https://www.real-statistics.com/multiple-regression/weighted-linear-regression/weighted-regression-basics/
          w = Matrix.diagonal(*data.weight) if data.weight

          removed = []

          # https://statsmaths.github.io/stat612/lectures/lec13/lecture13.pdf
          # unfortunately, this method is unstable
          # haven't found an efficient way to do QR-factorization in Ruby
          # the extendmatrix gem has householder and givens (givens has bug)
          # but methods are too slow
          xt = x.t
          xt *= w if w
          begin
            @xtxi = (xt * x).inverse
          rescue ExceptionForMatrix::ErrNotRegular
            # matrix cannot be inverted
            # https://en.wikipedia.org/wiki/Multicollinearity

            constant = {}
            (1...x.column_count).each do |i|
              constant[i] = constant?(x.column(i))
            end

            # remove constant columns
            removed = constant.select { |_, v| v }.keys

            # remove non-independent columns
            constant.select { |_, v| !v }.keys.combination(2) do |c2|
              if !x.column(c2[0]).independent?(x.column(c2[1]))
                removed << c2[1]
              end
            end

            vectors = x.column_vectors
            # delete in reverse of indexes stay the same
            removed.sort.reverse.each do |i|
              # @coefficient_names.delete_at(i)
              vectors.delete_at(i)
            end
            x = Matrix.columns(vectors)
            xt = x.t
            xt *= w if w

            # try again
            begin
              @xtxi = (xt * x).inverse
            rescue ExceptionForMatrix::ErrNotRegular
              raise "Multiple solutions - GSL is needed to select one"
            end
          end
          # huge performance boost
          # by multiplying xt * y first
          # for weighted, w is already included in wt
          v2 = @xtxi * (xt * y)

          # convert to array
          v2 = v2.to_a.map { |xi| xi[0].to_f }

          # add back removed
          removed.sort.each do |i|
            v2.insert(i, 0)
          end
          @removed = removed

          v2
        end

      if @xtxi && @xtxi.each(:diagonal).any? { |v| v < 0 }
        raise UnstableSolution, "GSL is needed to find a stable solution for this dataset"
      end

      @coefficient_names = data.columns.keys
      @coefficient_names.unshift("_intercept") if intercept
      @coefficients = @coefficient_names.zip(v3).to_h
      Evaluators::LinearRegression.new(coefficients: @coefficients, features: @features, text_features: @text_features)
    end

    def prep_x(x)
      x = x.dup
      @features.each do |k, type|
        if type == "categorical"
          values = x.columns.delete(k)
          labels = values.uniq[1..-1]
          labels.each do |label|
            x.columns[[k, label]] = values.map { |v| v == label ? 1 : 0 }
          end
        end
      end
      prep_text_features(x)
      x
    end

    def constant?(arr)
      arr.all? { |x| x == arr[0] }
    end

    def _predict_intervals(data, confidence, type)
      raise "Intervals not available for loaded models" unless @trained
      raise ArgumentError, "confidence must be between 0 and 1" if confidence <= 0.0 || confidence >= 1.0
      raise "Not enough degrees of freedom for intervals" if degrees_of_freedom <= 0

      singular = data.is_a?(Hash)
      data = [data] if singular

      y_hat = predict(data)
      rows = design_rows(Eps::DataFrame.new(data))

      t = Eps::Statistics.students_t_ppf((1.0 + confidence) / 2.0, degrees_of_freedom)

      intervals = rows.zip(y_hat).map do |xi, yh|
        h = leverage(xi)
        # mean interval estimates the mean response
        # prediction interval adds the variance of a single observation
        v = mse * (type == :prediction ? 1.0 + h : h)
        half_width = t * Math.sqrt(v)
        [yh - half_width, yh + half_width]
      end

      singular ? intervals.first : intervals
    end

    # design matrix rows for new data
    # aligned with the coefficient names from training
    def design_rows(data)
      text_counts = {}
      @text_features.each do |k, v|
        encoder = TextEncoder.new(**v)
        text_counts[k] = encoder.transform(data.columns[k])
      end

      rows = Array.new(data.size) { [] }
      @coefficient_names.each do |name|
        col =
          if name == "_intercept"
            Array.new(data.size, 1.0)
          elsif name.is_a?(Array)
            k, label = name
            if @features[k] == "text"
              counts = text_counts[k]
              data.size.times.map { |i| counts[i][label] ? 1.0 : 0.0 }
            else
              data.columns[k].map { |v| v == label ? 1.0 : 0.0 }
            end
          else
            data.columns[name].map(&:to_f)
          end
        col.each_with_index { |v, i| rows[i] << v }
      end

      # drop columns removed during training (multicollinearity)
      if @removed && !@removed.empty?
        removed = @removed.sort.reverse
        rows.each do |row|
          removed.each { |i| row.delete_at(i) }
        end
      end

      rows
    end

    # leverage of a new observation
    # h = x' (X' W X)^-1 x
    def leverage(xi)
      m = xtxi_rows
      n = xi.size
      h = 0.0
      n.times do |i|
        s = 0.0
        mi = m[i]
        n.times { |j| s += mi[j] * xi[j] }
        h += xi[i] * s
      end
      h
    end

    def xtxi_rows
      @xtxi_rows ||= begin
        if @xtxi
          @xtxi.to_a
        else
          # GSL only exposes the covariance matrix
          @covariance.to_a.map { |row| row.map { |v| v / mse } }
        end
      end
    end

    # add epsilon for perfect fits
    # consistent with GSL
    def t_value
      @t_value ||= @coefficients.to_h { |k, v| [k, v / (std_err[k] + Float::EPSILON)] }
    end

    def p_value
      @p_value ||= begin
        @coefficients.to_h do |k, _|
          [k, 2 * Eps::Statistics.students_t_cdf(-t_value[k].abs, degrees_of_freedom)]
        end
      end
    end

    def std_err
      @std_err ||= begin
        @coefficient_names.zip(diagonal.map { |v| Math.sqrt(v) }).to_h
      end
    end

    def diagonal
      @diagonal ||= begin
        if covariance.is_a?(Array)
          covariance.size.times.map do |i|
            covariance[i][i]
          end
        elsif covariance.respond_to?(:each)
          d = covariance.each(:diagonal).to_a
          @removed.each do |i|
            d.insert(i, 0)
          end
          d
        else
          covariance.diagonal.to_a
        end
      end
    end

    def covariance
      @covariance ||= mse * @xtxi
    end

    def y_bar
      @y_bar ||= begin
        if (w = @train_set.weight)
          @train_set.label.zip(w).sum { |y, wi| y * wi } / w.sum.to_f
        else
          mean(@train_set.label)
        end
      end
    end

    def y_hat
      @y_hat ||= predict(@train_set)
    end

    # total sum of squares
    def sst
      @sst ||= begin
        if (w = @train_set.weight)
          @train_set.label.zip(w).sum { |y, wi| wi * (y - y_bar)**2 }
        else
          @train_set.label.sum { |y| (y - y_bar)**2 }
        end
      end
    end

    # sum of squared errors of prediction
    # not to be confused with "explained sum of squares"
    def sse
      @sse ||= begin
        if (w = @train_set.weight)
          @train_set.label.zip(y_hat, w).sum { |y, yh, wi| wi * (y - yh)**2 }
        else
          @train_set.label.zip(y_hat).sum { |y, yh| (y - yh)**2 }
        end
      end
    end

    def mst
      @mst ||= sst / (weight_sum - 1)
    end

    def mse
      @mse ||= sse / degrees_of_freedom
    end

    # weights are treated as occurrence counts
    # so zero-weight rows do not count toward degrees of freedom
    def degrees_of_freedom
      weight_sum - @coefficients.size
    end

    def weight_sum
      @weight_sum ||= @train_set.weight ? @train_set.weight.sum.to_f : @train_set.size.to_f
    end

    def mean(arr)
      arr.sum / arr.size.to_f
    end
  end
end
