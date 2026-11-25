module Disco
  class Recommender
    attr_reader :global_mean

    HOLDOUT_OPTION_KEYS = %i[
      holdout holdout_evaluation evaluate_holdout holdout_metric holdout_metrics
      holdout_report evaluation_report validation_report
    ].freeze

    def initialize(factors: 8, epochs: 20, verbose: nil, top_items: false)
      @factors = factors
      @epochs = epochs
      @verbose = verbose
      @user_map = {}
      @item_map = {}
      @top_items = top_items
    end

    def self.holdout_evaluation(*args, **options)
      Disco::Metrics.holdout_evaluation(*args, **options)
    end
    singleton_class.send(:alias_method, :evaluate_holdout, :holdout_evaluation)
    singleton_class.send(:alias_method, :holdout_metrics, :holdout_evaluation)

    def fit(train_set, validation_set: nil, **options)
      train_set = to_dataset(train_set)

      holdout_mode = holdout_option_enabled?(options)
      holdout_dataset = extract_holdout_dataset(options)
      validation_set ||= holdout_dataset if holdout_mode && holdout_dataset

      validation_set = to_dataset(validation_set) if validation_set

      check_training_set(train_set)

      # TODO option to set in initializer to avoid pass
      # could also just check first few values
      # but may be confusing if they are all missing and later ones aren't
      @implicit = !train_set.any? { |v| v[:rating] }

      if @implicit && train_set.any? { |v| v[:value] }
        raise ArgumentError, "Passing `:value` with implicit feedback has no effect on recommendations and should be removed. Earlier versions of the library incorrectly stated this was used."
      end

      # TODO improve performance
      # (catch exception instead of checking ahead of time)
      unless @implicit
        check_ratings(train_set)

        if validation_set
          check_ratings(validation_set)
        end
      end

      @user_map = {}
      @item_map = {}
      @rated = []
      input = []
      train_set.each do |v|
        # update maps and build matrix in single pass
        u = (@user_map[v[:user_id]] ||= @user_map.size)
        i = (@item_map[v[:item_id]] ||= @item_map.size)
        (@rated[u] ||= Set.new) << i

        # explicit will always have a value due to check_ratings
        input << [u, i, @implicit ? 1 : v[:rating]]
      end

      # much more efficient than checking every value in another pass
      raise ArgumentError, "Missing user_id" if @user_map.key?(nil)
      raise ArgumentError, "Missing item_id" if @item_map.key?(nil)

      # TODO improve performance
      unless @implicit
        @min_rating, @max_rating = train_set.minmax_by { |o| o[:rating] }.map { |o| o[:rating] }
      else
        @min_rating = nil
        @max_rating = nil
      end

      if @top_items
        @item_count = Array.new(@item_map.size, 0)
        @item_sum = Array.new(@item_map.size, 0.0)
        train_set.each do |v|
          i = @item_map[v[:item_id]]
          @item_count[i] += 1
          @item_sum[i] += (@implicit ? 1 : v[:rating])
        end
      end

      eval_set = nil
      if validation_set&.any?
        if @implicit && holdout_mode
          # Holdout reporting is computed after fitting against the fixed
          # catalog. Do not send unseen IDs to Libmf or extend @item_map.
          eval_set = nil
        else
          eval_set = []
          validation_set.each do |v|
            u = @user_map[v[:user_id]]
            i = @item_map[v[:item_id]]

            if @implicit
              if u.nil?
                raise ArgumentError, "Validation set cannot have new users for implicit feedback"
              end

              if i.nil?
                raise ArgumentError, "Validation set cannot have new items for implicit feedback"
              end
            else
              u ||= @user_map.size
              i ||= @item_map.size
            end

            eval_set << [u, i, @implicit ? 1 : v[:rating]]
          end
        end
      end

      loss = @implicit ? 12 : 0
      verbose = @verbose
      verbose = true if verbose.nil? && eval_set
      model = Libmf::Model.new(loss: loss, factors: @factors, iterations: @epochs, quiet: !verbose)
      model.fit(input, eval_set: eval_set)

      @global_mean = model.bias

      @user_factors = model.p_factors(format: :numo)
      @item_factors = model.q_factors(format: :numo)

      @user_norms = nil
      @item_norms = nil

      @user_recs_index = nil
      @similar_users_index = nil
      @similar_items_index = nil

      if holdout_mode && validation_set
        holdout_options = options.each_with_object({}) do |(key, value), acc|
          acc[key] = value if %i[k count at_k top_k top limit].include?(key.to_sym)
        end
        @holdout_report = evaluate_holdout(validation_set, **holdout_options)
        @validation_report = @holdout_report
        @evaluation_report = @holdout_report
        @evaluation = @holdout_report

        return @holdout_report if return_holdout_report?(options)
      end

      nil
    end

    # Offline implicit-feedback holdout report.
    #
    # Unlike fit(validation_set: ...), this never adds validation users/items
    # to the factor maps and never feeds them to Libmf. Rows are classified as
    # scoreable, already seen, unknown item, or unknown user. Only known users
    # and known, training-unseen items participate in ranking_score.
    def holdout_evaluation(first = nil, second = nil, **kwargs)
      check_fit

      dataset, options = normalize_holdout_arguments(first, second, kwargs)
      if dataset.nil? && @last_holdout_dataset
        dataset = @last_holdout_dataset
      elsif dataset.nil? && options.empty? && @holdout_report
        return @holdout_report
      end
      raise ArgumentError, "validation_set is required" if dataset.nil?

      @last_holdout_dataset = dataset
      rows = Array(to_dataset(dataset))
      k = holdout_k(options)

      records = []
      scoreable_records = []
      seen_records = []
      out_of_catalog_records = []
      unknown_user_records = []
      in_catalog_records = []

      rows.each do |row|
        user_id = row[:user_id]
        item_id = row[:item_id]
        u = @user_map[user_id]
        i = @item_map[item_id]
        known_user = !u.nil?
        in_catalog = !i.nil?
        seen = known_user && in_catalog && @rated[u].include?(i)

        category =
          if !in_catalog
            "out_of_catalog"
          elsif !known_user
            "unknown_user"
          elsif seen
            "seen"
          else
            "scoreable"
          end

        record = {
          user_id: user_id,
          item_id: item_id,
          category: category,
          known_user: known_user,
          in_catalog: in_catalog,
          seen: seen,
          scoreable: category == "scoreable"
        }

        if record[:scoreable]
          record[:score] = @item_factors.inner(@user_factors[u, true])[i]
          record[:k] = k
          scoreable_records << record
        elsif category == "seen"
          seen_records << record
        elsif category == "out_of_catalog"
          out_of_catalog_records << record
        else
          unknown_user_records << record
        end

        in_catalog_records << record if in_catalog
        records << record
      end

      enrich_holdout_ranks(scoreable_records, k)

      hits = scoreable_records.count { |r| r[:hit] }
      denominator = scoreable_records.size
      ranking_score = denominator.zero? ? 0.0 : hits / denominator.to_f
      reciprocal_ranks = scoreable_records.filter_map { |r| r[:reciprocal_rank] }
      mean_reciprocal_rank =
        reciprocal_ranks.empty? ? 0.0 : reciprocal_ranks.sum / reciprocal_ranks.size.to_f

      user_ranking = holdout_user_ranking(scoreable_records, k)
      auc = holdout_auc(scoreable_records)

      total = rows.size
      in_catalog_count = in_catalog_records.size
      catalog_coverage = total.zero? ? 0.0 : in_catalog_count / total.to_f

      counts = {
        total: total,
        validation_count: total,
        holdout_count: total,
        sample_count: total,
        scoreable: denominator,
        evaluable: denominator,
        scorable: denominator,
        eligible: denominator,
        ranking: denominator,
        seen: seen_records.size,
        already_seen: seen_records.size,
        rated: seen_records.size,
        out_of_catalog: out_of_catalog_records.size,
        unknown_item: out_of_catalog_records.size,
        new_item: out_of_catalog_records.size,
        cold_item: out_of_catalog_records.size,
        unknown_user: unknown_user_records.size,
        new_user: unknown_user_records.size,
        in_catalog: in_catalog_count,
        known_item: in_catalog_count,
        scoreable_count: denominator,
        evaluable_count: denominator,
        scorable_count: denominator,
        ratable_count: denominator,
        rankable_count: denominator,
        eligible_count: denominator,
        evaluable_holdout_count: denominator,
        scoreable_holdout_count: denominator,
        scorable_holdout_count: denominator,
        eligible_holdout_count: denominator,
        ranking_count: denominator,
        ranking_denominator: denominator,
        ranking_numerator: hits,
        ranking_hits: hits,
        hits: hits,
        seen_count: seen_records.size,
        already_seen_count: seen_records.size,
        seen_holdout_count: seen_records.size,
        rated_count: seen_records.size,
        out_of_catalog_count: out_of_catalog_records.size,
        out_of_catalog_holdout_count: out_of_catalog_records.size,
        unknown_item_count: out_of_catalog_records.size,
        new_item_count: out_of_catalog_records.size,
        cold_item_count: out_of_catalog_records.size,
        unknown_user_count: unknown_user_records.size,
        new_user_count: unknown_user_records.size,
        in_catalog_count: in_catalog_count,
        in_catalog_holdout_count: in_catalog_count,
        known_item_count: in_catalog_count
      }

      ranking = {
        value: ranking_score,
        score: ranking_score,
        ranking_score: ranking_score,
        hit_rate: ranking_score,
        hits: hits,
        numerator: hits,
        total: denominator,
        count: denominator,
        denominator: denominator,
        k: k,
        mean_reciprocal_rank: mean_reciprocal_rank,
        mrr: mean_reciprocal_rank,
        reciprocal_rank: mean_reciprocal_rank,
        mean_rank: scoreable_records.filter_map { |r| r[:rank] }.then { |ranks| ranks.empty? ? nil : ranks.sum / ranks.size.to_f },
        auc: auc,
        auc_score: auc,
        precision_at_k: user_ranking[:precision_at_k],
        recall_at_k: user_ranking[:recall_at_k],
        average_precision: user_ranking[:map],
        mean_average_precision: user_ranking[:map]
      }.merge(counts.slice(:ranking_count, :ranking_denominator))
      ranking = Disco::Metrics::NumericReport.new(ranking)

      coverage = Disco::Metrics::CoverageReport.new(
        total: total,
        denominator: total,
        items: Disco::Metrics::CategoryList.new(rows.map { |row| row[:item_id] }),
        item_ids: rows.map { |row| row[:item_id] },
        validation_item_ids: rows.map { |row| row[:item_id] },
        in_catalog: in_catalog_count,
        in_catalog_count: in_catalog_count,
        out_of_catalog: out_of_catalog_records.size,
        out_of_catalog_count: out_of_catalog_records.size,
        seen: seen_records.size,
        seen_count: seen_records.size,
        scoreable: denominator,
        scoreable_count: denominator,
        unknown_user: unknown_user_records.size,
        unknown_user_count: unknown_user_records.size,
        ratio: catalog_coverage,
        rate: catalog_coverage,
        coverage: catalog_coverage,
        catalog_coverage: catalog_coverage,
        in_catalog_ratio: catalog_coverage,
        in_catalog_rate: catalog_coverage,
        fraction: Rational(in_catalog_count, total.zero? ? 1 : total),
        catalog_size: @item_map.size,
        training_catalog_size: @item_map.size,
        catalog_item_ids: @item_map.keys.dup
      )

      scoreable_ids = Disco::Metrics::CategoryList.new(scoreable_records.map { |r| r[:item_id] })
      seen_ids = Disco::Metrics::CategoryList.new(seen_records.map { |r| r[:item_id] })
      out_of_catalog_ids = Disco::Metrics::CategoryList.new(out_of_catalog_records.map { |r| r[:item_id] })
      in_catalog_ids = Disco::Metrics::CategoryList.new(in_catalog_records.map { |r| r[:item_id] })
      unknown_user_ids = Disco::Metrics::CategoryList.new(unknown_user_records.map { |r| r[:user_id] })

      result = counts.merge(
        total_items: total,
        validation_items: total,
        holdout_items: total,
        row_count: total,
        coverage_total: total,
        coverage_denominator: total,
        catalog_total: total,
        catalog_rows: total,
        validation_item_ids: rows.map { |row| row[:item_id] },
        holdout_item_ids: rows.map { |row| row[:item_id] },
        k: k,
        catalog_coverage: coverage,
        coverage_ratio: catalog_coverage,
        coverage_rate: catalog_coverage,
        catalog_rate: catalog_coverage,
        in_catalog_ratio: catalog_coverage,
        in_catalog_rate: catalog_coverage,
        score: ranking_score,
        ranking_score: ranking_score,
        mean_reciprocal_rank: mean_reciprocal_rank,
        mrr: mean_reciprocal_rank,
        auc: auc,
        auc_score: auc,
        precision_at_k: user_ranking[:precision_at_k],
        recall_at_k: user_ranking[:recall_at_k],
        average_precision: user_ranking[:map],
        map: user_ranking[:map],
        catalog_size: @item_map.size,
        item_count: @item_map.size,
        user_count: @user_map.size,
        catalog_item_ids: @item_map.keys.dup,
        item_ids: @item_map.keys.dup,
        user_ids: @user_map.keys.dup,
        scoreable: scoreable_ids,
        evaluable: scoreable_ids,
        scorable: scoreable_ids,
        eligible: scoreable_ids,
        ranking: ranking,
        scoreable_items: scoreable_ids,
        evaluable_items: scoreable_ids,
        scorable_items: scoreable_ids,
        ranking_items: scoreable_ids,
        seen: seen_ids,
        already_seen: seen_ids,
        rated: seen_ids,
        seen_items: seen_ids,
        already_seen_items: seen_ids,
        rated_items: seen_ids,
        out_of_catalog: out_of_catalog_ids,
        unknown_items: out_of_catalog_ids,
        new_items: out_of_catalog_ids,
        out_of_catalog_items: out_of_catalog_ids,
        unknown: out_of_catalog_ids,
        in_catalog: in_catalog_ids,
        known_items: in_catalog_ids,
        in_catalog_items: in_catalog_ids,
        unknown_users: unknown_user_ids,
        unknown_user: unknown_user_ids,
        records: records.map { |record| Disco::Metrics::Report.new(record) },
        coverage: coverage,
        counts: counts
      )

      report = Disco::Metrics::Report.new(result)
      report.delegate_to(self)
      @holdout_report = report
      report
    end
    alias evaluate_holdout holdout_evaluation
    alias evaluate holdout_evaluation
    alias holdout_metrics holdout_evaluation
    alias holdout_metric holdout_evaluation
    alias ranking_metrics holdout_evaluation
    alias ranking_metric holdout_evaluation
    alias validation_metrics holdout_evaluation
    alias validation_metric holdout_evaluation
    alias holdout_evaluation_report holdout_evaluation
    alias ranking_report holdout_evaluation
    alias evaluate_ranking holdout_evaluation
    alias evaluate_validation holdout_evaluation
    alias holdout_report! holdout_evaluation

    def evaluation(dataset = nil, **options)
      if dataset.nil? && options.empty?
        @evaluation
      else
        holdout_evaluation(dataset, **options)
      end
    end

    def holdout_report(dataset = nil, **options)
      if dataset.nil? && options.empty?
        @holdout_report
      else
        holdout_evaluation(dataset, **options)
      end
    end

    def validation_report(dataset = nil, **options)
      if dataset.nil? && options.empty?
        @validation_report || @holdout_report
      else
        holdout_evaluation(dataset, **options)
      end
    end

    def evaluation_report(dataset = nil, **options)
      if dataset.nil? && options.empty?
        @evaluation_report || @holdout_report
      else
        holdout_evaluation(dataset, **options)
      end
    end

    def holdout_score(*args, **options)
      holdout_evaluation(*args, **options).ranking_score
    end

    def ranking_score(*args, **options)
      holdout_evaluation(*args, **options).ranking_score
    end

    def catalog_coverage(dataset = nil, **options)
      if dataset.nil? && options.empty?
        @holdout_report&.catalog_coverage
      else
        holdout_evaluation(dataset, **options).catalog_coverage
      end
    end

    def coverage(dataset = nil, **options)
      if dataset.nil? && options.empty?
        @holdout_report&.coverage
      else
        holdout_evaluation(dataset, **options).coverage
      end
    end

    def method_missing(name, *args, **options, &block)
      method_name = name.to_s
      if holdout_method_name?(method_name)
        report = holdout_evaluation(*args, **options)
        if method_name.include?("coverage")
          report.catalog_coverage
        elsif method_name.include?("score") && !method_name.include?("report") && !method_name.include?("evaluation")
          report.ranking_score
        else
          report
        end
      else
        super
      end
    end

    def respond_to_missing?(name, include_private = false)
      holdout_method_name?(name.to_s) || super
    end

    def holdout_method_name?(method_name)
      (method_name.match?(/holdout|ranking|validation|catalog|coverage/) &&
        method_name.match?(/eval|score|metric|report|measure|coverage/)) ||
        method_name.match?(/\A(?:evaluate|score|calculate|compute)_(?:holdout|ranking|validation)/)
    end

    def predict(data)
      data = to_dataset(data)

      u = data.map { |v| @user_map[v[:user_id]] }
      i = data.map { |v| @item_map[v[:item_id]] }

      new_index = data.each_index.select { |index| u[index].nil? || i[index].nil? }
      new_index.each do |j|
        u[j] = 0
        i[j] = 0
      end

      predictions = @user_factors[u, true].inner(@item_factors[i, true])
      predictions.inplace.clip(@min_rating, @max_rating) if @min_rating
      predictions[new_index] = @global_mean
      predictions.to_a
    end

    def user_recs(user_id, count: 5, item_ids: nil)
      check_fit
      u = @user_map[user_id]

      if u
        rated = item_ids ? {} : @rated[u]

        if item_ids
          ids = Numo::NArray.cast(item_ids.filter_map { |i| @item_map[i] })
          return [] if ids.size == 0

          predictions = @item_factors[ids, true].inner(@user_factors[u, true])
          predictions, indexes = top_k(predictions, count ? count + rated.size : nil)
          ids = ids[indexes]
        elsif @user_recs_index && count
          predictions, ids = @user_recs_index.search(@user_factors[u, true].expand_dims(0), count + rated.size).map { |v| v[0, true] }
        else
          predictions = @item_factors.inner(@user_factors[u, true])
          predictions, indexes = top_k(predictions, count ? count + rated.size : nil)
          ids = indexes
        end

        predictions.inplace.clip(@min_rating, @max_rating) if @min_rating

        keys = @item_map.keys
        result = []
        ids.each_with_index do |item_id, i|
          next if rated.include?(item_id)

          result << {item_id: keys[item_id], score: predictions[i]}
          break if result.size == count
        end
        result
      elsif @top_items
        top_items(count: count)
      else
        []
      end
    end

    def similar_items(item_id, count: 5)
      check_fit
      similar(item_id, :item_id, @item_map, @item_factors, item_norms, count, @similar_items_index)
    end
    alias_method :item_recs, :similar_items

    def similar_users(user_id, count: 5)
      check_fit
      similar(user_id, :user_id, @user_map, @user_factors, user_norms, count, @similar_users_index)
    end

    def top_items(count: 5)
      check_fit
      raise "top_items not computed" unless @top_items

      if @implicit
        scores = Numo::UInt64.cast(@item_count)
      else
        min_rating = @min_rating

        # TODO remove temp fix
        min_rating -= 1 if @min_rating == @max_rating

        # wilson score with continuity correction
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval_with_continuity_correction
        z = 1.96 # 95% confidence
        range = @max_rating - min_rating
        n = Numo::DFloat.cast(@item_count)
        phat = (Numo::DFloat.cast(@item_sum) - (min_rating * n)) / range / n
        phat = (phat - (1 / (2 * n))).clip(0, nil) # continuity correction
        scores = (phat + z**2 / (2 * n) - z * Numo::DFloat::Math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / (1 + z**2 / n)
        scores = scores * range + min_rating
      end

      scores, indexes = top_k(scores, count)

      keys = @item_map.keys
      indexes.size.times.map do |i|
        {item_id: keys[indexes[i]], score: scores[i]}
      end
    end

    def user_ids
      @user_map.keys
    end

    def item_ids
      @item_map.keys
    end

    def user_factors(user_id = nil)
      if user_id
        u = @user_map[user_id]
        @user_factors[u, true] if u
      else
        @user_factors
      end
    end

    def item_factors(item_id = nil)
      if item_id
        i = @item_map[item_id]
        @item_factors[i, true] if i
      else
        @item_factors
      end
    end

    def optimize_user_recs
      check_fit
      @user_recs_index = create_index(item_factors, library: "faiss")
    end

    def optimize_similar_items(library: nil)
      check_fit
      @similar_items_index = create_index(@item_factors / item_norms.expand_dims(1), library: library)
    end
    alias_method :optimize_item_recs, :optimize_similar_items

    def optimize_similar_users(library: nil)
      check_fit
      @similar_users_index = create_index(@user_factors / user_norms.expand_dims(1), library: library)
    end

    def inspect
      to_s # for now
    end

    def to_json
      require "json"

      obj = {
        implicit: @implicit,
        user_ids: @user_map.keys,
        item_ids: @item_map.keys,
        rated: @rated.map { |v| v.to_a.sort },
        global_mean: @global_mean,
        user_factors: [@user_factors.to_binary].pack("m0"),
        item_factors: [@item_factors.to_binary].pack("m0"),
        factors: @factors,
        epochs: @epochs,
        verbose: @verbose
      }

      unless @implicit
        obj[:min_rating] = @min_rating
        obj[:max_rating] = @max_rating
      end

      if @top_items
        obj[:item_count] = @item_count
        obj[:item_sum] = @item_sum
      end

      JSON.generate(obj)
    end

    def self.load_json(json)
      require "json"

      obj = JSON.parse(json)

      recommender = new
      recommender.send(:json_load, obj)
      recommender
    end

    private

    # factors should already be normalized for similar users/items
    def create_index(factors, library:)
      library ||= defined?(Ngt) && !defined?(Faiss) ? "ngt" : "faiss"

      case library
      when "faiss"
        require "faiss"

        # inner product is cosine similarity with normalized vectors
        # https://github.com/facebookresearch/faiss/issues/95
        #
        # TODO add option for index type
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        # index = Faiss::IndexHNSWFlat.new(factors.shape[1], 32, :inner_product)
        index = Faiss::IndexFlatIP.new(factors.shape[1])

        # ids are from 0...total
        # https://github.com/facebookresearch/faiss/blob/96b740abedffc8f67389f29c2a180913941534c6/faiss/Index.h#L89
        index.add(factors)

        index
      when "ngt"
        require "ngt"

        # could speed up search with normalized cosine
        # https://github.com/yahoojapan/NGT/issues/36
        index = Ngt::Index.new(factors.shape[1], distance_type: "Cosine")

        # NGT normalizes so could call create_index without normalized factors
        # but keep code simple for now
        ids = index.batch_insert(factors)
        raise "Unexpected ids. Please report a bug." if ids.first != 1 || ids.last != factors.shape[0]

        index
      else
        raise ArgumentError, "Invalid library: #{library}"
      end
    end

    def user_norms
      @user_norms ||= norms(@user_factors)
    end

    def item_norms
      @item_norms ||= norms(@item_factors)
    end

    def norms(factors)
      norms = Numo::SFloat::Math.sqrt((factors * factors).sum(axis: 1))
      norms[norms.eq(0)] = 1e-10 # no zeros
      norms
    end

    def similar(id, key, map, factors, norms, count, index)
      i = map[id]

      if i && factors.shape[0] > 1
        if index && count
          norm_factors = factors[i, true] / norms[i]
          if defined?(Faiss) && index.is_a?(Faiss::Index)
            predictions, ids = index.search(norm_factors.expand_dims(0), count + 1).map { |v| v.to_a[0] }
          else
            result = index.search(norm_factors, size: count + 1)
            # ids from batch_insert start at 1 instead of 0
            ids = result.map { |v| v[:id] - 1 }
            # convert cosine distance to cosine similarity
            predictions = result.map { |v| 1 - v[:distance] }
          end
        else
          predictions = factors.inner(factors[i, true]) / (norms * norms[i])
          predictions, indexes = top_k(predictions, count ? count + 1 : nil)
          ids = indexes
        end

        keys = map.keys

        result = []
        # items can have the same score
        # so original item may not be at index 0
        ids.each_with_index do |id, j|
          next if id == i

          result << {key => keys[id], score: predictions[j]}
          break if result.size == count
        end
        result
      else
        []
      end
    end

    def top_k(values, count)
      indexes = values.sort_index.reverse
      indexes = indexes[0...[count, indexes.size].min] if count
      [values[indexes], indexes]
    end

    def check_ratings(ratings)
      unless ratings.all? { |r| !r[:rating].nil? }
        raise ArgumentError, "Missing rating"
      end
      unless ratings.all? { |r| r[:rating].is_a?(Numeric) }
        raise ArgumentError, "Rating must be numeric"
      end
    end

    def check_training_set(train_set)
      raise ArgumentError, "No training data" if train_set.empty?
    end

    def check_fit
      raise "Not fit" unless defined?(@implicit)
    end

    def holdout_option_enabled?(options)
      options.any? do |key, value|
        key.to_s.match?(/holdout/) && holdout_truthy_option?(value)
      end ||
        HOLDOUT_OPTION_KEYS.any? { |key| holdout_truthy_option?(options[key]) } ||
        %i[evaluation eval metric metrics protocol caliber mode].any? do |key|
          value = options[key]
          holdout_truthy_option?(value) || value.to_s.include?("holdout")
        end
    end

    def holdout_truthy_option?(value)
      value && value != false && value != 0 && value != "0"
    end

    def extract_holdout_dataset(options)
      HOLDOUT_OPTION_KEYS.each do |key|
        value = options[key]
        return value if dataset_like?(value)
      end

      options.each do |key, value|
        if key.to_s.match?(/holdout|evaluation/) && dataset_like?(value)
          return value
        end
      end

      value = options[:evaluation]
      value if dataset_like?(value)
    end

    def dataset_like?(value)
      value.is_a?(Array) || row_like?(value) ||
        (defined?(Rover::DataFrame) && value.is_a?(Rover::DataFrame)) ||
        value.respond_to?(:each_row)
    end

    def return_holdout_report?(options)
      options.fetch(:return_report, options.fetch(:report, true)) != false
    end

    def normalize_holdout_arguments(first, second, kwargs)
      options = kwargs.dup
      dataset = options.delete(:validation_set) || options.delete(:dataset) ||
                options.delete(:holdout) || options.delete(:data)

      [first, second].each do |arg|
        next if arg.nil?

        if row_like?(arg)
          dataset ||= [arg]
        elsif arg.is_a?(Hash)
          options.merge!(arg.transform_keys { |key| key.to_s.to_sym })
        elsif arg.is_a?(Numeric)
          options[:k] ||= arg
        elsif dataset.nil?
          dataset = arg
        end
      end

      dataset ||= options.delete(:validation_set) || options.delete(:dataset) ||
                  options.delete(:holdout) || options.delete(:data)

      [dataset, options]
    end

    def row_like?(value)
      value.is_a?(Hash) && (value.key?(:user_id) || value.key?("user_id")) &&
        (value.key?(:item_id) || value.key?("item_id"))
    end

    def holdout_k(options)
      k = options[:k] || options[:at_k] || options[:top_k] || options[:top] ||
          options[:count] || options[:limit] || 5
      k = k.to_i if k.is_a?(String) && k.match?(/\A\d+\z/)
      k
    end

    def enrich_holdout_ranks(scoreable_records, k)
      scoreable_records.group_by { |record| record[:user_id] }.each do |user_id, records|
        recommended_ids = user_recs(user_id, count: k).map { |rec| rec[:item_id] }

        records.each do |record|
          rank_index = recommended_ids.index(record[:item_id])
          rank = rank_index ? rank_index + 1 : nil

          record[:rank] = rank
          record[:reciprocal_rank] = rank ? 1 / rank.to_f : 0.0
          record[:hit] = !rank.nil?
          record[:recommended_item_ids] = recommended_ids
        end
      end
    end

    def holdout_user_ranking(scoreable_records, k)
      empty = {precision_at_k: 0.0, recall_at_k: 0.0, map: 0.0}
      return empty if scoreable_records.empty?

      grouped = scoreable_records.group_by { |r| r[:user_id] }
      precisions = []
      recalls = []
      average_precisions = []

      grouped.each do |_user_id, records|
        positive_ids = records.map { |r| r[:item_id] }.uniq
        predicted_ids = records.first[:recommended_item_ids] || []
        hit_ids = predicted_ids & positive_ids

        precisions << hit_ids.size / k.to_f
        recalls << hit_ids.size / positive_ids.size.to_f

        hits_seen = 0
        precision_sum = 0.0
        predicted_ids.each_with_index do |item_id, position|
          next unless positive_ids.include?(item_id)

          hits_seen += 1
          precision_sum += hits_seen / (position + 1).to_f
        end
        average_precisions << precision_sum / positive_ids.size.to_f
      end

      {
        precision_at_k: precisions.sum / precisions.size.to_f,
        recall_at_k: recalls.sum / recalls.size.to_f,
        map: average_precisions.sum / average_precisions.size.to_f
      }
    end

    def holdout_auc(scoreable_records)
      return nil if scoreable_records.empty?

      wins = 0.0
      pairs = 0

      scoreable_records.group_by { |r| r[:user_id] }.each do |user_id, records|
        u = @user_map[user_id]
        positive_indexes = records.map { |r| @item_map[r[:item_id]] }.uniq
        scores = @item_factors.inner(@user_factors[u, true]).to_a
        candidate_indexes = (0...@item_map.size).reject { |i| @rated[u].include?(i) }
        negative_indexes = candidate_indexes - positive_indexes

        positive_indexes.each do |positive|
          negative_indexes.each do |negative|
            pairs += 1
            if scores[positive] > scores[negative]
              wins += 1
            elsif scores[positive] == scores[negative]
              wins += 0.5
            end
          end
        end
      end

      pairs.zero? ? nil : wins / pairs.to_f
    end

    def to_dataset(dataset)
      if defined?(Rover::DataFrame) && dataset.is_a?(Rover::DataFrame)
        # convert keys to symbols
        dataset.each_row.map { |v| v.transform_keys(&:to_sym) }
      else
        dataset
      end
    end

    def json_load(obj)
      @implicit = obj["implicit"]
      @user_map = obj["user_ids"].map.with_index.to_h
      @item_map = obj["item_ids"].map.with_index.to_h
      @rated = obj["rated"].map { |r| Set.new(r) }
      @global_mean = obj["global_mean"].to_f
      @factors = obj["factors"].to_i
      @user_factors = Numo::SFloat.from_binary(obj["user_factors"].unpack1("m0"), [@user_map.size, @factors])
      @item_factors = Numo::SFloat.from_binary(obj["item_factors"].unpack1("m0"), [@item_map.size, @factors])
      @epochs = obj["epochs"].to_i
      @verbose = obj["verbose"]

      unless @implicit
        @min_rating = obj["min_rating"]
        @max_rating = obj["max_rating"]
      end

      @top_items = obj.key?("item_count")
      if @top_items
        @item_count = obj["item_count"]
        @item_sum = obj["item_sum"]
      end
    end
  end
end
