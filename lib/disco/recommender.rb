module Disco
  class Recommender
    attr_reader :global_mean

    def max_per_category(value = nil)
      return @max_per_category if value.nil?

      self.max_per_category = value
    end
    alias category_limit max_per_category

    LIMIT_OPTION_KEYS = %i[
      max_per_category max_per_category_limit per_category_limit category_limit
      category_cap per_category max_same_category max_per_same_category
      same_category_limit same_category_max max_items_per_category
      limit_per_category category_quota category_quotas category_max limit
    ].freeze
    private_constant :LIMIT_OPTION_KEYS

    CATEGORY_OPTION_KEYS = %i[
      categories item_categories category_map category_for category_resolver
      categorize classifier category
    ].freeze
    private_constant :CATEGORY_OPTION_KEYS

    def initialize(factors: 8, epochs: 20, verbose: nil, top_items: false,
                   max_per_category: nil, category_limit: nil,
                   categories: nil, item_categories: nil, **options, &category_block)
      @factors = factors
      @epochs = epochs
      @verbose = verbose
      @user_map = {}
      @item_map = {}
      @top_items = top_items
      @categories = categories || category_block
      @item_categories = item_categories || {}
      @training_categories = {}

      limit = max_per_category || category_limit || limit_from_options(options)
      self.max_per_category = limit

      source = category_source_from_options(options)
      @categories ||= source unless source.nil? || source.is_a?(Numeric)
    end

    def max_per_category=(value)
      @max_per_category = normalize_category_limit(value)
      @category_limit = @max_per_category
    end
    alias category_limit= max_per_category=
    alias per_category max_per_category
    alias per_category= max_per_category=
    alias category_cap max_per_category
    alias category_cap= max_per_category=
    alias limit max_per_category
    alias limit= max_per_category=

    attr_writer :item_categories

    def categories(source = nil, &block)
      if block_given?
        @categories = block
      elsif !source.nil?
        @categories = source
      else
        @categories
      end
    end

    def categories=(source)
      @categories = source
    end
    alias category_map categories
    alias category_map= categories=
    alias category_resolver categories
    alias category_resolver= categories=

    def item_categories
      if @item_categories.is_a?(Hash)
        @training_categories.merge(@item_categories)
      else
        @item_categories || @training_categories
      end
    end

    def category_for(item_id)
      internal_id = @item_map[item_id]

      found, category = category_from_source(@categories, item_id, internal_id)
      return category if found

      found, category = category_from_source(@item_categories, item_id, internal_id)
      return category if found

      return @training_categories[item_id] if @training_categories.key?(item_id)
      return category(item_id) if respond_to?(:category)

      nil
    end

    def fit(train_set, validation_set: nil)
      train_set = to_dataset(train_set)
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
      @training_categories = {}
      input = []
      train_set.each do |v|
        # update maps and build matrix in single pass
        u = (@user_map[v[:user_id]] ||= @user_map.size)
        i = (@item_map[v[:item_id]] ||= @item_map.size)
        (@rated[u] ||= Set.new) << i

        category = v[:category]
        category = v[:category_id] if category.nil?
        category = v["category"] if category.nil?
        @training_categories[v[:item_id]] = category unless category.nil?

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
    end

    # generates a prediction even if a user has already rated the item
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

    def user_recs(user_id, count: 5, item_ids: nil, max_per_category: nil,
                  category_limit: nil, categories: nil, item_categories: nil,
                  **options, &block)
      check_fit
      u = @user_map[user_id]

      limit = user_category_limit(max_per_category, category_limit, options)
      if limit
        source = user_category_source(categories, item_categories, options)
        return user_recs_with_category_limit(
          user_id, u, count: count, item_ids: item_ids, limit: limit,
          category_source: source, category_block: block
        )
      end

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

    private def user_recs_with_category_limit(_external_user_id, u, count:, item_ids:, limit:,
                                              category_source:, category_block:)
      return [] if count == 0

      unless u
        return [] unless @top_items

        ranked = top_items(count: nil)
        ids = ranked.map { |rec| @item_map[rec[:item_id]] }
        predictions = ranked.map { |rec| rec[:score] }
        return select_by_category(
          ids, predictions, {}, count, limit, category_source, category_block
        )
      end

      # Explicit candidate lists preserve the existing semantics and do not
      # remove items the user has already rated.
      rated = item_ids ? {} : @rated[u]
      query = @user_factors[u, true]

      if item_ids
        if category_source.is_a?(Array) && category_source.size == item_ids.size
          category_source = item_ids.zip(category_source).to_h
        end

        ids = Numo::NArray.cast(item_ids.filter_map { |i| @item_map[i] })
        return [] if ids.size == 0

        predictions = @item_factors[ids, true].inner(query)
        predictions, indexes = top_k(predictions, nil)
        ids = ids[indexes]
        predictions.inplace.clip(@min_rating, @max_rating) if @min_rating

        return select_by_category(
          ids, predictions, rated, count, limit, category_source, category_block
        )
      end

      if @user_recs_index && count
        total = @item_map.size
        fetch_count = [count + rated.size, total].min

        loop do
          predictions, ids = @user_recs_index
            .search(query.expand_dims(0), fetch_count)
            .map { |v| v[0, true] }
          predictions.inplace.clip(@min_rating, @max_rating) if @min_rating

          result = select_by_category(
            ids, predictions, rated, count, limit, category_source, category_block
          )
          return result if result.size == count

          returned = ids.respond_to?(:size) ? ids.size : ids.length
          return result if returned < fetch_count || fetch_count >= total

          # Grow the ranked prefix only as far as necessary. This preserves
          # the index order without rescoring or requesting the whole catalog
          # before a replacement candidate is known to be needed.
          fetch_count = [fetch_count + 1, total].min
        end
      else
        predictions = @item_factors.inner(query)
        predictions, indexes = top_k(predictions, nil)
        predictions.inplace.clip(@min_rating, @max_rating) if @min_rating

        select_by_category(
          indexes, predictions, rated, count, limit, category_source, category_block
        )
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

      obj[:max_per_category] = @max_per_category if @max_per_category

      serialized_categories = @item_map.keys.map { |item_id| category_for(item_id) }
      if serialized_categories.any? { |category| !category.nil? }
        obj[:item_categories] = serialized_categories
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

    def limit_from_options(options)
      value = LIMIT_OPTION_KEYS.lazy.map { |key| options[key] }.find { |v| !v.nil? }
      value = options[:category] if value.nil? && options[:category].is_a?(Numeric)
      value
    end

    def category_source_from_options(options)
      value = CATEGORY_OPTION_KEYS.lazy.map { |key| options[key] }.find { |v| !v.nil? }
      value = nil if value.is_a?(Numeric)

      if value.is_a?(Hash) && value.key?(:limit) &&
         (value.key?(:categories) || value.key?(:source) || value.key?(:for))
        value = value[:categories] || value[:source] || value[:for]
      end

      value
    end

    def user_category_limit(max_per_category, category_limit, options)
      value =
        if !max_per_category.nil?
          max_per_category
        elsif !category_limit.nil?
          category_limit
        else
          limit_from_options(options)
        end
      value = @max_per_category if value.nil?
      value = @category_limit if value.nil? && instance_variable_defined?(:@category_limit)

      normalize_category_limit(value)
    end

    def user_category_source(categories, item_categories, options)
      source = categories unless categories.nil?
      source ||= item_categories unless item_categories.nil?
      source ||= category_source_from_options(options)
      source = nil if source.is_a?(Numeric)
      source
    end

    def normalize_category_limit(value)
      return nil if value.nil? || value == false

      if value.is_a?(Hash)
        value.each_with_object({}) do |(category, quota), normalized|
          normalized[category] = normalize_quota(quota)
        end
      else
        normalize_quota(value)
      end
    end

    def normalize_quota(value)
      value = value.to_int if value.respond_to?(:to_int) && !value.is_a?(Integer)
      unless value.is_a?(Integer) && value.positive?
        raise ArgumentError, "max_per_category must be a positive integer"
      end

      value
    end

    def select_by_category(ids, predictions, rated, count, limit, source, block)
      keys = @item_map.keys
      category_counts = Hash.new(0)
      result = []

      ids.each_with_index do |internal_id, index|
        next if rated.include?(internal_id)

        item_id = keys[internal_id]
        category = resolve_category(item_id, internal_id, source, block)
        quota = category_quota(limit, category)

        next if quota && category_counts[category] >= quota

        category_counts[category] += 1
        result << {item_id: item_id, score: predictions[index]}
        break if count && result.size == count
      end

      result
    end

    def resolve_category(item_id, internal_id, source, block)
      return call_category(block, item_id, internal_id) if block

      if source
        found, category = category_from_source(source, item_id, internal_id)
        return category if found
      end

      category_for(item_id)
    end

    # Returns [found, category]. An explicitly provided nil category is
    # therefore distinguished from a missing source entry.
    def category_from_source(source, item_id, internal_id)
      return [false, nil] if source.nil? || source.is_a?(Numeric)

      if source.is_a?(Hash)
        key = hash_key_for(source, item_id)
        return key ? [true, source[key]] : [false, nil]
      end

      if source.is_a?(Array)
        if internal_id && internal_id >= 0 && internal_id < source.size
          return [true, source[internal_id]]
        end
        return [false, nil]
      end

      if source.is_a?(Symbol)
        return [true, send(source, item_id)] if respond_to?(source, true)
        return [true, source]
      end

      if source.is_a?(String)
        return [true, source]
      end

      if source.respond_to?(:call)
        [true, call_category(source, item_id, internal_id)]
      elsif source.respond_to?(:category_for)
        [true, source.category_for(item_id)]
      elsif source.respond_to?(:[])
        [true, source[item_id]]
      else
        [false, nil]
      end
    end

    def hash_key_for(hash, item_id)
      return item_id if hash.key?(item_id)

      if item_id.is_a?(String)
        symbol = item_id.to_sym
        return symbol if hash.key?(symbol)
      elsif item_id.is_a?(Symbol)
        string = item_id.to_s
        return string if hash.key?(string)
      end

      string = item_id.to_s
      return string if hash.key?(string)

      nil
    end

    def call_category(callable, item_id, internal_id)
      case callable.arity
      when 0
        callable.call
      when 1
        callable.call(item_id)
      else
        callable.call(item_id, internal_id)
      end
    end

    def category_quota(limit, category)
      return limit unless limit.is_a?(Hash)

      keys = [category]
      keys << category.to_s unless category.nil?
      keys << category.to_sym if category.is_a?(String)

      keys.each do |key|
        return limit[key] if limit.key?(key)
      end

      return limit[:default] if limit.key?(:default)
      return limit["default"] if limit.key?("default")

      nil
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

      @training_categories = {}
      @item_categories = {}
      if obj["item_categories"]
        @item_categories = @item_map.keys.zip(obj["item_categories"]).to_h
      end
      @max_per_category = normalize_category_limit(obj["max_per_category"])

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
