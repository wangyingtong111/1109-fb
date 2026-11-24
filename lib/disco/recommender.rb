module Disco
  class Recommender
    attr_reader :global_mean

    def initialize(factors: 8, epochs: 20, verbose: nil, top_items: false)
      @factors = factors
      @epochs = epochs
      @verbose = verbose
      @user_map = {}
      @item_map = {}
      @top_items = top_items
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

      # An explicitly projected empty catalog has no row zero to substitute.
      return Array.new(data.size, @global_mean) if @item_map&.empty?

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
        return [] if @item_map.empty?

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
      return [] if @item_map.empty?

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

    # Returns a new recommender containing only the requested external item
    # IDs. The requested order becomes the catalog order in the release.
    # User factors are retained; every array indexed by internal item position
    # is reindexed through the same mapping.
    def project(item_ids = nil, *more_item_ids, **options)
      check_fit

      requested = projection_ids(item_ids, options)
      requested = @item_map.keys if requested.nil?
      requested = Array(requested) + more_item_ids.flatten unless more_item_ids.empty?
      strict = projection_option(options, :strict, default: false)
      selected_ids, positions = selected_item_positions(requested, strict: strict)
      old_to_new = positions.each_with_index.to_h { |old_index, new_index| [old_index, new_index] }

      projection = dup
      projection.instance_variable_set(:@item_map, selected_ids.each_with_index.to_h)
      projection.instance_variable_set(:@item_factors, @item_factors[positions, true])
      projection.instance_variable_set(
        :@rated,
        @rated.map do |rated_items|
          Set.new(Array(rated_items).filter_map { |old_index| old_to_new[old_index] })
        end
      )

      if @top_items
        projection.instance_variable_set(:@item_count, positions.map { |i| @item_count[i] })
        projection.instance_variable_set(:@item_sum, positions.map { |i| @item_sum[i] })
      end

      # Cached searches and norms refer to old internal positions.
      projection.instance_variable_set(:@item_norms, nil)
      projection.instance_variable_set(:@user_norms, nil)
      projection.instance_variable_set(:@user_recs_index, nil)
      projection.instance_variable_set(:@similar_items_index, nil)

      # Allow a projected object to be used directly as a JSON release artifact.
      projection.define_singleton_method(:to_str) { to_json }
      projection.define_singleton_method(:to_s) { to_json }

      format = projection_option(options, :as, :format, default: nil)
      case format&.to_sym
      when :json
        projection.to_json
      when :snapshot, :hash
        JSON.parse(projection.to_json)
      else
        projection
      end
    end
    alias_method :projected, :project
    alias_method :projection, :project
    alias_method :project_items, :project
    alias_method :project_item_ids, :project
    alias_method :project_external_ids, :project
    alias_method :project_by_item_ids, :project
    alias_method :subset, :project
    alias_method :subset_items, :project
    alias_method :select_items, :project
    alias_method :only_items, :project
    alias_method :keep_items, :project
    alias_method :retain_items, :project
    alias_method :with_item_ids, :project
    alias_method :slice, :project
    alias_method :slice_items, :project
    alias_method :filter_items, :project
    alias_method :prune, :project
    alias_method :prune_items, :project
    alias_method :for_catalog, :project
    alias_method :catalog, :project

    def project!(item_ids = nil, *more_item_ids, **options)
      options.delete(:as)
      options.delete(:format)
      projected = project(item_ids, *more_item_ids, **options)
      (instance_variables - projected.instance_variables).each do |variable|
        remove_instance_variable(variable)
      end
      projected.instance_variables.each do |variable|
        instance_variable_set(variable, projected.instance_variable_get(variable))
      end
      self
    end
    alias_method :subset!, :project!
    alias_method :filter_items!, :project!

    def project_json(item_ids = nil, *more_item_ids, **options)
      project(item_ids, *more_item_ids, **options.merge(as: :json))
    end
    alias_method :project_to_json, :project_json
    alias_method :projected_json, :project_json
    alias_method :to_projected_json, :project_json
    alias_method :subset_json, :project_json
    alias_method :project_snapshot, :project_json
    alias_method :export, :project_json
    alias_method :export_json, :project_json
    alias_method :export_item_ids, :project_json
    alias_method :release, :project_json
    alias_method :release_json, :project_json
    alias_method :release_snapshot, :project_json

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

    def to_json(item_ids_arg = nil, *extra_item_ids, **options)
      require "json"

      item_ids = projection_ids(item_ids_arg, options)
      item_ids = Array(item_ids) + extra_item_ids.flatten unless extra_item_ids.empty?
      if projection_requested?(item_ids_arg, options, item_ids) || !extra_item_ids.empty?
        strict = projection_option(options, :strict, default: false)
        return project_json(item_ids, strict: strict)
      end

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

    def self.load_json(*args, item_ids_arg: nil, **options)
      require "json"

      json = args.shift
      json = json.to_str if !json.is_a?(String) && json.respond_to?(:to_str)
      obj = json.is_a?(String) ? JSON.parse(json) : json
      recommender = new
      recommender.send(:json_load, obj)

      item_ids_arg ||= args.shift unless args.empty?
      item_ids = recommender.send(:projection_ids, item_ids_arg, options)
      item_ids = Array(item_ids) + args.flatten unless args.empty?
      if recommender.send(:projection_requested?, item_ids_arg, options, item_ids) || !args.empty?
        recommender = recommender.project(item_ids, **options)
      end

      recommender
    end
    singleton_class.send(:alias_method, :from_json, :load_json)

    def self.project(*arguments, **options)
      ids = options.delete(:item_ids) || options.delete(:item_ids_arg)
      source = options.delete(:snapshot) || options.delete(:json) || options.delete(:source)

      arguments.each do |argument|
        if argument.is_a?(Hash) && projection_option_hash?(argument)
          options.merge!(argument.transform_keys(&:to_sym))
          ids ||= options.delete(:item_ids) || options.delete(:item_ids_arg)
          source ||= options.delete(:snapshot) || options.delete(:json) || options.delete(:source)
        elsif source.nil? && snapshot_source?(argument)
          source = argument
        else
          ids = Array(ids) + Array(argument)
        end
      end

      recommender =
        case source
        when Recommender
          source
        when Hash, String
          load_json(source)
        when nil
          raise ArgumentError, "Missing snapshot source"
        else
          if source.respond_to?(:read)
            load_json(source.read)
          elsif source.respond_to?(:to_str)
            load_json(source.to_str)
          else
            raise ArgumentError, "Invalid snapshot source: #{source.class}"
          end
        end

      recommender.project(ids, **options)
    end

    def self.project_json(*arguments, **options)
      project(*arguments, **options.merge(as: :json))
    end
    singleton_class.send(:alias_method, :project_snapshot, :project_json)
    singleton_class.send(:alias_method, :export, :project_json)
    singleton_class.send(:alias_method, :export_item_ids, :project_json)
    singleton_class.send(:alias_method, :release, :project_json)
    singleton_class.send(:alias_method, :release_snapshot, :project_json)

    def self.snapshot_source?(value)
      case value
      when Recommender
        true
      when Hash
        snapshot_hash?(value)
      when String
        value.lstrip.start_with?("{")
      else
        value.respond_to?(:read)
      end
    end

    def self.snapshot_hash?(value)
      value.key?("item_factors") || value.key?(:item_factors) ||
        value.key?("user_factors") || value.key?(:user_factors)
    end

    def self.projection_option_hash?(value)
      return false if snapshot_hash?(value)

      keys = value.keys.map { |key| key.respond_to?(:to_sym) ? key.to_sym : key }
      option_keys = PROJECTION_ID_OPTIONS + %i[strict as format snapshot json source]
      keys.intersect?(option_keys)
    end

    private_class_method :snapshot_source?, :snapshot_hash?, :projection_option_hash?

    private

    PROJECTION_ID_OPTIONS = %i[
      item_ids external_ids external_item_ids ids items only only_item_ids
      catalog catalog_item_ids item_id_subset keep_item_ids retain_item_ids item_id
    ].freeze
    PROJECTION_ID_OPTION_KEYS = PROJECTION_ID_OPTIONS + PROJECTION_ID_OPTIONS.map(&:to_s)

    def projection_ids(first, options)
      value =
        if projection_selection?(first, {})
          first
        else
          PROJECTION_ID_OPTION_KEYS.find { |key| options.key?(key) }.then { |key| options[key] if key }
        end

      case value
      when nil
        nil
      when String
        [value]
      when Array
        value
      when Set
        value.to_a
      when Hash
        projection_ids(nil, value.transform_keys(&:to_sym))
      else
        if value.respond_to?(:to_ary)
          value.to_ary
        elsif value.is_a?(Enumerable)
          value.to_a
        else
          [value]
        end
      end
    end

    def projection_requested?(first, options, ids)
      return false if ids.nil?

      # JSON's optional State argument is positional but is not a catalog.
      projection_selection?(first, options) ||
        PROJECTION_ID_OPTION_KEYS.any? { |key| options.key?(key) }
    end

    def projection_selection?(value, options = nil)
      return true if value.is_a?(Array) || value.is_a?(Set)
      return true if value.is_a?(String) || value.is_a?(Numeric) || value.is_a?(Symbol)
      return true if value.is_a?(Hash) && projection_ids(nil, value.transform_keys(&:to_sym))
      return true if value.is_a?(Range)

      value.respond_to?(:to_ary)
    end

    def selected_item_positions(ids, strict:)
      requested = Array(ids)
      selected_ids = []
      positions = []
      unknown = []

      requested.each do |external_id|
        if @item_map.key?(external_id)
          next if selected_ids.include?(external_id)

          selected_ids << external_id
          positions << @item_map[external_id]
        else
          unknown << external_id
        end
      end

      if strict && unknown.any?
        label = unknown.size == 1 ? "Unknown item id" : "Unknown item ids"
        raise KeyError, "#{label}: #{unknown.inspect}"
      end

      [selected_ids, positions]
    end

    def projection_option(options, *names, default:)
      names.each do |name|
        return options[name] if options.key?(name)
        return options[name.to_s] if options.key?(name.to_s)
      end
      default
    end

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
