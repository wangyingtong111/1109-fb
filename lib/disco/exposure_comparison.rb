module Disco
  # Offline exposure comparison of two *already fitted* recommenders.
  #
  # Both recommenders are read through their public, read-only API
  # (+user_ids+, +item_ids+ and +user_recs+) and aligned by external
  # +user_id+/+item_id+ values, so internal index differences between the
  # two models never affect the comparison.
  class ExposureComparison
    # Outcome for a single user on one side of the comparison.
    UserResult = Struct.new(:user_id, :status, :top_item_ids, :excluded_item_ids, keyword_init: true) do
      def missing?
        status == :missing_user
      end
    end

    # Aggregated exposure of a single item on one side of the comparison.
    ItemResult = Struct.new(
      :item_id, :exposure, :ranked_user_ids, :missing_catalog_user_ids,
      :missing_user_ids, :unranked_user_ids, :excluded_user_ids,
      keyword_init: true
    ) do
      # A zero exposure can have several distinct causes. Keep them
      # separately auditable instead of collapsing everything into 0.
      def missing_catalog?
        !missing_catalog_user_ids.empty?
      end

      def missing_user?
        !missing_user_ids.empty?
      end

      def unranked?
        !unranked_user_ids.empty?
      end

      def excluded?
        !excluded_user_ids.empty?
      end
    end

    # Everything observed for one recommender.
    SideResult = Struct.new(:name, :recommender, :users, :items, :exposure_total, keyword_init: true) do
      def item(item_id)
        items[item_id]
      end

      def user(user_id)
        users[user_id]
      end
    end

    # Full comparison result.
    Result = Struct.new(:user_ids, :item_ids, :k, :sides, :seen, keyword_init: true) do
      def side(name)
        sides.fetch(name)
      end

      # Exposure counts keyed by external item id, e.g. {"rice" => 1}.
      def exposure_counts(name)
        side(name).items.transform_values(&:exposure)
      end

      # Per-item side-by-side exposure counts and the exposure delta.
      def item_diffs
        item_ids.map do |item_id|
          a = sides[:a].items[item_id]
          b = sides[:b].items[item_id]
          {
            item_id: item_id,
            a_exposure: a.exposure,
            b_exposure: b.exposure,
            delta: a.exposure - b.exposure
          }
        end
      end
    end

    # Compare two fitted recommenders.
    #
    #   Disco::ExposureComparison.compare(
    #     model_a, model_b,
    #     user_ids: ["u"],
    #     item_ids: ["rice", "tofu", "corn"],
    #     seen: {"u" => ["corn"]},
    #     k: 1
    #   )
    #
    # +seen+ maps an external user id to the external item ids that user has
    # already consumed. It is applied identically to both sides and may be
    # omitted (or empty) when there is nothing to exclude.
    def self.compare(recommender_a, recommender_b, user_ids:, item_ids:, k: 5, seen: nil)
      new(recommender_a, recommender_b).compare(
        user_ids: user_ids, item_ids: item_ids, k: k, seen: seen
      )
    end

    def initialize(recommender_a, recommender_b)
      @recommenders = {a: recommender_a, b: recommender_b}
    end

    def compare(user_ids:, item_ids:, k: 5, seen: nil)
      raise ArgumentError, "k must be positive" unless k.is_a?(Integer) && k > 0

      # Same batch of users and de-duplicated candidates for both sides,
      # keyed by external id. Order is preserved for stable output.
      user_ids = dedup(user_ids)
      item_ids = dedup(item_ids)
      raise ArgumentError, "No users" if user_ids.empty?
      raise ArgumentError, "No items" if item_ids.empty?

      seen = normalize_seen(seen, user_ids)

      sides = {}
      @recommenders.each do |name, recommender|
        sides[name] = build_side(name, recommender, user_ids, item_ids, k, seen)
      end

      Result.new(user_ids: user_ids, item_ids: item_ids, k: k, sides: sides, seen: seen)
    end

    private

    def build_side(name, recommender, user_ids, item_ids, k, seen)
      known_users = recommender.user_ids.to_set
      catalog = recommender.item_ids.to_set

      users = {}
      items = {}
      item_ids.each do |item_id|
        items[item_id] = ItemResult.new(
          item_id: item_id,
          exposure: 0,
          ranked_user_ids: [],
          # An item absent from this side's catalog cannot be recommended to
          # anyone on this side, regardless of user. When the item exists in
          # the catalog this stays empty and per-user reasons take over.
          missing_catalog_user_ids: catalog.include?(item_id) ? [] : user_ids.dup,
          missing_user_ids: [],
          unranked_user_ids: [],
          excluded_user_ids: []
        )
      end

      exposure_total = 0

      user_ids.each do |user_id|
        # Do not substitute a cold-start/popularity ranking for a user this
        # model has never seen.
        unless known_users.include?(user_id)
          users[user_id] = UserResult.new(
            user_id: user_id, status: :missing_user,
            top_item_ids: [], excluded_item_ids: []
          )
          item_ids.each { |item_id| items[item_id].missing_user_ids << user_id }
          next
        end

        excluded = (seen[user_id] & item_ids)
        candidate_ids = item_ids.reject { |item_id| excluded.include?(item_id) }

        # Only items this model actually knows can be ranked.
        rankable_ids = candidate_ids.select { |item_id| catalog.include?(item_id) }

        if rankable_ids.empty?
          ranked = []
        else
          # Read-only call against the fitted model. Internal item indices
          # are resolved inside user_recs; we only handle external ids.
          ranked = recommender.user_recs(
            user_id, count: k, item_ids: rankable_ids
          ).map { |rec| rec[:item_id] }
        end
        ranked_set = ranked.to_set

        candidate_ids.each do |item_id|
          # Rankable candidates that did not make the top k are "unranked".
          # Explicitly seen items and catalog gaps are tracked separately and
          # must not be mixed into this bucket.
          next unless catalog.include?(item_id)
          next if excluded.include?(item_id)

          unless ranked_set.include?(item_id)
            items[item_id].unranked_user_ids << user_id
          end
        end

        ranked.each do |item_id|
          item = items[item_id]
          item.exposure += 1
          item.ranked_user_ids << user_id
          exposure_total += 1
        end

        excluded.each do |item_id|
          items[item_id].excluded_user_ids << user_id
        end

        users[user_id] = UserResult.new(
          user_id: user_id,
          status: :ok,
          top_item_ids: ranked,
          excluded_item_ids: excluded
        )
      end

      SideResult.new(
        name: name,
        recommender: recommender,
        users: users,
        items: items,
        exposure_total: exposure_total
      )
    end

    def normalize_seen(seen, user_ids)
      seen ||= {}
      raise ArgumentError, "seen must be a hash" unless seen.is_a?(Hash)

      result = {}
      user_ids.each do |user_id|
        values = seen[user_id] || seen[user_id.to_s] || []
        raise ArgumentError, "seen values must be enumerable" unless values.respond_to?(:to_a)

        result[user_id] = dedup(values.to_a)
      end
      result
    end

    def dedup(values)
      values.uniq
    end
  end
end