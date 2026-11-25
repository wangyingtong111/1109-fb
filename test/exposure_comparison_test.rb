# Standalone test: only depends on minitest and disco itself, so it can run
# without the optional ActiveRecord/Rover/Faiss test dependencies.
require "minitest/autorun"
require_relative "../lib/disco"

class ExposureComparisonTest < Minitest::Test
  # Two fitted recommenders with deliberately different catalogs and internal
  # item-index order for the same external item ids.
  def build_models
    data_a = [
      {user_id: "u",  item_id: "rice", rating: 5},
      {user_id: "u",  item_id: "corn", rating: 5},
      {user_id: "ua", item_id: "tofu", rating: 1},
      {user_id: "ua", item_id: "rice", rating: 5}
    ]
    a = Disco::Recommender.new(factors: 8, epochs: 30, top_items: true)
    a.fit(data_a)

    # B has no "rice" and maps tofu/corn to different internal indices than A.
    data_b = [
      {user_id: "u",  item_id: "tofu", rating: 5},
      {user_id: "u",  item_id: "corn", rating: 1},
      {user_id: "ub", item_id: "tofu", rating: 5}
    ]
    b = Disco::Recommender.new(factors: 8, epochs: 30, top_items: true)
    b.fit(data_b)

    [a, b]
  end

  def compare
    a, b = build_models
    result = Disco::ExposureComparison.compare(
      a, b,
      user_ids: ["u", "ua"],
      item_ids: ["rice", "tofu", "corn"],
      seen: {"u" => ["corn"]},
      k: 1
    )
    [a, b, result]
  end

  def test_aligns_by_external_ids_despite_index_mapping
    a, b, result = compare
    # The two models fit their data in different orders, so the same external
    # item is not stored at the same internal index everywhere.
    assert_equal ["rice", "corn", "tofu"], a.item_ids
    assert_equal ["tofu", "corn"], b.item_ids
    # "corn" sits at index 1 in A but index 1 in B as well here; the point is
    # that results are joined on external id, so index equality is irrelevant.
    assert_equal "corn", a.item_ids[a.item_ids.index("corn")]
    assert_equal "corn", b.item_ids[b.item_ids.index("corn")]
    refute_equal a.item_ids, b.item_ids

    assert_equal ["u", "ua"], result.user_ids
    assert_equal ["rice", "tofu", "corn"], result.item_ids
    assert_equal 1, result.k
  end

  def test_top_k_and_exposure_counts
    _, _, result = compare

    # A's first rec for u is rice; B's first rec for u is tofu.
    assert_equal ["rice"], result.side(:a).user("u").top_item_ids
    assert_equal ["tofu"], result.side(:b).user("u").top_item_ids

    # rice goes 1 -> 0 across A/B, tofu goes 0 -> 1.
    assert_equal ({"rice" => 2, "tofu" => 0, "corn" => 0}), result.exposure_counts(:a)
    assert_equal ({"rice" => 0, "tofu" => 1, "corn" => 0}), result.exposure_counts(:b)

    diffs = result.item_diffs.to_h { |d| [d[:item_id], d] }
    assert_equal 2, diffs["rice"][:delta]
    assert_equal(-1, diffs["tofu"][:delta])
  end

  def test_missing_catalog_item_is_distinct_from_other_zero_causes
    _, b, result = compare
    refute_includes b.item_ids, "rice"

    rice_b = result.side(:b).item("rice")
    assert_equal 0, rice_b.exposure
    assert rice_b.missing_catalog?
    # absent from the whole catalog: no user on this side could rank it
    assert_equal ["u", "ua"], rice_b.missing_catalog_user_ids
    # the catalog reason must not be confused with the per-user reasons
    assert_empty rice_b.ranked_user_ids
    assert_empty rice_b.unranked_user_ids
    assert_empty rice_b.excluded_user_ids

    # rice is fine in A's catalog and actually ranked there
    rice_a = result.side(:a).item("rice")
    refute rice_a.missing_catalog?
    assert_equal 2, rice_a.exposure
  end

  def test_missing_user_never_uses_cold_start_ranking
    _, b, result = compare
    # B has top_items enabled, so user_recs would otherwise return a
    # cold-start/popularity list for the unknown user "ua".
    assert b.top_items
    refute_includes b.user_ids, "ua"

    user_b = result.side(:b).user("ua")
    assert user_b.missing?
    assert_equal :missing_user, user_b.status
    assert_empty user_b.top_item_ids
    refute_equal b.user_recs("ua", count: 1), [] # cold-start would exist

    # every candidate item records the missing user, with zero exposure
    result.side(:b).items.each_value do |item|
      assert_includes item.missing_user_ids, "ua"
      assert_equal 0, item.exposure if item.item_id == "rice"
    end

    # the user that B does know still gets a real personalized ranking
    refute result.side(:b).user("u").missing?
  end

  def test_unranked_candidate_is_distinct_even_with_zero_exposure
    _, _, result = compare

    # corn exists in both catalogs but never makes the k=1 list
    %i[a b].each do |side|
      corn = result.side(side).item("corn")
      assert_equal "corn", corn.item_id
      assert_equal 0, corn.exposure
      refute corn.missing_catalog?
    end
    # A knows both users, so nobody is missing there
    refute result.side(:a).item("corn").missing_user?

    # on A, only ua could have ranked corn; u had it explicitly excluded
    corn_a = result.side(:a).item("corn")
    assert corn_a.unranked?
    assert_equal ["ua"], corn_a.unranked_user_ids

    # on B, u excluded corn and ua is missing, so there is nobody left who
    # could have ranked it: unranked stays empty but the other buckets fill
    corn_b = result.side(:b).item("corn")
    refute corn_b.unranked?
    assert_empty corn_b.unranked_user_ids
    assert_equal ["u"], corn_b.excluded_user_ids
    assert_equal ["ua"], corn_b.missing_user_ids
  end

  def test_unranked_candidate_on_side_where_user_is_known
    # Build a clean unranked case: both models know both users and all items,
    # but "corn" never makes the k=1 list for an eligible, non-excluded user.
    data = [
      {user_id: "u",  item_id: "rice", rating: 5},
      {user_id: "u",  item_id: "corn", rating: 1},
      {user_id: "ua", item_id: "rice", rating: 5},
      {user_id: "ua", item_id: "corn", rating: 1}
    ]
    a = Disco::Recommender.new(factors: 8, epochs: 30); a.fit(data)
    b = Disco::Recommender.new(factors: 8, epochs: 30); b.fit(data)

    result = Disco::ExposureComparison.compare(
      a, b, user_ids: ["u", "ua"], item_ids: ["rice", "corn"], k: 1
    )
    corn = result.side(:a).item("corn")
    assert_equal 0, corn.exposure
    assert corn.unranked?
    refute corn.missing_catalog?
    refute corn.missing_user?
    refute corn.excluded?
    assert_equal ["u", "ua"], corn.unranked_user_ids
  end

  def test_explicit_seen_exclusion_applied_identically_to_both_sides
    _, _, result = compare

    %i[a b].each do |side|
      corn = result.side(side).item("corn")
      assert corn.excluded?
      assert_equal ["u"], corn.excluded_user_ids
      assert_equal ["corn"], result.side(side).user("u").excluded_item_ids
    end

    # excluded item never reaches the model's ranking call
    refute_includes result.side(:a).user("u").top_item_ids, "corn"
    refute_includes result.side(:b).user("u").top_item_ids, "corn"
  end

  def test_three_zero_exposure_causes_are_separately_verifiable
    _, _, result = compare

    # Side A alone exhibits three distinct zero-exposure reasons.
    a = result.side(:a)

    # 1) missing catalog: rice is absent from B's whole catalog. For the user
    #    B does know ("u") that is the operative reason; "ua" is additionally a
    #    missing user, and the two causes stay in separate buckets.
    rice_b = result.side(:b).item("rice")
    assert_equal 0, rice_b.exposure
    assert rice_b.missing_catalog?
    assert_equal ["u", "ua"], rice_b.missing_catalog_user_ids
    refute rice_b.unranked?
    refute rice_b.excluded?
    assert_equal ["ua"], rice_b.missing_user_ids
    assert_empty rice_b.ranked_user_ids

    # 2) missing user: "ua" is unknown to B; every B item records it
    a.items.each_value { |item| refute item.missing_user? }
    result.side(:b).items.each_value { |item| assert item.missing_user? }

    # 3) unranked: tofu exists in A's catalog, users are known, but miss k=1
    tofu_a = a.item("tofu")
    assert_equal 0, tofu_a.exposure
    assert tofu_a.unranked?
    refute tofu_a.missing_catalog?
    refute tofu_a.missing_user?
    refute tofu_a.excluded?
    assert_equal ["u", "ua"], tofu_a.unranked_user_ids

    # and a fourth, explicit seen exclusion, stays its own bucket too.
    # u excluded corn, while ua merely failed to rank it: both zero-exposure
    # reasons coexist on the same item but are recorded separately.
    corn_a = a.item("corn")
    assert_equal 0, corn_a.exposure
    assert corn_a.excluded?
    assert corn_a.unranked?
    assert_equal ["u"], corn_a.excluded_user_ids
    assert_equal ["ua"], corn_a.unranked_user_ids
  end

  def test_dedupes_users_and_candidates
    a, b = build_models
    result = Disco::ExposureComparison.compare(
      a, b,
      user_ids: ["u", "u", "ua"],
      item_ids: ["rice", "rice", "tofu", "corn"],
      k: 2
    )
    assert_equal ["u", "ua"], result.user_ids
    assert_equal ["rice", "tofu", "corn"], result.item_ids
    # each user contributes at most one exposure per item
    result.sides.each_value do |side|
      side.items.each_value do |item|
        assert_equal item.ranked_user_ids, item.ranked_user_ids.uniq
      end
    end
  end

  def test_only_reads_fitted_models
    a, b = build_models
    snapshot_a = [a.user_ids, a.item_ids, a.item_factors.to_binary]
    snapshot_b = [b.user_ids, b.item_ids, b.item_factors.to_binary]

    Disco::ExposureComparison.compare(
      a, b, user_ids: ["u"], item_ids: a.item_ids | b.item_ids, k: 2
    )

    assert_equal snapshot_a, [a.user_ids, a.item_ids, a.item_factors.to_binary]
    assert_equal snapshot_b, [b.user_ids, b.item_ids, b.item_factors.to_binary]
  end

  def test_k_larger_than_candidates
    a, b = build_models
    result = Disco::ExposureComparison.compare(
      a, b, user_ids: ["u"], item_ids: ["rice", "tofu", "corn"], k: 10
    )
    # u has not rated tofu, so A can return at most the unrated candidates
    assert_includes result.side(:a).user("u").top_item_ids, "tofu"
  end

  def test_raises_on_invalid_arguments
    a, b = build_models
    assert_raises(ArgumentError) do
      Disco::ExposureComparison.compare(a, b, user_ids: ["u"], item_ids: ["rice"], k: 0)
    end
    assert_raises(ArgumentError) do
      Disco::ExposureComparison.compare(a, b, user_ids: [], item_ids: ["rice"])
    end
    assert_raises(ArgumentError) do
      Disco::ExposureComparison.compare(a, b, user_ids: ["u"], item_ids: [])
    end
    assert_raises(ArgumentError) do
      Disco::ExposureComparison.compare(
        a, b, user_ids: ["u"], item_ids: ["rice"], seen: ["rice"]
      )
    end
  end

  # Deterministic stub recommender exercising the classification logic
  # directly, including catalog gaps and a user absent from one side, while
  # proving nothing depends on internal index values.
  class StubRecommender
    def initialize(user_ids:, item_ids:, ranking:, cold_start: nil)
      @user_ids = user_ids
      @item_ids = item_ids
      @ranking = ranking
      @cold_start = cold_start
    end

    attr_reader :user_ids, :item_ids
    attr_reader :cold_start

    def top_items
      @cold_start
    end

    # emulate the real signature, including the item_ids filter
    def user_recs(user_id, count: 5, item_ids: nil)
      raise "Not fit" unless defined?(@ranking)
      unless @user_ids.include?(user_id)
        # mimic Disco with top_items: true (cold-start fallback)
        return @cold_start ? @cold_start.first(count) : []
      end

      ranked = @ranking.fetch(user_id, []).select { |id| item_ids.include?(id) }
      ranked.first(count).map { |id| {item_id: id, score: 1.0} }
    end
  end

  def test_stub_full_classification_and_cold_start_guard
    # internal order here is intentionally shuffled vs external candidate order
    left = StubRecommender.new(
      user_ids: ["u"],
      item_ids: ["tofu", "rice"],
      ranking: {"u" => ["rice", "tofu"]},
      cold_start: [{item_id: "tofu", score: 9.0}]
    )
    right = StubRecommender.new(
      user_ids: [],
      item_ids: ["corn", "tofu"],
      ranking: {},
      cold_start: [{item_id: "tofu", score: 9.0}]
    )

    result = Disco::ExposureComparison.compare(
      left, right,
      user_ids: ["u"],
      item_ids: ["rice", "tofu", "corn"],
      seen: {"u" => ["tofu"]},
      k: 1
    )

    # left: rice ranked, tofu explicitly seen/excluded, corn missing catalog
    assert_equal 1, result.side(:a).item("rice").exposure
    assert result.side(:a).item("corn").missing_catalog?
    assert result.side(:a).item("tofu").excluded?

    # right: user is missing and must not receive the cold-start ranking
    assert result.side(:b).user("u").missing?
    assert_empty result.side(:b).user("u").top_item_ids
    result.side(:b).items.each_value { |item| assert_includes item.missing_user_ids, "u" }

    # rice missing from right catalog; corn present but unranked for left-less right
    assert result.side(:b).item("rice").missing_catalog?
    assert result.side(:b).item("corn").missing_user?
  end

  def test_implicit_models_are_supported
    data = [
      {user_id: 1, item_id: "x"},
      {user_id: 1, item_id: "y"},
      {user_id: 2, item_id: "x"}
    ]
    a = Disco::Recommender.new(factors: 4, epochs: 10)
    a.fit(data)
    b = Disco::Recommender.new(factors: 4, epochs: 10)
    b.fit(data)

    result = Disco::ExposureComparison.compare(
      a, b, user_ids: [1, 2], item_ids: ["x", "y"], k: 1
    )
    assert_equal result.exposure_counts(:a), result.exposure_counts(:b)
    assert result.side(:a).exposure_total.positive?
  end
end