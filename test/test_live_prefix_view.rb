# frozen_string_literal: true

require "test_helper"

class TestLivePrefixView < Minitest::Test
  Map = MiniFTS::SearchableMap

  def test_partial_edge_view_write_is_visible_to_root_and_view
    root = Map.from([["stone", 1], ["story", 2]])
    view = root.at_prefix("sto", live: true)

    view.set("rk", 3)

    assert_equal 3, root.get("stork")
    assert_equal 3, view.get("rk")
    assert_equal true, root.has?("stork")
    assert_equal true, view.has?("rk")
  end

  def test_partial_edge_write_and_delete_stay_bound_to_root
    root = Map.from([["stone", 1], ["story", 2]])
    st = root.at_prefix("st")

    st.set("em", 3)
    st.delete("one")

    assert_equal 3, root.get("stem")
    assert_nil root.get("stone")
    assert_equal 2, root.get("story")
    assert_equal 3, st.get("em")
  end

  def test_held_views_survive_pruning_compression_and_root_reinsertion
    root = Map.from([["stone", 1], ["story", 2], ["tale", 3]])
    st = root.at_prefix("st")
    sto = root.at_prefix("sto", live: true)

    root.delete("story")
    root.delete("stone")

    st.set("em", 4)
    sto.set("ry", 5)
    st.delete("ory")
    root.set("stop", 6)

    assert_equal %w[stem stop tale], root.keys.sort
    assert_equal 6, st.get("op")
    assert_equal 6, sto.get("p")
    assert_equal 4, st.get("em")
    assert_nil sto.get("ry")
    assert_equal %w[stem stop], st.keys.sort
    assert_equal %w[stop], sto.keys.sort
    assert_equal 2, st.size
    assert_equal 1, sto.size
  end

  def test_online_option_and_named_relative_view_are_supported
    root = Map.from([["stone", 1], ["story", 2]])

    online = root.at_prefix("sto", online: true)
    online.set("rk", 3)
    assert_equal 3, root.get("stork")
    assert_equal 3, online.get("rk")

    relative = root.prefix_view("sto")
    assert_equal %w[ne rk ry], relative.keys.sort
    assert_equal [["ne", 1], ["rk", 3], ["ry", 2]], relative.entries.sort

    relative.delete("rk")
    assert_nil root.get("stork")
    assert_nil relative.get("rk")
  end

  def test_relative_view_resolves_a_relative_child_prefix
    root = Map.from([["stone", 1], ["story", 2]])
    st = root.prefix_view("st")
    sto = st.relative_view("o")

    assert_equal %w[ne ry], sto.keys.sort
    sto.set("rk", 3)
    assert_equal 3, root.get("stork")
    assert_equal 3, sto.get("rk")
  end

  def test_parent_fuzzy_lookup_keeps_original_distances
    map = Map.from([["stone", 8], ["story", 9]])

    assert_equal [8, 1], map.fuzzy_get("ston", 1)["stone"]
  end
end
