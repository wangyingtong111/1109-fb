require_relative "test_helper"

class NaiveBayesTest < Minitest::Test
  def test_mpg
    data = mpg_data
    data = data.map { |r| r.slice(:displ, :year, :cyl, :class, :drv) }
    model = Eps::NaiveBayes.new(data, target: :drv, split: false)
    assert model.summary

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_weight
    data = [
      {x: 1.0, c: "a", y: "A", w: 1},
      {x: 2.0, c: "b", y: "A", w: 3},
      {x: 3.0, c: "a", y: "B", w: 1},
      {x: 4.0, c: "b", y: "B", w: 2}
    ]

    model = Eps::NaiveBayes.new(data, target: :y, weight: :w, split: false)
    assert_equal({"A" => 4, "B" => 3}, model.probabilities[:prior])

    numeric = model.probabilities[:conditional]["x"]
    assert_in_delta 1.75, numeric["A"][:mean]
    assert_in_delta 0.5, numeric["A"][:stdev]
    assert_in_delta 11.0 / 3, numeric["B"][:mean]
    assert_in_delta Math.sqrt(1.0 / 3), numeric["B"][:stdev]

    counts = Eps::NaiveBayes.new(data, target: :y, weight: :w, smoothing: false, split: false)
      .probabilities[:conditional]["c"]
    assert_equal({"A" => 1, "B" => 1}, counts["a"])
    assert_equal({"A" => 3, "B" => 2}, counts["b"])

    query = [{x: 1.5, c: "b"}, {x: 3.5, c: "a"}]
    expanded = data.flat_map do |row|
      Array.new(row[:w]) { row.reject { |k, _| k == :w } }
    end
    expanded_model = Eps::NaiveBayes.new(expanded, target: :y, split: false)

    assert_equal expanded_model.predict(query), model.predict(query)
    model.predict_probability(query).zip(expanded_model.predict_probability(query)) do |actual, expected|
      expected.each { |label, probability| assert_in_delta probability, actual[label] }
    end

    pmml = model.to_pmml
    assert_valid_pmml(pmml)
    loaded = Eps::Model.load_pmml(pmml)
    assert_equal model.predict(query), loaded.predict(query)
    model.predict_probability(query).zip(loaded.predict_probability(query)) do |actual, expected|
      expected.each { |label, probability| assert_in_delta probability, actual[label] }
    end
  end

  def test_fractional_weight
    data = [
      {x: 1.0, c: "a", y: "A", w: 0.5},
      {x: 2.0, c: "b", y: "A", w: 1.5},
      {x: 3.0, c: "a", y: "B", w: 2.25},
      {x: 4.0, c: "b", y: "B", w: 0.75}
    ]

    model = Eps::NaiveBayes.new(data, target: :y, weight: :w, smoothing: false, split: false)
    assert_equal({"A" => 2.0, "B" => 3.0}, model.probabilities[:prior])

    numeric = model.probabilities[:conditional]["x"]
    assert_in_delta Math.sqrt(0.375), numeric["A"][:stdev]
    assert_in_delta Math.sqrt(0.28125), numeric["B"][:stdev]

    loaded = Eps::Model.load_pmml(model.to_pmml)
    query = {x: 2.5, c: "a"}
    assert_in_delta model.predict_probability(query)["A"], loaded.predict_probability(query)["A"]
  end

  def test_unit_and_zero_weights
    data = [
      {x: 1.0, c: "a", y: "A", w: 1},
      {x: 2.0, c: "b", y: "A", w: 3},
      {x: 3.0, c: "a", y: "B", w: 1},
      {x: 4.0, c: "b", y: "B", w: 2}
    ]

    unweighted = Eps::NaiveBayes.new(data, target: :y, split: false)
    unit_weighted = Eps::NaiveBayes.new(
      data, target: :y, weight: Array.new(data.size, 1), split: false
    )
    assert_equal unweighted.probabilities, unit_weighted.probabilities

    zero_data = data + [{x: 100.0, c: "zero-only", y: "A", w: 0}]
    zero_model = Eps::NaiveBayes.new(zero_data, target: :y, weight: :w, split: false)
    weighted = Eps::NaiveBayes.new(data, target: :y, weight: :w, split: false)
    assert_equal weighted.probabilities, zero_model.probabilities

    assert_raises ArgumentError do
      data.first[:w] = -1
      Eps::NaiveBayes.new(data, target: :y, weight: :w, split: false)
    end
  end

  def test_python_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/python/naive_bayes.pmml"))

    # different expected than Ruby and R
    expected = %w(f f r r f f f r f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_r_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/r/naive_bayes.pmml"))

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_bad_mpg
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/bad_naive_bayes.pmml"))

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_rover
    df = Rover.read_csv("test/support/mpg.csv")
    df = df[["displ", "year", "cyl", "class", "drv"]]
    df["drv"] = df["drv"].map(&:to_s)

    model = Eps::NaiveBayes.new(df, target: :drv, split: false)

    expected = %w(f f 4 r f f f 4 f f)
    predictions = model.predict(df.first(10))
    assert_equal expected, predictions
  end

  def test_unknown_categorical_value
    data = [
      {x: "Sunday", y: "red"},
      {x: "Sunday", y: "red"},
      {x: "Sunday", y: "red"},
      {x: "Monday", y: "blue"},
      {x: "Monday", y: "blue"}
    ]

    model = Eps::NaiveBayes.new(data, target: :y, split: false)
    assert_equal "red", model.predict(x: "Tuesday")
  end

  def test_text_features
    skip

    data = [
      {message: "This is the first document.", tag: "ham"},
      {message: "Hello, this document is the second document.", tag: "spam"},
      {message: "Hello, and this is the third one.", tag: "spam"},
      {message: "Is this the first document?", tag: "ham"}
    ]

    test_data = [
      {message: "what a cool first document"},
      {message: "hello document"}
    ]

    model = Eps::NaiveBayes.new(data, target: :tag, text_features: [:message])
    assert_equal ["ham", "spam"], model.predict(test_data)

    pmml = model.to_pmml
    assert_includes pmml, "NaiveBayesModel"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal ["ham", "spam"], model.predict(test_data)
  end

  def test_boolean
    data = [
      {x: false, y: "ham"},
      {x: false, y: "ham"},
      {x: true, y: "spam"},
      {x: true, y: "spam"}
    ]

    model = Eps::NaiveBayes.new(data, target: :y)
    assert_equal ["ham", "spam"], model.predict([{x: false}, {x: true}])

    pmml = model.to_pmml
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal ["ham", "spam"], model.predict([{x: false}, {x: true}])
  end

  def test_probabilities
    data = mpg_data
    data = data.map { |r| r.slice(:displ, :year, :cyl, :class, :drv) }
    model = Eps::NaiveBayes.new(data, target: :drv, split: false)
    assert model.summary

    predictions = model.predict_probability(data.first)
    assert_in_delta 0.007528105, predictions["4"]
    assert_in_delta 9.924719e-01, predictions["f"]
    assert_in_delta 6.506897e-10, predictions["r"]

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict_probability(data.first)
    assert_in_delta 0.007528105, predictions["4"]
    assert_in_delta 9.924719e-01, predictions["f"]
    assert_in_delta 6.506897e-10, predictions["r"]
  end
end
