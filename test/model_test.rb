require_relative "test_helper"

class ModelTest < Minitest::Test
  def test_example
    data = [
      {bedrooms: 1, bathrooms: 1, price: 100000},
      {bedrooms: 2, bathrooms: 1, price: 125000},
      {bedrooms: 2, bathrooms: 2, price: 135000},
      {bedrooms: 3, bathrooms: 2, price: 162000}
    ]
    model = Eps::Model.new(data, target: :price)
    assert_in_delta 125000, model.predict(bedrooms: 2, bathrooms: 1)
  end

  def test_untrained
    model = Eps::Model.new

    # TODO better error
    assert_raises(NoMethodError) do
      model.predict([{x: 1}])
    end
  end

  def test_bad_pmml
    assert_raises(Nokogiri::XML::SyntaxError) do
      Eps::Model.load_pmml("bad")
    end
  end

  def test_unknown_target
    data = [1, 2, 3, 4, 5].map { |xi| {x: xi, y: 3 + xi * 5 + rand} }
    error = assert_raises do
      Eps::Model.new(data, target: :unknown)
    end
    assert_equal "Missing column: unknown", error.message
  end

  def test_boolean_target
    data = [
      {x: "Sunday", y: true},
      {x: "Sunday", y: true},
      {x: "Monday", y: false},
      {x: "Monday", y: false}
    ]

    # treat boolean values as strings
    model = Eps::Model.new(data, target: :y)
    assert_equal "false", model.predict(x: "Monday")

    model = Eps::Model.load_pmml(model.to_pmml)
    assert_equal "false", model.predict(x: "Monday")
  end

  def test_split
    Eps::Model.new(mpg_data, target: :hwy, split: true)
  end

  def test_split_symbol
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    Eps::Model.new(data, target: :hwy, split: :listed_at)
  end

  def test_split_shuffle_false
    data = mpg_data
    Eps::Model.new(data, target: :hwy, split: {shuffle: false})
  end

  def test_split_column
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    Eps::Model.new(data, target: :hwy, split: {column: :listed_at})
  end

  def test_split_column_value
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    Eps::Model.new(data, target: :hwy, split: {column: :listed_at, value: Date.today - 5})
  end

  def test_split_no_training_data
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    error = assert_raises do
      Eps::Model.new(data, target: :hwy, split: {column: :listed_at, value: Date.today - 20})
    end
    assert_equal "No data in training set", error.message
  end

  def test_split_no_validation_data
    data = mpg_data.map { |r| r.merge(listed_at: Date.today - rand(10)) }
    error = assert_raises do
      Eps::Model.new(data, target: :hwy, split: {column: :listed_at, value: Date.today + 20})
    end
    assert_equal "No data in validation set", error.message
  end

  def test_split_bad_validation_type
    data = 4.times.map { |i| {x: i + 1, y: 3 + (i + 1) * 5} }
    data += 2.times.map { |i| {x: "bad", y: 10 + i} }

    error = assert_raises(ArgumentError) do
      Eps::LinearRegression.new(data, target: :y, split: {shuffle: false, validation_size: 1.0 / 3})
    end
    assert_equal "Bad type for column x: Expected numeric but got categorical", error.message
  end

  def test_split_column_bad_validation_type
    data = 4.times.map { |i| {x: i + 1, y: 3 + (i + 1) * 5, listed_at: Date.new(2020, 1, i + 1)} }
    data += 2.times.map { |i| {x: "bad", y: 10 + i, listed_at: Date.new(2020, 2, i + 1)} }

    error = assert_raises(ArgumentError) do
      Eps::LinearRegression.new(data, target: :y, split: {column: :listed_at, value: Date.new(2020, 2, 1)})
    end
    assert_equal "Bad type for column x: Expected numeric but got categorical", error.message
  end

  def test_validation_set_bad_type
    data = 4.times.map { |i| {x: i + 1, y: 3 + (i + 1) * 5} }
    validation_set = 2.times.map { |i| {x: "bad", y: 10 + i} }

    error = assert_raises(ArgumentError) do
      Eps::LinearRegression.new(data, target: :y, validation_set: validation_set)
    end
    assert_equal "Bad type for column x: Expected numeric but got categorical", error.message
  end

  def test_valid_validation_types
    train = 4.times.map { |i| {x: i + 1, y: 3 + (i + 1) * 5} }
    validation = 2.times.map { |i| {x: i + 5, y: 13 + i * 2} }

    # position split
    Eps::LinearRegression.new(train + validation, target: :y, split: {shuffle: false, validation_size: 1.0 / 3})

    # time split
    data = 4.times.map { |i| {x: i + 1, y: 3 + (i + 1) * 5, listed_at: Date.new(2020, 1, i + 1)} }
    data += 2.times.map { |i| {x: i + 5, y: 13 + i * 2, listed_at: Date.new(2020, 2, i + 1)} }
    Eps::LinearRegression.new(data, target: :y, split: {column: :listed_at})

    # explicit validation set
    Eps::LinearRegression.new(train, target: :y, validation_set: validation)
  end

  def test_split_column_not_feature
    data = 8.times.map do |i|
      {x: i + 1, target_col: 3 + (i + 1) * 5, weight_col: 1, listed_at: Date.new(2020, 1, i + 1)}
    end

    model = Eps::LinearRegression.new(data, target: :target_col, weight: :weight_col, split: {column: :listed_at})
    assert_equal [:_intercept, :x], model.coefficients.keys

    data = 8.times.map { |i| {x: i + 1, target_col: 3 + (i + 1) * 5, weight_col: 1} }
    model = Eps::LinearRegression.new(data, target: :target_col, weight: :weight_col, split: false)
    assert_equal [:_intercept, :x], model.coefficients.keys
  end

  def test_auto_text_features_from_training_set
    # text-like values are only in the validation set,
    # so automatic text features should not include the column
    data = [
      {x: "alpha beta", y: 3},
      {x: "alpha beta", y: 3},
      {x: "gamma delta", y: 5},
      {x: "gamma delta", y: 5}
    ]
    data += 6.times.map { |i| {x: "one two three four #{i}", y: 7 + i} }

    model = Eps::LinearRegression.new(data, target: :y, split: {shuffle: false, validation_size: 0.6})
    assert_equal [:_intercept, :"xgamma delta"], model.coefficients.keys
  end

  def test_regression_comparison
    _lr = Eps::LinearRegression.new(mpg_data, target: :hwy)
    _lgb = Eps::LightGBM.new(mpg_data, target: :hwy)
    # puts lr.summary
    # puts lgb.summary
  end

  def test_classification_comparison
    _nb = Eps::NaiveBayes.new(mpg_data, target: :drv)
    _lgb = Eps::LightGBM.new(mpg_data, target: :drv)
    # puts nb.summary
    # puts lgb.summary
  end
end
