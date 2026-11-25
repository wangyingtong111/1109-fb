require_relative "test_helper"

class LightGBMTest < Minitest::Test
  def test_regression
    data = mpg_data(extra_fields: [:model])
    model = Eps::LightGBM.new(data, target: :hwy, split: false, text_features: {model: {max_features: 5}})
    assert model.summary

    expected = [30.65282474, 33.95768195, 17.89527884, 16.93797369, 28.38019179, 29.40602519, 27.6259532, 18.81081728, 24.88466865, 29.40602519]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_regression_weight
    data = mpg_data
    model = Eps::LightGBM.new(data, target: :hwy, weight: :cyl, split: false)
    assert model.summary

    expected = [30.816408220564853, 33.67316784681556, 17.374629901348282, 17.1710193131715, 29.403161956316275, 29.17722520031403, 27.29606023681178, 18.389428934078527, 24.583836209258205, 29.17722520031403]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_regression_python_pmml
    data = mpg_data(extra_fields: [:model])
    model = Eps::Model.load_pmml(File.read("test/support/python/lightgbm_regression.pmml"))

    expected = [30.65282474, 33.95768195, 17.89527884, 16.93797369, 28.38019179, 29.40602519, 27.6259532, 18.81081728, 24.88466865, 29.40602519]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_binary
    data = mpg_data(binary: true)
    model = Eps::LightGBM.new(data, target: :drv, split: false)

    expected = %w(f f 4 4 f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_binary_probabilities
    data = mpg_data(binary: true)
    model = Eps::LightGBM.new(data, target: :drv, split: false)

    predictions = model.predict_probability(data.first)
    assert_in_delta 0, predictions["4"]
    assert_in_delta 1, predictions["f"]

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict_probability(data.first)
    assert_in_delta 0, predictions["4"]
    assert_in_delta 1, predictions["f"]
  end

  def test_binary_python_pmml
    data = mpg_data(binary: true)
    model = Eps::Model.load_pmml(File.read("test/support/python/lightgbm_binary.pmml"))

    expected = %w(f f 4 4 f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_multiclass
    data = mpg_data
    data.each { |r| r.delete(:hwy) }
    model = Eps::LightGBM.new(data, target: :drv, split: false)

    expected = %w(f f 4 r f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_multiclass_probabilities
    data = mpg_data
    data.each { |r| r.delete(:hwy) }
    model = Eps::LightGBM.new(data, target: :drv, split: false)

    predictions = model.predict_probability(data.first)
    assert_in_delta 1.19861155e-03, predictions["4"]
    assert_in_delta 9.98772665e-01, predictions["f"]
    assert_in_delta 2.87230554e-05, predictions["r"]

    model = Eps::Model.load_pmml(model.to_pmml)
    predictions = model.predict_probability(data.first)
    assert_in_delta 1.19861155e-03, predictions["4"]
    assert_in_delta 9.98772665e-01, predictions["f"]
    assert_in_delta 2.87230554e-05, predictions["r"]
  end

  def test_multiclass_python_pmml
    data = mpg_data
    model = Eps::Model.load_pmml(File.read("test/support/python/lightgbm_multiclass.pmml"))

    expected = %w(f f 4 r f f f 4 4 f)
    predictions = model.predict(data.first(10))
    assert_equal expected, predictions
  end

  def test_missing_column
    test_set = mpg_data(extra_fields: [:model]).first
    test_set.delete(:year)

    error = assert_raises(ArgumentError) do
      model.predict(test_set)
    end
    assert_equal "Missing column: year", error.message
  end

  def test_bad_type
    test_set = mpg_data(extra_fields: [:model]).first
    test_set[:year] = "bad"

    error = assert_raises(ArgumentError) do
      model.predict(test_set)
    end
    assert_equal "Bad type for column year: Expected numeric but got categorical", error.message
  end

  def test_unseen
    data = mpg_data
    train_set = data[0...150]
    validation_set = data[150..-1]
    validation_set.each_with_index do |r, i|
      r[:class] = "unseen" if i % 3 == 0
    end
    model = Eps::LightGBM.new(train_set, target: :hwy, weight: :cyl, validation_set: validation_set)
    assert model.summary
  end

  def test_boolean
    data = [
      {x: false, y: 3},
      {x: false, y: 3},
      {x: true, y: 5},
      {x: true, y: 5}
    ]

    model = Eps::LightGBM.new(data, target: :y, early_stopping: false)
    assert_elements_in_delta [3, 5], model.predict([{x: false}, {x: true}])

    pmml = model.to_pmml
    assert_includes pmml, "Segmentation"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: false}, {x: true}])
  end

  def test_text_features_regression
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the day", y: 3},
      {x: "Monday is the best", y: 5},
      {x: "Monday is the day", y: 5}
    ]

    model = Eps::LightGBM.new(data, target: :y, early_stopping: false)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best"}, {x: "Monday is the best"}])

    pmml = model.to_pmml
    assert_includes pmml, "Segmentation"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "Monday is the best!!"}])
  end

  def test_text_features_regression_case_sensitive
    data = [
      {x: "Sunday is the best", y: 3},
      {x: "Sunday is the day", y: 3},
      {x: "sunday is the best", y: 5},
      {x: "sunday is the day", y: 5}
    ]

    model = Eps::LightGBM.new(data, target: :y, early_stopping: false, text_features: {x: {case_sensitive: true}})
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best"}, {x: "sunday is the best"}])

    model = Eps::Model.load_pmml(model.to_pmml)
    assert_elements_in_delta [3, 5], model.predict([{x: "Sunday is the best!!"}, {x: "sunday is the best!!"}])
  end

  def test_text_features_classification
    data = [
      {message: "This is the first document.", tag: "ham"},
      {message: "Hello, this is the second document.", tag: "spam"},
      {message: "Hello, and this is the third one.", tag: "spam"},
      {message: "Is this the first document?", tag: "ham"}
    ]

    test_data = [
      {message: "what a cool first document"},
      {message: "hello document"}
    ]

    model = Eps::LightGBM.new(data, target: :tag)
    assert_equal ["ham", "spam"], model.predict(test_data)

    pmml = model.to_pmml
    assert_includes pmml, "Segmentation"
    assert_valid_pmml(pmml)

    model = Eps::Model.load_pmml(pmml)
    assert_equal ["ham", "spam"], model.predict(test_data)
  end

  def test_text_features_classification_categorical
    data = [
      {message: "This is the first document.", category: "A", tag: "ham"},
      {message: "Hello, this is the second document.", category: "B", tag: "spam"}
    ]
    Eps::LightGBM.new(data, target: :tag)
  end

  def test_learning_rate
    data = mpg_data
    model = Eps::LightGBM.new(data, target: :hwy, split: false, learning_rate: 1)
    assert model.summary

    expected = [30.80980036, 34.39919293, 17.99841545, 17.298401, 28.4685196, 29.31558087, 27.79557906, 18.44068633, 24.64178236, 29.31558087]
    predictions = model.predict(data.first(10))
    assert_elements_in_delta expected, predictions
  end

  def test_missing_default_child
    model = Eps::Model.load_pmml(File.read("test/support/default_child.pmml"))

    # root splits on x, missing x takes the declared right child; that child
    # splits on z, missing z takes the declared left child (score 12)
    rows = [
      {x: nil, z: nil}, # both missing -> right, then left -> 12
      {x: nil, z: 5},   # missing x -> right, z positive -> 13
      {x: 5, z: nil},   # x positive -> right, missing z -> left -> 12
      {x: 0, z: 5},     # satisfies x <= 0 -> left -> 10
      {x: 5, z: 5},     # satisfies both predicates -> 13
      {x: -1, z: 5}     # satisfies x <= 0 -> left -> 10
    ]

    assert_elements_in_delta [12, 13, 12, 10, 13, 10], model.predict(rows)
  end

  def test_missing_default_child_round_trip
    pmml = File.read("test/support/default_child.pmml")
    model = Eps::Model.load_pmml(pmml)

    rows = [
      {x: nil, z: nil},
      {x: nil, z: 5},
      {x: 5, z: nil},
      {x: 0, z: 5}
    ]
    expected = [12, 13, 12, 10]
    assert_elements_in_delta expected, model.predict(rows)

    exported = model.to_pmml
    assert_includes exported, "defaultChild"

    model = Eps::Model.load_pmml(exported)
    assert_elements_in_delta expected, model.predict(rows)
  end

  def test_missing_without_default_child
    pmml = <<~XML
      <?xml version="1.0" encoding="UTF-8"?>
      <PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">
        <Header><Application name="test" version="1.0"/></Header>
        <DataDictionary>
          <DataField name="y" optype="continuous" dataType="double"/>
          <DataField name="x" optype="continuous" dataType="double"/>
        </DataDictionary>
        <MiningModel functionName="regression">
          <MiningSchema>
            <MiningField name="y" usageType="target"/>
            <MiningField name="x"/>
          </MiningSchema>
          <Segmentation multipleModelMethod="sum">
            <Segment id="1">
              <True/>
              <TreeModel functionName="regression" missingValueStrategy="none" noTrueChildStrategy="returnLastPrediction" splitCharacteristic="multiSplit">
                <MiningSchema><MiningField name="x"/></MiningSchema>
                <Node score="7">
                  <True/>
                  <Node score="10"><SimplePredicate field="x" operator="lessOrEqual" value="0"/></Node>
                  <Node score="11"><SimplePredicate field="x" operator="greaterThan" value="0"/></Node>
                </Node>
              </TreeModel>
            </Segment>
          </Segmentation>
        </MiningModel>
      </PMML>
    XML

    model = Eps::Model.load_pmml(pmml)
    assert_elements_in_delta [7, 10, 11], model.predict([{x: nil}, {x: 0}, {x: 1}])
  end

  def test_missing_default_child_export
    trees = [
      Eps::Evaluators::Node.new(score: 0, children: [
        Eps::Evaluators::Node.new(score: 10, predicate: {field: "x", operator: "lessOrEqual", value: 0}),
        Eps::Evaluators::Node.new(score: 11, predicate: {field: "x", operator: "greaterThan", value: 0}, children: [
          Eps::Evaluators::Node.new(score: 12, predicate: {field: "z", operator: "lessOrEqual", value: 0}),
          Eps::Evaluators::Node.new(score: 13, predicate: {field: "z", operator: "greaterThan", value: 0})
        ])
      ])
    ]
    trees.first.default_child = trees.first.children.last
    right = trees.first.children.last
    right.default_child = right.children.first

    model = Eps::LightGBM.new
    features = {"x" => "numeric", "z" => "numeric"}
    model.instance_variable_set(:@features, features)
    model.instance_variable_set(:@text_features, {})
    model.instance_variable_set(:@label_encoders, {})
    model.instance_variable_set(:@objective, "regression")
    model.instance_variable_set(:@labels, nil)
    model.instance_variable_set(:@trees, trees)
    model.instance_variable_set(:@target, "y")
    model.instance_variable_set(:@evaluator, Eps::Evaluators::LightGBM.new(trees: trees, objective: "regression", labels: nil, features: features, text_features: {}))

    pmml = model.to_pmml
    assert_includes pmml, "defaultChild"
    assert_valid_pmml(pmml)

    rows = [{x: nil, z: nil}, {x: nil, z: 5}, {x: 5, z: nil}, {x: 0, z: 5}]
    assert_elements_in_delta [12, 13, 12, 10], model.predict(rows)

    model = Eps::Model.load_pmml(pmml)
    assert_elements_in_delta [12, 13, 12, 10], model.predict(rows)
  end

  def test_missing_default_child_sparse_zero
    absent = Eps::Evaluators::Node.new(score: 99, predicate: {field: ["t", "w"], operator: "equal", value: "sentinel"})
    gt = Eps::Evaluators::Node.new(score: 1, predicate: {field: ["t", "w"], operator: "greaterThan", value: 0})
    le = Eps::Evaluators::Node.new(score: 2, predicate: {field: ["t", "w"], operator: "lessOrEqual", value: 0})
    root = Eps::Evaluators::Node.new(score: 0, children: [absent, gt, le])
    root.default_child = absent

    evaluator = Eps::Evaluators::LightGBM.new(trees: [root], objective: "regression", labels: nil, features: {"t" => "text"}, text_features: {})

    df = Eps::DataFrame.new
    df.columns[["t", "w"]] = [nil]

    # a sparse term of zero is a real value, so it takes the <= branch, not the default child
    assert_elements_in_delta [2], evaluator.predict(df)
  end

  private

  def model
    @model ||= Eps::Model.load_pmml(File.read("test/support/python/lightgbm_regression.pmml"))
  end
end


