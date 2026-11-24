module Eps
  module PMML
    class Loader
      attr_reader :data

      def initialize(pmml)
        if pmml.is_a?(String)
          pmml = Nokogiri::XML(pmml) { |config| config.strict }
        end
        @data = pmml
      end

      def load
        if data.css("Segmentation").any?
          lightgbm
        elsif data.css("RegressionModel").any?
          linear_regression
        elsif data.css("NaiveBayesModel").any?
          naive_bayes
        else
          raise "Unknown model"
        end
      end

      private

      def lightgbm
        objective = data.css("MiningModel").first.attribute("functionName").value
        if objective == "classification"
          labels = data.css("RegressionModel OutputField").map { |n| n.attribute("value").value }
          objective = labels.size > 2 ? "multiclass" : "binary"
        end

        features = {}
        text_features, derived_fields = extract_text_features(data, features)
        node = data.css("DataDictionary").first
        node.css("DataField")[1..-1].to_a.each do |node|
          features[node.attribute("name").value] =
            if node.attribute("optype").value == "categorical"
              "categorical"
            else
              "numeric"
            end
        end

        trees = []
        data.css("Segmentation TreeModel").each do |tree|
          node = find_nodes(tree.css("Node").first, derived_fields)
          trees << node
        end

        Evaluators::LightGBM.new(trees: trees, objective: objective, labels: labels, features: features, text_features: text_features)
      end

      def linear_regression
        node = data.css("RegressionTable")

        coefficients = {
          "_intercept" => node.attribute("intercept").value.to_f
        }

        features = {}

        text_features, derived_fields = extract_text_features(data, features)

        node.css("NumericPredictor").each do |n|
          name = n.attribute("name").value
          if derived_fields[name]
            name = derived_fields[name]
          else
            features[name] = "numeric"
          end
          coefficients[name] = n.attribute("coefficient").value.to_f
        end

        node.css("CategoricalPredictor").each do |n|
          name = n.attribute("name").value
          coefficients[[name, n.attribute("value").value]] = n.attribute("coefficient").value.to_f
          features[name] = "categorical"
        end

        field_settings = parse_field_settings(data, features)

        Evaluators::LinearRegression.new(
          coefficients: coefficients,
          features: features,
          text_features: text_features,
          field_settings: field_settings
        )
      end

      def parse_field_settings(data, features)
        domains = pmml_domains(data)
        settings = {}

        data.css("MiningField").each do |node|
          name = node.attribute("name")&.value
          next unless name && features.key?(name)

          usage_type = node.attribute("usageType")&.value
          next if %w[target predicted].include?(usage_type)

          data_type = domains.dig(name, :data_type)
          invalid_treatment = node.attribute("invalidValueTreatment")&.value
          invalid_replacement = node.attribute("invalidValueReplacement")&.value
          invalid_policy_declared = invalid_treatment || invalid_replacement

          settings[name] = (domains[name] || {}).merge(
            missing_replacement: parse_scalar(node.attribute("missingValueReplacement")&.value, data_type),
            missing_treatment: node.attribute("missingValueTreatment")&.value,
            invalid_treatment: invalid_treatment || (invalid_replacement ? "asValue" : "returnInvalid"),
            invalid_replacement: parse_scalar(invalid_replacement, data_type),
            invalid_policy_declared: !!invalid_policy_declared,
            outliers: node.attribute("outliers")&.value || "asIs",
            low_value: parse_number(node.attribute("lowValue")&.value),
            high_value: parse_number(node.attribute("highValue")&.value)
          )
        end

        features.each_key do |name|
          settings[name] ||= domains[name] if domains[name]
        end

        settings
      end

      def pmml_domains(data)
        data.css("DataDictionary DataField").each_with_object({}) do |node, domains|
          name = node.attribute("name").value
          data_type = node.attribute("dataType")&.value
          values = node.css("Value")

          domains[name] = {
            data_type: data_type,
            optype: node.attribute("optype")&.value,
            valid_values: pmml_values(values, "valid", data_type),
            invalid_values: pmml_values(values, "invalid", data_type),
            missing_values: pmml_values(values, "missing", data_type),
            intervals: node.css("Interval").map { |interval| parse_interval(interval) }
          }
        end
      end

      def pmml_values(nodes, property, data_type)
        values = nodes.select do |node|
          (node.attribute("property")&.value || "valid") == property
        end
        values.map { |node| parse_scalar(node.attribute("value").value, data_type) }
      end

      def parse_interval(node)
        {
          closure: node.attribute("closure").value,
          left: parse_number(node.attribute("leftMargin")&.value),
          right: parse_number(node.attribute("rightMargin")&.value)
        }
      end

      def parse_scalar(value, data_type)
        return nil if value.nil?

        case data_type
        when "integer"
          value.to_i
        when "float", "double", "real", "number"
          value.to_f
        when "boolean"
          value == "true" || value == "1"
        else
          value
        end
      end

      def parse_number(value)
        value&.to_f
      end

      def naive_bayes
        node = data.css("NaiveBayesModel")

        prior = {}
        node.css("BayesOutput TargetValueCount").each do |n|
          prior[n.attribute("value").value] = n.attribute("count").value.to_f
        end

        legacy = false

        conditional = {}
        features = {}
        node.css("BayesInput").each do |n|
          prob = {}

          # numeric
          n.css("TargetValueStat").each do |n2|
            n3 = n2.css("GaussianDistribution")
            prob[n2.attribute("value").value] = {
              mean: n3.attribute("mean").value.to_f,
              stdev: Math.sqrt(n3.attribute("variance").value.to_f)
            }
          end

          # detect bad form in Eps < 0.3
          bad_format = n.css("PairCounts").map { |n2| n2.attribute("value").value } == prior.keys

          # categorical
          n.css("PairCounts").each do |n2|
            if bad_format
              n2.css("TargetValueCount").each do |n3|
                prob[n3.attribute("value").value] ||= {}
                prob[n3.attribute("value").value][n2.attribute("value").value] = n3.attribute("count").value.to_f
              end
            else
              boom = {}
              n2.css("TargetValueCount").each do |n3|
                boom[n3.attribute("value").value] = n3.attribute("count").value.to_f
              end
              prob[n2.attribute("value").value] = boom
            end
          end

          if bad_format
            legacy = true
            prob.each do |k, v|
              prior.keys.each do |k|
                v[k] ||= 0.0
              end
            end
          end

          name = n.attribute("fieldName").value
          conditional[name] = prob
          features[name] = n.css("TargetValueStat").any? ? "numeric" : "categorical"
        end

        probabilities = {
          prior: prior,
          conditional: conditional
        }

        # get derived fields
        derived = {}
        data.css("DerivedField").each do |n|
          name = n.attribute("name").value
          field = n.css("NormDiscrete").attribute("field").value
          value = n.css("NormDiscrete").attribute("value").value
          features.delete(name)
          features[field] = "derived"
          derived[field] ||= {}
          derived[field][name] = value
        end

        Evaluators::NaiveBayes.new(probabilities: probabilities, features: features, derived: derived, legacy: legacy)
      end

      def extract_text_features(data, features)
        # updates features object
        vocabulary = {}
        function_mapping = {}
        derived_fields = {}
        data.css("LocalTransformations DerivedField, TransformationDictionary DerivedField").each do |n|
          name = n.attribute("name")&.value
          field = n.css("FieldRef").attribute("field").value
          value = n.css("Constant").text

          field = field[10..-2] if field =~ /\Alowercase\(.+\)\z/
          next if value.empty?

          (vocabulary[field] ||= []) << value

          function_mapping[field] = n.css("Apply").attribute("function").value

          derived_fields[name] = [field, value]
        end

        functions = {}
        data.css("TransformationDictionary DefineFunction").each do |n|
          name = n.attribute("name").value
          text_index = n.css("TextIndex")
          functions[name] = {
            tokenizer: Regexp.new(text_index.attribute("wordSeparatorCharacterRE").value),
            case_sensitive: text_index.attribute("isCaseSensitive")&.value == "true"
          }
        end

        text_features = {}
        function_mapping.each do |field, function|
          text_features[field] = functions[function].merge(vocabulary: vocabulary[field])
          features[field] = "text"
        end

        [text_features, derived_fields]
      end

      def find_nodes(xml, derived_fields)
        score = xml.attribute("score").value.to_f

        elements = xml.elements
        xml_predicate = elements.first

        predicate =
          if xml_predicate.name == "True"
            nil
          elsif xml_predicate.name == "SimpleSetPredicate"
            operator = "in"
            value = xml_predicate.css("Array").text.scan(/"(.+?)(?<!\\)"|(\S+)/).flatten.compact.map { |v| v.gsub('\"', '"') }
            field = xml_predicate.attribute("field").value
            field = derived_fields[field] if derived_fields[field]
            {
              field: field,
              operator: operator,
              value: value
            }
          else
            operator = xml_predicate.attribute("operator").value
            value = xml_predicate.attribute("value").value
            value = value.to_f if operator == "greaterThan" || operator == "lessOrEqual"
            field = xml_predicate.attribute("field").value
            field = derived_fields[field] if derived_fields[field]
            {
              field: field,
              operator: operator,
              value: value
            }
          end

        children = elements[1..-1].map { |n| find_nodes(n, derived_fields) }

        Evaluators::Node.new(score: score, predicate: predicate, children: children)
      end
    end
  end
end
