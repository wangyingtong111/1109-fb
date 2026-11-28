# frozen_string_literal: true

module Grape
  class DeclaredParamsHandler
    def initialize(include_missing: true, evaluate_given: false, stringify: false, contract_key_map: [])
      @include_missing = include_missing
      @evaluate_given = evaluate_given
      @stringify = stringify
      @contract_key_map = contract_key_map
    end

    # @param passed_params [Hash, Array] the params to filter
    # @param declared_params [Array<Grape::DeclaredParams::Param>] the params
    #   to keep of them
    #
    # An element of an Array scope that is not a Hash has no keys to filter,
    # so it is returned as it came in, the way a leaf value is. Validation
    # lets one through wherever nothing in the scope is required of it: an
    # optional scope passes over blank elements, and a scope whose params are
    # all optional checks nothing on an element that is not a Hash. Filtering
    # it anyway meant building an empty copy of it, which raised for +nil+,
    # +false+ and numbers and indexed a String with a Symbol.
    def call(passed_params, declared_params)
      return passed_params unless passed_params.is_a?(Array) || passed_params.respond_to?(:key?)

      res = if passed_params.is_a?(Array)
              passed_params.map { |passed_param| call(passed_param, declared_params) }
            else
              declared_hash(passed_params, declared_params)
            end

      @contract_key_map.each { |key_map| key_map.write(passed_params, res) }

      res
    end

    private

    def declared_hash(passed_params, declared_params)
      declared_params.each_with_object(passed_params.class.new) do |declared_param, memo|
        next if @evaluate_given && !declared_param.scope.attr_meets_dependency?(passed_params)

        name = declared_param.name
        sent = passed_params.key?(name)
        next unless @include_missing || sent

        memo[@stringify ? declared_param.string_key : declared_param.key] = declared_value(passed_params[name], declared_param, passed_params, sent)
      end
    end

    # What a declared param is filed as.
    #
    # A param the request left out stands for the empty value its type has --
    # a fresh one every time, since the caller may fill it -- or for what its
    # declared children compile to when the type has none. A param the
    # request sent stands for what it sent, an explicit +nil+ included: null
    # is how a client clears a field, so it stays null rather than being
    # filed as an empty Array, Hash or Set and read back as a param that was
    # never sent.
    def declared_value(passed_param, declared_param, passed_params, sent)
      return if sent && passed_param.nil?

      children = declared_param.children
      return declared_leaf(passed_param, declared_param) unless children

      declared_children(passed_param || passed_params.class.new, declared_param, children)
    end

    # A leaf param's value, or the empty value its type stands for when the
    # request left it out and it has one.
    def declared_leaf(passed_param, declared_param)
      passed_param || declared_param.empty_value&.dup || passed_param
    end

    # A param with children is filed as what #call keeps of its value, a Hash
    # of the same class as the params around it when it was not passed.
    #
    # A value that is neither a Hash nor an Array where children are declared
    # -- left unvalidated by a +given+ whose dependency is not met, say -- is
    # returned as it came in, as #call returns an element like it.
    def declared_children(passed_children, declared_param, children)
      has_passed_children = !passed_children.respond_to?(:any?) || passed_children.any?
      (declared_param.empty_value&.dup unless has_passed_children) || call(passed_children, children)
    end
  end
end
