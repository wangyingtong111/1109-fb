# frozen_string_literal: true

module PragmaticSegmenter
  # Plans how Unicode line and paragraph separators describe a visual layout.
  #
  # U+2029 is always a hard paragraph boundary. The meaning of each U+2028 is
  # supplied by its original character (or byte) position, or by its
  # zero/one-based occurrence number:
  #
  #   * :hard_line   - end the current logical line
  #   * :inter_word  - replace the wrap with an ordinary word space
  #   * :intra_word  - join the two wrapped word fragments without a space
  class LayoutText
    LINE_SEPARATOR = "\u2028".freeze
    PARAGRAPH_SEPARATOR = "\u2029".freeze

    HARD_LINE = :hard_line
    INTER_WORD = :inter_word
    INTRA_WORD = :intra_word

    TEXT_MODES = %i[
      layout_text text_layout layout layout_text_mode text_layout_mode
      unicode_layout unicode_layout_text unicode text
    ].freeze

    ALIASES = {
      hard_line: HARD_LINE,
      hard_line_break: HARD_LINE,
      hard_break: HARD_LINE,
      hardline: HARD_LINE,
      hard_wrap: HARD_LINE,
      hard: HARD_LINE,
      line: HARD_LINE,
      line_break: HARD_LINE,
      break: HARD_LINE,
      newline: HARD_LINE,
      cr: HARD_LINE,
      split: HARD_LINE,
      paragraph: HARD_LINE,
      inter_word: INTER_WORD,
      inter_word_break: INTER_WORD,
      inter_word_wrap: INTER_WORD,
      interword: INTER_WORD,
      between_words: INTER_WORD,
      between_word_break: INTER_WORD,
      word_boundary: INTER_WORD,
      word_break: INTER_WORD,
      word_wrap: INTER_WORD,
      wrap: INTER_WORD,
      word: INTER_WORD,
      space: INTER_WORD,
      whitespace: INTER_WORD,
      soft_wrap: INTER_WORD,
      soft_line_break: INTER_WORD,
      intra_word: INTRA_WORD,
      intra_word_break: INTRA_WORD,
      intra_word_wrap: INTRA_WORD,
      intraword: INTRA_WORD,
      in_word: INTRA_WORD,
      in_word_wrap: INTRA_WORD,
      within_word: INTRA_WORD,
      within_word_break: INTRA_WORD,
      word_internal: INTRA_WORD,
      word_internal_break: INTRA_WORD,
      internal_word: INTRA_WORD,
      internal_word_break: INTRA_WORD,
      intra: INTRA_WORD,
      join: INTRA_WORD,
      merge: INTRA_WORD,
      concatenate: INTRA_WORD,
      remove: INTRA_WORD,
      soft_hyphen: INTRA_WORD,
      dehyphenate: INTRA_WORD
    }.freeze

    CHINESE_ALIASES = {
      '硬行' => HARD_LINE,
      '硬行折行' => HARD_LINE,
      '硬换行' => HARD_LINE,
      '硬边界' => HARD_LINE,
      '硬段' => HARD_LINE,
      '词间折行' => INTER_WORD,
      '词间断行' => INTER_WORD,
      '词间换行' => INTER_WORD,
      '词间' => INTER_WORD,
      '词内折行' => INTRA_WORD,
      '词内断行' => INTRA_WORD,
      '词内换行' => INTRA_WORD,
      '词内' => INTRA_WORD
    }.freeze

    DIRECT_OPTION_KEYS = %i[
      breaks line_breaks layout_breaks break_positions line_break_positions
      break_types line_break_types line_break_modes line_break_kinds
      line_separators unicode_line_breaks u2028 separators positions
      layout_options
    ].freeze

    GROUP_OPTION_KEYS = {
      HARD_LINE => %i[
        hard_line hard_line_breaks hard_breaks hard_line_positions hard_positions hard_lines
      ],
      INTER_WORD => %i[
        inter_word inter_word_breaks interword_breaks word_boundary_breaks
        inter_word_positions word_boundary_positions word_wrap_positions
      ],
      INTRA_WORD => %i[
        intra_word intra_word_breaks intraword_breaks in_word_breaks word_internal_breaks
        intra_word_positions intraword_positions in_word_positions
        word_internal_positions
      ]
    }.freeze

    attr_reader :default_line_break

    def self.resolve(layout_text: nil, layout: nil, line_breaks: nil, options: {}, &block)
      return layout_text if layout_text.is_a?(LayoutText)
      return layout if layout.is_a?(LayoutText)

      enabled = block_given?
      declaration = nil
      default = HARD_LINE

      [layout_text, layout, options[:layout_mode], options[:text_layout],
       options[:unicode_layout], options[:layout_text_mode], options[:mode]].each do |value|
        next if value.nil?

        if value == true || text_mode_name?(value)
          enabled = true
        elsif value == false
          next
        elsif value.is_a?(Proc)
          declaration = value
          enabled = true
        elsif value.is_a?(Hash) || value.is_a?(Array)
          declaration = value
          enabled = true
        else
          category = category_for(value)
          if category
            default = category
            enabled = true
          end
        end
      end

      options.each do |key, value|
        next unless key.is_a?(Symbol) && key.to_s.include?('layout')
        next if value.nil? || value == false

        if value == true || text_mode_name?(value)
          enabled = true
        elsif value.is_a?(Proc) || value.is_a?(Hash) || value.is_a?(Array)
          declaration ||= value
          enabled = true
        elsif (category = category_for(value))
          default = category
          enabled = true
        end
      end

      declaration ||= line_breaks if line_breaks
      declaration ||= declaration_from_options(options)

      return nil unless enabled || declaration || block_given?

      new(declaration, default: default, &block)
    end

    def self.text_mode_name?(value)
      return false unless value.is_a?(Symbol) || value.is_a?(String)

      TEXT_MODES.include?(value.to_s.downcase.gsub(/[\s-]+/, '_').to_sym) ||
        value.to_s == '版面文本' || value.to_s == '版面文本模式'
    end

    def self.category_for(value)
      return CHINESE_ALIASES[value.to_s] if CHINESE_ALIASES.key?(value.to_s)

      case value
      when '' then INTRA_WORD
      when ' ', /\A[[:space:]]\z/ then INTER_WORD
      when "\n", "\r", LINE_SEPARATOR, PARAGRAPH_SEPARATOR then HARD_LINE
      when Symbol, String
        name = value.to_s.downcase.gsub(/[\s-]+/, '_').to_sym
        ALIASES[name]
      end
    end

    def self.category_from_option_name(name)
      value = name.to_s.downcase
      if value =~ /intra|in_word|in-word|within|internal|hyphen|join|merge/
        INTRA_WORD
      elsif value =~ /hard/
        HARD_LINE
      elsif value =~ /inter|between|boundary|space|word|wrap|soft/
        INTER_WORD
      end
    end

    def self.declaration_from_options(options)
      groups = {}
      GROUP_OPTION_KEYS.each do |category, keys|
        keys.each do |key|
          next unless options.key?(key)
          groups[category] ||= []
          groups[category].concat(Array(options[key]))
        end
      end

      matching = options.select do |key, value|
        key.is_a?(Symbol) &&
          key.to_s =~ /break|separator|u_?2028|2028|wrap|position|index|offset|location/ &&
          (value.is_a?(Hash) || value.is_a?(Array))
      end

      direct = nil
      DIRECT_OPTION_KEYS.each do |key|
        next unless options.key?(key)
        value = options[key]
        if key == :layout_options && value.is_a?(Hash)
          direct = declaration_from_options(value)
        elsif value.is_a?(Hash) || value.is_a?(Array)
          direct = value
        end
        break if direct
      end

      matching.each do |key, value|
        next if DIRECT_OPTION_KEYS.include?(key) || GROUP_OPTION_KEYS.values.flatten.include?(key)

        all_positions = value.is_a?(Array) && value.all? { |item| item.is_a?(Integer) }
        category = category_from_option_name(key)
        if all_positions && category
          groups[category] ||= []
          groups[category].concat(value)
        elsif direct.nil?
          direct = value
        end
      end

      return [groups, direct] if groups.any? && direct
      return groups if groups.any?
      return direct if direct

      matching.values.first if matching.one?
    end

    def initialize(declarations = nil, default: HARD_LINE, &resolver)
      @resolver = resolver
      if declarations.is_a?(Proc)
        @resolver = declarations
        declarations = nil
      end
      @default_line_break = self.class.category_for(default) || HARD_LINE
      @position_rules = {}
      @ordinal_rules = {}
      @ambiguous_rules = {}
      ingest(declarations) if declarations
    end

    # Returns logical hard blocks. Soft U+2028 breaks are joined or converted
    # to spaces before the existing sentence segmenter is invoked.
    def hard_blocks(text)
      blocks = []
      buffer = +''
      line_break_ordinal = -1
      char_position = 0
      byte_position = 0
      position_rules, ordinal_rules = resolved_rules(text)

      text.each_char do |char|
        case char
        when PARAGRAPH_SEPARATOR
          blocks << buffer
          buffer = +''
        when LINE_SEPARATOR
          line_break_ordinal += 1
          mode = mode_for(
            char_position: char_position,
            byte_position: byte_position,
            ordinal: line_break_ordinal,
            position_rules: position_rules,
            ordinal_rules: ordinal_rules
          )

          case mode
          when INTRA_WORD
            # Intentionally append nothing.
          when INTER_WORD
            append_inter_word_space(buffer, text, char_position + 1)
          else
            blocks << buffer
            buffer = +''
          end
        else
          buffer << char
        end

        char_position += 1
        byte_position += char.bytesize
      end

      blocks << buffer
      blocks
    end

    def prepare(text)
      hard_blocks(text).join("\r")
    end
    alias process prepare
    alias call prepare

    private

    def ingest(value, forced_category = nil)
      case value
      when LayoutText
        @position_rules.merge!(value.instance_variable_get(:@position_rules))
        @ordinal_rules.merge!(value.instance_variable_get(:@ordinal_rules))
        @ambiguous_rules.merge!(value.instance_variable_get(:@ambiguous_rules))
      when Hash
        if layout_entry_hash?(value)
          ingest_hash_item(value, @default_line_break)
          return
        end

        nested = self.class.declaration_from_options(value)
        if nested && nested != value
          ingest(nested, forced_category)
        elsif category_group_hash?(value)
          value.each do |category_key, entries|
            category = self.class.category_for(category_key)
            ingest(entries, category)
          end
        else
          value.each { |key, item| ingest_pair(key, item, forced_category) }
        end
      when Array
        if value.all? { |item| item.is_a?(Integer) }
          category = forced_category || @default_line_break
          value.each { |position| add_position(position, category) }
        else
          value.each_with_index do |item, index|
            category = self.class.category_for(item)
            if category && (item.is_a?(Symbol) || item.is_a?(String))
              add_ordinal(index, forced_category || category)
            else
              ingest(item, forced_category)
            end
          end
        end
      when Integer
        add_position(value, forced_category || @default_line_break)
      when String, Symbol
        category = self.class.category_for(value)
        @default_line_break = category if category && !forced_category
      end
    end

    def ingest_pair(key, value, forced_category = nil)
      if (category = self.class.category_for(key)) && (value.is_a?(Array) || value.is_a?(Integer) || value.is_a?(Hash))
        ingest(value, category)
        return
      end

      category = forced_category || self.class.category_for(value) || @default_line_break

      case key
      when Integer
        add_ambiguous(key, category)
      when /\A-?\d+\z/
        add_ambiguous(key.to_i, category)
      when Symbol, String
        ordinal = named_ordinal(key)
        ordinal ? add_ordinal(ordinal, category) : nil
      when Hash
        ingest_hash_item(key, category)
      when Array
        key.each { |part| ingest_pair(part, value, forced_category) }
      end
    end

    def named_ordinal(key)
      name = key.to_s.downcase.gsub(/[\s-]+/, '_')
      named = {
        'first' => 0, 'first_line_break' => 0,
        'second' => 1, 'second_line_break' => 1,
        'third' => 2, 'third_line_break' => 2,
        'fourth' => 3, 'fourth_line_break' => 3,
        'fifth' => 4, 'fifth_line_break' => 4
      }
      return named[name] if named.key?(name)

      case name
      when /\A(?:line_break|break|ordinal|occurrence|index)_(\d+)\z/
        Regexp.last_match(1).to_i
      end
    end

    def layout_entry_hash?(hash)
      keys = hash.keys
      keys.any? do |key|
        key.respond_to?(:to_sym) &&
          %i[position positions at index indices offset offsets
             byte_position byte_positions char_position char_positions
             location locations occurrence ordinal number nth].include?(key.to_sym)
      end
    end

    def ingest_hash_item(hash, fallback_category)
      position_keys = %i[
        position positions at index indices offset offsets
        byte_position byte_positions char_position char_positions
        location locations
      ]
      ordinal_keys = %i[occurrence occurrences ordinal ordinals number numbers nth]

      positions = collect_hash_values(hash, position_keys)
      ordinals = collect_hash_values(hash, ordinal_keys)

      category_value = hash[:type] || hash['type'] ||
                       hash[:as] || hash['as'] ||
                       hash[:kind] || hash['kind'] ||
                       hash[:mode] || hash['mode'] ||
                       hash[:role] || hash['role'] ||
                       hash[:break] || hash['break'] ||
                       fallback_category
      category = self.class.category_for(category_value) || fallback_category

      positions.each { |position| add_position(position, category) if position.is_a?(Integer) }
      ordinals.each { |ordinal| add_ordinal(ordinal, category) if ordinal.is_a?(Integer) }
    end

    def collect_hash_values(hash, keys)
      keys.flat_map do |key|
        value = hash[key] || hash[key.to_s]
        value.nil? ? [] : Array(value)
      end
    end

    def category_group_hash?(hash)
      keys = hash.keys
      keys.any? && keys.all? do |key|
        category = self.class.category_for(key)
        category && ![key.to_s, ' ', '', "\n", "\r", LINE_SEPARATOR, PARAGRAPH_SEPARATOR].include?(key)
      end
    end

    def add_position(position, category)
      return unless position.is_a?(Integer)
      @position_rules[position] = category
    end

    def add_ambiguous(key, category)
      return unless key.is_a?(Integer)
      @ambiguous_rules[key] = category
    end

    def add_ordinal(ordinal, category)
      @ordinal_rules[ordinal.to_i] = category if ordinal
    end

    def resolved_rules(text)
      position_rules = @position_rules.dup
      ordinal_rules = @ordinal_rules.dup
      return [position_rules, ordinal_rules] if @ambiguous_rules.empty?

      occurrence_keys = @ambiguous_rules.keys.reject do |key|
        text[key] == LINE_SEPARATOR || text.byteslice(key, 3) == LINE_SEPARATOR
      end
      line_break_count = text.scan(LINE_SEPARATOR).length
      ordinal_base = if occurrence_keys.include?(0)
                       0
                     elsif occurrence_keys.any? { |key| key == line_break_count }
                       1
                     else
                       0
                     end

      @ambiguous_rules.each do |key, category|
        if text[key] == LINE_SEPARATOR || text.byteslice(key, 3) == LINE_SEPARATOR
          position_rules[key] = category
        else
          ordinal_rules[key - ordinal_base] = category
        end
      end

      [position_rules, ordinal_rules]
    end

    def mode_for(char_position:, byte_position:, ordinal:, position_rules:, ordinal_rules:)
      declared = resolver_value(
        char_position: char_position,
        byte_position: byte_position,
        ordinal: ordinal
      )
      category = self.class.category_for(declared) if declared
      return category if category

      position_rules[char_position] ||
        position_rules[byte_position] ||
        ordinal_rules[ordinal] ||
        @default_line_break
    end

    def resolver_value(char_position:, byte_position:, ordinal:)
      return unless @resolver

      parameters = @resolver.parameters
      if parameters.any? { |type, _| %i[key keyreq keyrest].include?(type) }
        return @resolver.call(
          position: char_position,
          char_position: char_position,
          byte_position: byte_position,
          index: ordinal,
          ordinal: ordinal,
          occurrence: ordinal,
          character: LINE_SEPARATOR
        )
      end

      case @resolver.arity
      when 0 then @resolver.call
      when 1 then @resolver.call(char_position)
      when 2 then @resolver.call(char_position, ordinal)
      else @resolver.call(char_position, ordinal, byte_position)
      end
    end

    def append_inter_word_space(buffer, text, next_position)
      following = text[next_position]
      previous_is_space = !buffer.empty? && buffer[-1] =~ /[[:space:]]/
      following_is_space = following && following =~ /[[:space:]]/
      return if previous_is_space || following_is_space

      buffer << ' '
    end
  end

  LayoutTextMode = LayoutText unless const_defined?(:LayoutTextMode)
  TextLayout = LayoutText unless const_defined?(:TextLayout)
  Layout = LayoutText unless const_defined?(:Layout)
end
