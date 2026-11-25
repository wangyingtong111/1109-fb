module PragmaticTokenizer
  module Languages
    module Common
      # U+E000 is in the Private Use Area, so it will not collide with the literal U+2607 (☇) symbol in text.
      PUNCTUATION_MAP = { "。" => "♳", "．" => "♴", "." => "♵", "！" => "♶", "!" => "♷", "?" => "♸", "？" => "♹", "、" => "♺", "¡" => "⚀", "¿" => "⚁", "„" => "⚂", "“" => "⚃", "[" => "⚄", "]" => "⚅", "\"" => "\uE000", "#" => "☈", "$" => "☉", "%" => "☊", "&" => "☋", "(" => "☌", ")" => "☍", "*" => "☠", "+" => "☢", "," => "☣", ":" => "☤", ";" => "☥", "<" => "☦", "=" => "☧", ">" => "☀", "@" => "☁", "^" => "☂", "_" => "☃", "`" => "☄", "'" => "☮", "{" => "♔", "|" => "♕", "}" => "♖", "~" => "♗", "-" => "♘", "«" => "♙", "»" => "♚", "”" => "⚘", "‘" => "⚭" }.freeze
      ABBREVIATIONS   = Set.new([]).freeze
      STOP_WORDS      = Set.new([]).freeze
      CONTRACTIONS    = {}.freeze

      class SingleQuotes

        ALNUM_QUOTE     = /(\w|\D)'(?!')(?=\W|$)/
        QUOTE_WORD      = /(\W|^)'(?=\w)/
        QUOTE_NOT_TWAS1 = /(\W|^)'(?!twas)/i
        QUOTE_NOT_TWAS2 = /(\W|^)‘(?!twas)/i

        # This 'special treatment' is actually relevant for many other tests. Alter core regular expressions!
        def handle_single_quotes(text)
          # special treatment for "'twas"
          text.gsub!(QUOTE_NOT_TWAS1, '\1 ' << PUNCTUATION_MAP["'".freeze] << ' ')
          text.gsub!(QUOTE_NOT_TWAS2, '\1 ' << PUNCTUATION_MAP["‘".freeze] << ' ')

          text.gsub!(QUOTE_WORD,      ' '   << PUNCTUATION_MAP["'".freeze])
          text.gsub!(ALNUM_QUOTE,     '\1 ' << PUNCTUATION_MAP["'".freeze] << ' ')
          text
        end

      end
    end
  end
end
