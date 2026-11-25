module Unicode
  TURKIC_LANGUAGES = [
      PragmaticTokenizer::Languages::Turkish
  ].freeze

  def self.downcase(text)
    text.downcase
  end

  def self.capitalize(text)
    text.capitalize
  end

  # Builds a language dependent canonical key used for stop word matching.
  #
  # The key applies the language specific case folding rules (for example the
  # Turkish/Azeri dotted/dotless i mapping) followed by Unicode canonical
  # normalization, so that composed and decomposed glyphs (e.g. "é" and
  # "e" + combining acute) share a key while distinct letters such as "é" and
  # "e" do not. The original token is never modified: the key only serves the
  # comparison, leaving the output token's glyph and casing untouched.
  def self.canonicalize(text, language_module = nil)
    locale = locale_for(language_module)
    folded = locale ? text.downcase(locale) : text.downcase
    folded.unicode_normalize(:nfc)
  end

  def self.locale_for(language_module)
    :turkic if TURKIC_LANGUAGES.include?(language_module)
  end
end
