require 'spec_helper'

RSpec.describe PragmaticSegmenter::Segmenter, "document side abbreviation configuration" do
  describe "#segment with :abbreviation" do
    let(:config) do
      {
        unit_abbreviations: ['ms'],
        number_abbreviations: ['fig']
      }
    end

    it "keeps the built-in defaults when no configuration is provided" do
      text = "The value is 2.0 ms. Next run passed. Refer to Fig. 2. Another source."
      ps = PragmaticSegmenter::Segmenter.new(text: text)
      expect(ps.segment).to eq(
        [
          "The value is 2.0 ms. Next run passed.",
          "Refer to Fig.",
          "2.",
          "Another source."
        ]
      )
    end

    it "does not mutate the language's built-in abbreviation table" do
      text = "The value is 2.0 ms. Next run passed."
      expect do
        PragmaticSegmenter::Segmenter.new(text: text, abbreviation: config).segment
      end.not_to change {
        PragmaticSegmenter::Languages::English::Abbreviation::PREPOSITIVE_ABBREVIATIONS.include?('ms')
      }.from(true)
    end

    it "splits the four sentences while keeping 2.0 and Fig. 2 intact" do
      text = "The value is 2.0 ms. Next run passed. Refer to Fig. 2. Another source."
      ps = PragmaticSegmenter::Segmenter.new(text: text, abbreviation: config)
      expect(ps.segment).to eq(
        [
          "The value is 2.0 ms.",
          "Next run passed.",
          "Refer to Fig. 2.",
          "Another source."
        ]
      )
    end

    context "unit of measure abbreviations" do
      it "closes a sentence before an explicit capitalized sentence starter" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "It took 5 ms. Next run passed.",
          abbreviation: config
        )
        expect(ps.segment).to eq(["It took 5 ms.", "Next run passed."])
      end

      it "stays mid-sentence when followed by a lower case word" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "It waited 5 ms. before timing out.",
          abbreviation: config
        )
        expect(ps.segment).to eq(["It waited 5 ms. before timing out."])
      end

      it "stays mid-sentence when followed by a digit" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "Delay was 3 ms. 5 retries happened.",
          abbreviation: config
        )
        expect(ps.segment).to eq(["Delay was 3 ms. 5 retries happened."])
      end
    end

    context "number prefix abbreviations" do
      it "glues the period to the following figure reference" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "Refer to Fig. 2. Another source.",
          abbreviation: config
        )
        expect(ps.segment).to eq(["Refer to Fig. 2.", "Another source."])
      end

      it "is case insensitive (lower case fig.)" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "See fig. 7 for details. Then continue.",
          abbreviation: config
        )
        expect(ps.segment).to eq(["See fig. 7 for details.", "Then continue."])
      end

      it "works at the start of the text" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "Fig. 9 shows the curve. Next paragraph.",
          abbreviation: config
        )
        expect(ps.segment).to eq(["Fig. 9 shows the curve.", "Next paragraph."])
      end
    end

    context "ordinary abbreviations" do
      let(:plain_config) { { abbreviations: ['xyz'] } }

      it "protects the period before a lower case continuation" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "We used xyz. alpha continues here.",
          abbreviation: plain_config
        )
        expect(ps.segment).to eq(["We used xyz. alpha continues here."])
      end

      it "closes the sentence before a capitalized starter" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "We used xyz. Then it ended.",
          abbreviation: plain_config
        )
        expect(ps.segment).to eq(["We used xyz.", "Then it ended."])
      end
    end

    context "prepositive (honorific) abbreviations" do
      let(:honorific_config) { { prepositive_abbreviations: ['eng'] } }

      it "always protects the period, even before a capitalized word" do
        ps = PragmaticSegmenter::Segmenter.new(
          text: "Ask Eng. Adams. Later.",
          abbreviation: honorific_config
        )
        expect(ps.segment).to eq(["Ask Eng. Adams.", "Later."])
      end
    end
  end
end
