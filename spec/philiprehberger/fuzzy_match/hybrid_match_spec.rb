# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Philiprehberger::FuzzyMatch do
  describe '.hybrid_match' do
    # Deterministic fixtures:
    # - text-only pair (no usable phonetic code on either side), text ratio exactly 0.8
    let(:text_only_query) { '12345' }
    let(:text_only_candidate) { '12346' }

    # - phonetic-zero pair: text ratio exactly 0.8, different Soundex codes
    let(:phonetic_query) { 'rube' }
    let(:phonetic_zero_candidate) { 'ruble' }

    before do
      expect(described_class.ratio(text_only_query, text_only_candidate)).to eq(0.8)
      expect(described_class.soundex(text_only_query)).to eq('')
      expect(described_class.soundex(text_only_candidate)).to eq('')

      expect(described_class.ratio(phonetic_query, phonetic_zero_candidate)).to eq(0.8)
      expect(described_class.soundex(phonetic_query)).not_to eq(described_class.soundex(phonetic_zero_candidate))
    end

    it 'returns results, unscored, and the weights used' do
      out = described_class.hybrid_match('rube', ['ruble'])
      expect(out.keys).to contain_exactly(:results, :unscored, :weights)
      expect(out[:weights]).to eq(text: 0.6, phonetic: 0.4)
    end

    it 'defaults to text weight 0.6 and phonetic weight 0.4' do
      out = described_class.hybrid_match(phonetic_query, [phonetic_zero_candidate])
      expect(out[:weights]).to eq(text: 0.6, phonetic: 0.4)
    end

    context 'component participation per candidate' do
      it 'records only the text component when no phonetic code is available' do
        entry = described_class.hybrid_match(text_only_query, [text_only_candidate])[:results].first
        expect(entry[:components]).to eq([:text])
        expect(entry[:component_scores]).to eq(text: 0.8)
        expect(entry[:applied_weights]).to eq(text: 0.6)
      end

      it 'records both components when a phonetic code exists on both sides' do
        entry = described_class.hybrid_match(phonetic_query, [phonetic_zero_candidate])[:results].first
        expect(entry[:components]).to eq(%i[text phonetic])
        expect(entry[:component_scores]).to eq(text: 0.8, phonetic: 0.0)
        expect(entry[:applied_weights]).to eq(text: 0.6, phonetic: 0.4)
      end

      it 'reports which value each participating component produced' do
        entry = described_class.hybrid_match('Robert', ['Rupert'])[:results].first
        expect(entry[:component_scores][:text]).to eq(described_class.ratio('Robert', 'Rupert'))
        expect(entry[:component_scores][:phonetic]).to eq(1.0)
      end
    end

    context 'weight renormalization over available components' do
      it 'keeps a text score of 0.8 when the phonetic component is unavailable' do
        entry = described_class.hybrid_match(text_only_query, [text_only_candidate])[:results].first
        expect(entry[:score]).to eq(0.8)
        expect(entry[:status]).to eq(:scored)
      end

      it 'scores a real phonetic mismatch as 0.48 (0.8*0.6 + 0.0*0.4)' do
        entry = described_class.hybrid_match(phonetic_query, [phonetic_zero_candidate])[:results].first
        expect(entry[:component_scores][:phonetic]).to eq(0.0)
        expect(entry[:score]).to eq(0.48)
      end

      it 'recomputes from the reported weights and component scores' do
        out = described_class.hybrid_match('rube', ['rube', 'ruble', text_only_candidate])
        out[:results].each do |entry|
          weight_sum = entry[:components].sum { |name| entry[:applied_weights][name] }
          recomputed = entry[:components].sum do |name|
            entry[:component_scores][name] * entry[:applied_weights][name]
          end / weight_sum
          expect(entry[:score]).to eq(recomputed.round(4))
        end
      end

      it 'preserves the classic weighted average when every component has a value' do
        entry = described_class.hybrid_match('rube', ['rube'])[:results].first
        expect(entry[:score]).to eq(1.0)
        expect(entry[:component_scores]).to eq(text: 1.0, phonetic: 1.0)
      end

      it 'applies custom weights and still renormalizes missing components' do
        text_only = described_class.hybrid_match(text_only_query, [text_only_candidate],
                                                 text_weight: 0.9, phonetic_weight: 0.1)[:results].first
        expect(text_only[:score]).to eq(0.8)

        both = described_class.hybrid_match(phonetic_query, [phonetic_zero_candidate],
                                            text_weight: 0.9, phonetic_weight: 0.1)[:results].first
        expect(both[:score]).to eq(0.72)
        expect(both[:applied_weights]).to eq(text: 0.9, phonetic: 0.1)
      end
    end

    context 'unscored candidates' do
      it 'lists nil and empty candidates separately with an unscored status' do
        out = described_class.hybrid_match('rube', [nil, ''])
        expect(out[:results]).to eq([])
        expect(out[:unscored].length).to eq(2)
        out[:unscored].each do |entry|
          expect(entry[:status]).to eq(:unscored)
          expect(entry[:score]).to be_nil
          expect(entry[:components]).to eq([])
        end
      end

      it 'keeps unscored candidates out of the scored ranking' do
        out = described_class.hybrid_match('rube', ['ruble', '', 'rube'])
        expect(out[:results].map { |e| e[:value] }).to eq(%w[rube ruble])
        expect(out[:unscored].map { |e| e[:value] }).to eq([''])
      end

      it 'preserves input order among unscored entries' do
        out = described_class.hybrid_match('rube', ['', 'ruble', nil])
        expect(out[:unscored].map { |e| e[:value] }).to eq(['', ''])
        expect(out[:unscored].map { |e| e[:index] }).to eq([0, 2])
      end

      it 'still scores a whitespace-only candidate on its (zero) text similarity' do
        entry = described_class.hybrid_match('rube', ['   '])[:results].first
        expect(entry[:status]).to eq(:scored)
        expect(entry[:components]).to eq([:text])
        expect(entry[:score]).to eq(0.0)
      end
    end

    context 'sorting' do
      it 'sorts scored entries by score descending' do
        out = described_class.hybrid_match('rube', %w[ruble rube])
        scores = out[:results].map { |e| e[:score] }
        expect(scores).to eq(scores.sort.reverse)
      end

      it 'breaks ties by original input order (stable, matching the existing convention)' do
        out = described_class.hybrid_match('rube', %w[rube rube])
        expect(out[:results].map { |e| e[:index] }).to eq([0, 1])
        expect(out[:results].map { |e| e[:score] }).to eq([1.0, 1.0])
      end
    end

    context 'algorithms' do
      it 'supports metaphone as the phonetic algorithm' do
        expect(described_class.metaphone('rube')).not_to eq(described_class.metaphone('ruble'))
        entry = described_class.hybrid_match('rube', ['ruble'],
                                             phonetic_algorithm: :metaphone)[:results].first
        expect(entry[:component_scores][:phonetic]).to eq(0.0)
        expect(entry[:score]).to eq(0.48)
      end

      it 'supports jaro_winkler as the text algorithm' do
        entry = described_class.hybrid_match('rube', ['ruble'],
                                             text_algorithm: :jaro_winkler)[:results].first
        expected = (described_class.jaro_winkler('rube', 'ruble') * 0.6).round(4)
        expect(entry[:score]).to eq(expected)
      end

      it 'supports dice as the text algorithm' do
        entry = described_class.hybrid_match('rube', ['ruble'], text_algorithm: :dice)[:results].first
        expected = (described_class.dice_coefficient('rube', 'ruble') * 0.6).round(4)
        expect(entry[:score]).to eq(expected)
      end
    end

    context 'validation' do
      it 'raises Error for a negative weight' do
        expect { described_class.hybrid_match('a', ['b'], text_weight: -0.1) }
          .to raise_error(Philiprehberger::FuzzyMatch::Error, /negative/)
      end

      it 'raises Error when all weights are zero' do
        expect { described_class.hybrid_match('a', ['b'], text_weight: 0.0, phonetic_weight: 0.0) }
          .to raise_error(Philiprehberger::FuzzyMatch::Error, /positive/)
      end

      it 'raises Error for an unknown text algorithm' do
        expect { described_class.hybrid_match('a', ['b'], text_algorithm: :nope) }
          .to raise_error(Philiprehberger::FuzzyMatch::Error, /Unknown algorithm/)
      end

      it 'raises Error for an unknown phonetic algorithm' do
        expect { described_class.hybrid_match('a', ['b'], phonetic_algorithm: :nope) }
          .to raise_error(Philiprehberger::FuzzyMatch::Error, /Unknown phonetic/)
      end
    end

    it 'returns empty result sets for empty candidates' do
      expect(described_class.hybrid_match('anything', []))
        .to eq(results: [], unscored: [], weights: { text: 0.6, phonetic: 0.4 })
    end
  end

  describe '.weighted_score' do
    it 'keeps the fully-valued weighted behavior unchanged' do
      score = described_class.weighted_score(
        'rube', 'ruble', weights: { levenshtein_ratio: 0.6, dice: 0.4 }
      )
      manual = (described_class.ratio('rube', 'ruble') * 0.6) +
               (described_class.dice_coefficient('rube', 'ruble') * 0.4)
      expect(score).to eq(manual)
    end
  end
end
