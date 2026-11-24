# frozen_string_literal: true

require 'spec_helper'

# Counting, asymmetric scorer double: keeps oak -> elm (0.9) but
# filters elm -> oak (0.1) at threshold 0.5. Declares itself reusable
# (ordered pairs may be cached) but NOT symmetric.
class CountingAsymmetricScorer
  attr_reader :calls

  def initialize
    @calls = []
  end

  def reusable?
    true
  end

  def symmetric?
    false
  end

  def call(text_a, text_b)
    @calls << [text_a, text_b]
    return 1.0 if text_a == text_b
    return 0.9 if text_a == 'oak' && text_b == 'elm'
    return 0.1 if text_a == 'elm' && text_b == 'oak'

    0.0
  end
end

RSpec.describe Philiprehberger::FuzzyMatch do
  describe '.similarity_matrix_by_id' do
    let(:scorer) { CountingAsymmetricScorer.new }
    let(:records) do
      [
        { id: 1, name: 'oak' },
        { id: 2, name: 'oak' },
        { id: 3, name: 'elm' }
      ]
    end
    let(:matrix) do
      described_class.similarity_matrix_by_id(records, id_key: :id, text_key: :name,
                                                       scorer: scorer, threshold: 0.5)
    end

    it 'keeps a separate row and column for each record even when names collide' do
      expect(matrix.keys).to contain_exactly(1, 2, 3)
      expect(matrix[1].keys).to contain_exactly(1, 2, 3)
      expect(matrix[2].keys).to contain_exactly(1, 2, 3)
      expect(matrix[3].keys).to contain_exactly(1, 2, 3)
    end

    it 'keeps the oak -> elm edge above threshold in both oak rows' do
      expect(matrix[1][3]).to eq(0.9)
      expect(matrix[2][3]).to eq(0.9)
    end

    it 'stores below-threshold edges as nil so they differ from unknown IDs' do
      expect(matrix[3]).to have_key(1)
      expect(matrix[3]).to have_key(2)
      expect(matrix[3][1]).to be_nil
      expect(matrix[3][2]).to be_nil
      expect(matrix).not_to have_key(99)
      expect { matrix.fetch(99) }.to raise_error(KeyError)
    end

    it 'scores identical texts on the diagonal and between same-name records' do
      expect(matrix[1][1]).to eq(1.0)
      expect(matrix[1][2]).to eq(1.0)
      expect(matrix[2][1]).to eq(1.0)
    end

    it 'does not recompute the same ordered text pair within one call' do
      matrix
      expect(scorer.calls.count(%w[oak elm])).to eq(1)
      expect(scorer.calls.count(%w[oak oak])).to eq(1)
      expect(scorer.calls.count(%w[elm elm])).to eq(1)
    end

    it 'computes the reverse direction separately for non-symmetric scorers' do
      matrix
      expect(scorer.calls.count(%w[elm oak])).to eq(1)
    end

    it 'reuses both directions when the scorer declares itself symmetric' do
      symmetric_scorer = Class.new(CountingAsymmetricScorer) do
        def symmetric?
          true
        end
      end.new

      described_class.similarity_matrix_by_id(records, id_key: :id, text_key: :name,
                                                       scorer: symmetric_scorer, threshold: 0.5)
      expect(symmetric_scorer.calls.count(%w[elm oak])).to eq(0)
      expect(symmetric_scorer.calls.count(%w[oak elm])).to eq(1)
    end

    it 'evaluates every ordered pair when the scorer is not reusable' do
      plain_scorer = Class.new do
        attr_reader :calls

        def initialize
          @calls = []
        end

        def call(_a, _b)
          @calls << true
          0.7
        end
      end.new

      described_class.similarity_matrix_by_id(records, id_key: :id, text_key: :name,
                                                       scorer: plain_scorer)
      expect(plain_scorer.calls.length).to eq(9)
    end

    it 'works with the built-in algorithm when no scorer is given' do
      result = described_class.similarity_matrix_by_id(records, id_key: :id, text_key: :name)
      expected = described_class.jaro_winkler('oak', 'elm').round(4)
      expect(result[1][3]).to eq(expected)
      expect(result[2][3]).to eq(expected)
      expect(result[1][2]).to eq(1.0)
    end

    it 'accepts string hash keys' do
      string_records = [{ 'id' => 'a', 'name' => 'oak' }, { 'id' => 'b', 'name' => 'elm' }]
      result = described_class.similarity_matrix_by_id(string_records, id_key: :id, text_key: :name)
      expect(result.keys).to contain_exactly('a', 'b')
    end

    it 'accepts objects responding to the field names' do
      record_class = Struct.new(:id, :name)
      object_records = [record_class.new(1, 'oak'), record_class.new(2, 'elm')]
      result = described_class.similarity_matrix_by_id(object_records, id_key: :id, text_key: :name)
      expect(result.keys).to contain_exactly(1, 2)
    end

    it 'raises on duplicate record IDs' do
      dup_records = [{ id: 1, name: 'oak' }, { id: 1, name: 'elm' }]
      expect do
        described_class.similarity_matrix_by_id(dup_records, id_key: :id, text_key: :name)
      end.to raise_error(Philiprehberger::FuzzyMatch::Error, /Duplicate record IDs/)
    end

    it 'returns an empty hash for empty input' do
      expect(described_class.similarity_matrix_by_id([], id_key: :id, text_key: :name)).to eq({})
    end
  end

  describe '.similarity_matrix (legacy string entry point)' do
    it 'still returns a hash-of-hashes keyed by the strings themselves' do
      matrix = described_class.similarity_matrix(%w[hello helo world])
      expect(matrix.keys).to eq(%w[hello helo world])
      expect(matrix['hello']['hello']).to eq(1.0)
    end

    it 'still omits pairs below the threshold' do
      matrix = described_class.similarity_matrix(%w[hello world], threshold: 0.9)
      expect(matrix['hello']).not_to have_key('world')
    end
  end
end
