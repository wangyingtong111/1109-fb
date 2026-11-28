# frozen_string_literal: true

RSpec.describe Grape::Util::AcceptNegotiation do
  describe '.best_media_type' do
    subject(:media_type) { described_class.best_media_type(accept_header, available_media_types) }

    let(:available_media_types) { %w[application/json text/plain] }

    context 'when the header is blank' do
      %w[nil empty].each do |kind|
        let(:accept_header) { kind == 'nil' ? nil : '' }

        it "returns nil for a #{kind} header" do
          expect(media_type).to be_nil
        end
      end
    end

    context 'when the header names one available type' do
      let(:accept_header) { 'text/plain' }

      it { is_expected.to eq('text/plain') }
    end

    context 'when the header is not in lower case' do
      let(:accept_header) { 'TEXT/PLAIN' }

      it { is_expected.to eq('text/plain') }
    end

    context 'when two types tie on quality' do
      it 'keeps the type the API declared first, regardless of header order' do
        expect(described_class.best_media_type('application/json, text/plain', available_media_types)).to eq('application/json')
        expect(described_class.best_media_type('text/plain, application/json', available_media_types)).to eq('application/json')
      end

      it 'treats an omitted quality as 1.0 against an explicit one' do
        expect(described_class.best_media_type('text/plain; q=1.0, application/json', available_media_types)).to eq('application/json')
      end

      it 'flips with the order the API declares them' do
        reversed = %w[text/plain application/json]
        expect(described_class.best_media_type('application/json, text/plain', reversed)).to eq('text/plain')
      end
    end

    context 'when the client sets different qualities' do
      it 'prefers the higher quality' do
        expect(described_class.best_media_type('application/json;q=0.5, text/plain;q=0.9', available_media_types)).to eq('text/plain')
        expect(described_class.best_media_type('application/json;q=0.9, text/plain;q=0.5', available_media_types)).to eq('application/json')
      end
    end

    context 'when a type is refused with a quality of zero' do
      it 'serves the other type' do
        expect(described_class.best_media_type('application/json;q=0, text/plain', available_media_types)).to eq('text/plain')
      end

      it 'does not revive the refused type through a wildcard' do
        expect(described_class.best_media_type('application/json;q=0, */*', available_media_types)).to eq('text/plain')
        expect(described_class.best_media_type('application/json;q=0, application/*, text/plain', available_media_types)).to eq('text/plain')
      end

      it 'returns nil when every available type is refused' do
        expect(described_class.best_media_type('application/json;q=0, text/plain;q=0', available_media_types)).to be_nil
      end

      it 'lets a concrete acceptance outrank a wildcard refusal that also covers it' do
        expect(described_class.best_media_type('*/*;q=0, application/json', available_media_types)).to eq('application/json')
      end
    end

    context 'with a wildcard' do
      it 'matches the first declared type on */*' do
        expect(described_class.best_media_type('*/*', available_media_types)).to eq('application/json')
      end

      it 'matches a subtype wildcard' do
        expect(described_class.best_media_type('text/*', available_media_types)).to eq('text/plain')
        expect(described_class.best_media_type('application/*', available_media_types)).to eq('application/json')
      end
    end

    context 'when the header names no available type' do
      let(:accept_header) { 'text/html' }

      it { is_expected.to be_nil }
    end

    context 'with vendored media types' do
      let(:available_media_types) { %w[application/vnd.acme-v1+json text/plain] }

      it 'matches the vendored type' do
        expect(described_class.best_media_type('application/vnd.acme-v1+json', available_media_types)).to eq('application/vnd.acme-v1+json')
      end

      it 'breaks a tie by declaration order' do
        expect(described_class.best_media_type('text/plain, application/vnd.acme-v1+json', available_media_types)).to eq('application/vnd.acme-v1+json')
      end
    end
  end
end
