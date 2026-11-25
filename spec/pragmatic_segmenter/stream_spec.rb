# -*- encoding : utf-8 -*-
# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PragmaticSegmenter::Stream do
  it 'emits a one-sentence stream only on finish' do
    stream = described_class.new

    expect(stream.push('Hello.')).to eq([])
    expect(stream.sentences).to eq([])
    expect(stream.finish).to eq(['Hello.'])
    expect(stream.sentences).to eq(['Hello.'])
  end

  it 'keeps punctuation split across chunks unconfirmed until the next chunk confirms it' do
    stream = described_class.new

    expect(stream.push('Wait..')).to eq([])
    expect(stream.push(". we're here.")).to eq([])
    expect(stream.push(' Then go.')).to eq(["Wait... we're here."])
    expect(stream.finish).to eq(['Then go.'])
    expect(stream.sentences).to eq(["Wait... we're here.", 'Then go.'])
  end

  it 'does not let empty chunks advance a boundary' do
    stream = described_class.new
    stream.push('Hello.')

    expect(stream.push('')).to eq([])
    expect(stream.push(nil)).to eq([])
    expect(stream.finish).to eq(['Hello.'])
  end

  it 'flushes the tail only once' do
    stream = described_class.new
    stream.push('Hello.')

    expect(stream.finish).to eq(['Hello.'])
    expect(stream.finish).to eq([])
    expect(stream.flush).to eq([])
  end

  it 'accepts finish as an event and supports block listeners' do
    published = []
    stream = described_class.new { |s| published << s }
    stream.push('Hello.')
    stream.push(:finish)

    expect(published).to eq(['Hello.'])
    expect(stream.closed?).to be(true)
  end

  it 'is available through Segmenter and common aliases' do
    stream = PragmaticSegmenter::Segmenter.stream
    expect(stream).to be_a(described_class)

    segmenter = PragmaticSegmenter::Segmenter.new(text: '')
    expect(segmenter.feed('Hello.')).to eq([])
    expect(segmenter.flush).to eq(['Hello.'])

    expect(PragmaticSegmenter::StreamingSegmenter.new).to be_a(described_class)
    expect(PragmaticSegmenter::Segmenter::Stream.new).to be_a(described_class)
  end

  it 'can process finish-marked event arrays directly' do
    events = ['Wait..', ". we're here.", ' Then go.', :finish]

    expect(PragmaticSegmenter::Segmenter.stream_events(events)).to eq(
      [[], [], ["Wait... we're here."], ['Then go.']]
    )
  end
end
