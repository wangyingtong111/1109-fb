# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'HTML document type' do
  it 'segments paragraphs and keeps inline formatting as running text' do
    html = '<p>Dose&nbsp;<em>5.50</em></p><p>Next sample.</p><p>&lt;ok&gt;</p>'
    ps = PragmaticSegmenter::Segmenter.new(text: html, language: 'en', doc_type: 'html')
    expect(ps.segment).to eq(['Dose 5.50', 'Next sample.', '<ok>'])
  end

  it 'does not break sentences at inline tags' do
    html = '<p>This is <em>one sentence</em> with inline tags.</p>'
    ps = PragmaticSegmenter::Segmenter.new(text: html, language: 'en', doc_type: 'html')
    expect(ps.segment).to eq(['This is one sentence with inline tags.'])
  end

  it 'treats block-level elements as sentence boundaries' do
    html = '<div>First block</div><div>Second block</div>'
    ps = PragmaticSegmenter::Segmenter.new(text: html, language: 'en', doc_type: 'html')
    expect(ps.segment).to eq(['First block', 'Second block'])
  end

  it 'decodes entities into literal text' do
    html = '<p>Tom &amp; Jerry &lt;3 cartoons&nbsp;daily.</p>'
    ps = PragmaticSegmenter::Segmenter.new(text: html, language: 'en', doc_type: 'html')
    expect(ps.segment).to eq(['Tom & Jerry <3 cartoons daily.'])
  end

  it 'strips script and style content' do
    html = '<p>Visible text.</p><script>var x = 1.5;</script><style>.a { color: red; }</style>'
    ps = PragmaticSegmenter::Segmenter.new(text: html, language: 'en', doc_type: 'html')
    expect(ps.segment).to eq(['Visible text.'])
  end

  it 'does not convert HTML in the default plain text mode' do
    html = '<p>Dose&nbsp;<em>5.50</em></p><p>Next sample.</p><p>&lt;ok&gt;</p>'
    ps = PragmaticSegmenter::Segmenter.new(text: html, language: 'en')
    expect(ps.segment).to eq(['Dose&nbsp;5.50Next sample.'])
  end
end

RSpec.describe PragmaticSegmenter::HTMLConverter do
  it 'converts block tags to newlines and removes inline tags' do
    expect(described_class.to_text('<p>a <b>bold</b> word</p><p>next</p>'))
      .to eq("\na bold word\n\nnext\n")
  end

  it 'decodes named and numeric entities' do
    expect(described_class.to_text('&lt;a&gt; &amp; &#65;&#x42;&nbsp;c'))
      .to eq('<a> & AB c')
  end

  it 'leaves unknown entities untouched' do
    expect(described_class.to_text('&notanentity;')).to eq('&notanentity;')
  end
end
