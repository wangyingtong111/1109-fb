# -*- encoding: utf-8 -*-
require 'spec_helper'

describe FuzzyMatch do
  describe 'the :cross_encoding rule (explicit cross-encoding text mode)' do
    let(:utf8)    { "café".encode(::Encoding::UTF_8) }
    let(:latin1)  { "café".encode(::Encoding::ISO_8859_1) }
    let(:utf16le) { "café".encode(::Encoding::UTF_16LE) }

    it %{is off by default, preserving previous (binary-friendly) behavior} do
      d = FuzzyMatch.new [latin1]
      d.cross_encoding.should == false
      # without the mode, scoring is byte-based, so the same word stored in
      # a different encoding does NOT get a full score
      d.find utf8, :gather_last_result => true
      d.last_result.score.should < 1.0
    end

    it %{transcodes records for comparison but keeps the original (and its encoding) untouched} do
      r = FuzzyMatch::Record.new latin1, :cross_encoding => true
      r.whole.should == utf8
      r.whole.encoding.should == ::Encoding::UTF_8
      r.original.should == latin1
      r.original.encoding.should == ::Encoding::ISO_8859_1
    end

    it %{matches a UTF-8 needle against a Latin-1 record by actual characters, returning the original record and encoding} do
      d = FuzzyMatch.new [latin1, 'decoy'], :cross_encoding => true, :must_match_at_least_one_word => true
      winner = d.find(utf8)
      winner.should == latin1
      winner.encoding.should == ::Encoding::ISO_8859_1
    end

    it %{matches a UTF-8 needle against a UTF-16LE record by actual characters} do
      d = FuzzyMatch.new [utf16le, 'decoy'], :cross_encoding => true, :must_match_at_least_one_word => true
      winner = d.find(utf8)
      winner.should == utf16le
      winner.encoding.should == ::Encoding::UTF_16LE
    end

    it %{works with the encodings of query and target swapped (Latin-1 needle, UTF-8 record)} do
      d = FuzzyMatch.new [utf8, 'decoy'], :cross_encoding => true, :must_match_at_least_one_word => true
      winner = d.find(latin1)
      winner.should == utf8
      winner.encoding.should == ::Encoding::UTF_8
    end

    it %{works with a UTF-16LE needle against a UTF-8 record} do
      d = FuzzyMatch.new [utf8, 'decoy'], :cross_encoding => true, :must_match_at_least_one_word => true
      d.find(utf16le).should == utf8
    end

    it %{passes the word filter thanks to shared actual characters, not shared bytes} do
      # "decoy" shares no word with the needle, so it is filtered out
      d = FuzzyMatch.new ['decoy'], :cross_encoding => true, :must_match_at_least_one_word => true
      d.find(utf8).should be_nil
      # while the latin1/utf-16le stored twin of the needle passes the filter
      d = FuzzyMatch.new ['decoy', latin1, utf16le], :cross_encoding => true, :must_match_at_least_one_word => true
      d.find_all(utf8).should == [latin1, utf16le]
    end

    it %{gives full scores to the same word stored in different encodings} do
      d = FuzzyMatch.new [latin1, utf16le], :cross_encoding => true, :must_match_at_least_one_word => true
      d.find_with_score(utf8).should == [latin1, 1.0, 1.0]
      d.find_all_with_score(utf8).should == [[latin1, 1.0, 1.0], [utf16le, 1.0, 1.0]]
    end

    it %{keeps full scores for normal UTF-8 records} do
      d = FuzzyMatch.new [utf8], :cross_encoding => true, :must_match_at_least_one_word => true
      d.find_with_score(utf8).should == [utf8, 1.0, 1.0]
    end

    it %{raises a clear error on undecodable records} do
      binary = "\x63\x61\x66\xe9".dup.force_encoding(::Encoding::ASCII_8BIT) # Latin-1 bytes tagged as binary
      d = FuzzyMatch.new [binary], :cross_encoding => true
      lambda { d.find(utf8) }.should raise_error(FuzzyMatch::UndecodableError, /cross-encoding/)
    end

    it %{raises a clear error on undecodable needles} do
      broken = "\x63\x61\x66".dup.force_encoding(::Encoding::UTF_16LE) # odd byte count: invalid UTF-16LE
      d = FuzzyMatch.new [utf8], :cross_encoding => true
      lambda { d.find(broken) }.should raise_error(FuzzyMatch::UndecodableError)
    end

    it %{still allows binary usage when the mode is not enabled} do
      binary1 = "\x63\x61\x66\xe9".dup.force_encoding(::Encoding::ASCII_8BIT)
      binary2 = "\x63\x61\x66\xe9".dup.force_encoding(::Encoding::ASCII_8BIT)
      d = FuzzyMatch.new [binary1]
      d.find(binary2).should == binary1
    end
  end
end
