require 'spec_helper'

describe FuzzyMatch::Rule::Identity do
  it %{determines whether two records COULD be identical} do
    i = FuzzyMatch::Rule::Identity.new %r{(A)[ ]*(\d)}
    i.identical?(r('A1'), r('A     1foobar')).should == true
  end
  
  it %{determines that two records MUST NOT be identical} do
    i = FuzzyMatch::Rule::Identity.new %r{(A)[ ]*(\d)}
    i.identical?(r('A1'), r('A     2foobar')).should == false
  end
  
  it %{returns nil indicating no information} do
    i = FuzzyMatch::Rule::Identity.new %r{(A)[ ]*(\d)}
    i.identical?(r('B1'), r('A     2foobar')).should == nil
  end
  
  it %{embraces case insensitivity} do
    i = FuzzyMatch::Rule::Identity.new %r{(A)[ ]*(\d)}i
    i.identical?(r('A1'), r('a     1foobar')).should == true
  end

  describe 'several matches in one string' do
    let(:identity) { FuzzyMatch::Rule::Identity.new %r{sku=([a-z0-9-]+)(\{\})?}i }

    it %{confirms identity when one complete capture combination is shared} do
      # the first title lists the current and previous SKU, the other title only lists the previous one
      identity.identical?(r('sku=ab-12 sku=ac1-2{}'), r('sku=AC1-2{}')).should == true
    end

    it %{does not confirm identity when none of the complete combinations are shared} do
      identity.identical?(r('sku=ab-12 sku=ac1-2{}'), r('sku=ab1-2{}')).should == false
    end

    it %{treat a present-but-empty capture as different from a missing capture} do
      identity.identical?(r('sku=ab-12'), r('sku=ab-12{}')).should == false
    end

    it %{never recombines captures taken from different matches} do
      i = FuzzyMatch::Rule::Identity.new %r{(x|y)(\d)?}
      # "x" (missing digit) and "y1" appear in both strings, but never as the
      # same complete combination, and the "x"/"1" pieces must not be joined
      i.identical?(r('x y1'), r('x y1')).should == true
      i.identical?(r('x y1'), r('x1 y')).should == false
    end
  end

  it %{keeps an empty-string capture distinct from a group that never matched} do
    # the second group is either captured as an empty string or not captured at all
    i = FuzzyMatch::Rule::Identity.new %r{\A(a)(?:(\d)|())?\z}
    i.identical?(r('a'), r('a')).should == true
    i.identical?(r('a1'), r('a1')).should == true
    i.identical?(r('a'), r('a1')).should == false
  end

  it %{returns nil when only one of the strings matches the regexp} do
    i = FuzzyMatch::Rule::Identity.new %r{sku=([a-z0-9-]+)}i
    i.identical?(r('sku=ab-12'), r('no sku here')).should == nil
  end

  private

  def r(str)
    FuzzyMatch::Record.new str
  end
end
