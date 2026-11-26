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

  describe '#status' do
    it %{reports :verified when both match and the captures agree} do
      i = FuzzyMatch::Rule::Identity.new %r{(A)[ ]*(\d)}
      i.status(r('A1'), r('A     1foobar')).should == :verified
    end

    it %{reports :conflict when both match but the captures disagree} do
      i = FuzzyMatch::Rule::Identity.new %r{(A)[ ]*(\d)}
      i.status(r('A1'), r('A     2foobar')).should == :conflict
    end

    it %{reports :unknown when either side does not match} do
      i = FuzzyMatch::Rule::Identity.new %r{(A)[ ]*(\d)}
      i.status(r('B1'), r('A     2foobar')).should == :unknown
    end

    it %{translates proc answers into the same three states} do
      i = FuzzyMatch::Rule::Identity.new lambda { |a, b| a[0] == b[0] ? true : false }
      i.status(r('abc'), r('ax')).should == :verified
      i.status(r('abc'), r('bx')).should == :conflict
      n = FuzzyMatch::Rule::Identity.new lambda { |_a, _b| nil }
      n.status(r('abc'), r('bx')).should == :unknown
    end
  end

  private

  def r(str)
    FuzzyMatch::Record.new str
  end
end
