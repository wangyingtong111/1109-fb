require 'spec_helper'

describe 'explicit target states of an event' do
  let(:form) { ApprovalForm.new }

  context 'review with quick and manual transitions' do
    it 'accepts a string as explicit target and forwards only the business args' do
      form.review!('manual', 'R7')

      expect(form.aasm.current_state).to eq(:manual)
      expect(form.guard_args).to eq([['R7']])
      expect(form.callback_args).to eq([['R7']])
    end

    it 'accepts a symbol as explicit target and forwards only the business args' do
      form.review!(:manual, 'R7')

      expect(form.aasm.current_state).to eq(:manual)
      expect(form.guard_args).to eq([['R7']])
      expect(form.callback_args).to eq([['R7']])
    end

    it 'treats string and symbol targets alike when preflighting' do
      expect(form.may_review?('manual', 'R7')).to be true
      expect(form.may_review?(:manual, 'R7')).to be true
    end

    it 'keeps the original transition order when no target is given' do
      form.review!('R7')

      expect(form.aasm.current_state).to eq(:quick)
      expect(form.guard_args).to eq([['R7']])
      expect(form.callback_args).to eq([['R7']])
    end

    it 'does not swallow old-style business strings' do
      form.submit!('manual')

      expect(form.aasm.current_state).to eq(:submitted)
      expect(form.guard_args).to eq([['manual']])
      expect(form.callback_args).to eq([['manual']])
    end

    it 'keeps business strings behind a non-target first argument' do
      form.review!('R7', 'manual')

      expect(form.aasm.current_state).to eq(:quick)
      expect(form.guard_args).to eq([['R7', 'manual']])
      expect(form.callback_args).to eq([['R7', 'manual']])
    end

    context 'with an explicit target that is not reachable from the current state' do
      it 'rejects it when preflighting, for strings and symbols alike' do
        expect(form.may_review?(:approved)).to be false
        expect(form.may_review?('approved')).to be false
      end

      it 'does not run another transition when committing' do
        expect { form.review!(:approved) }.to raise_error(AASM::InvalidTransition)

        expect(form.aasm.current_state).to eq(:pending)
        expect(form.guard_args).to be_empty
        expect(form.callback_args).to be_empty
      end

      it 'judges it the same way without persistence' do
        expect(form.may_review?(:approved)).to be false
        expect { form.review(:approved) }.to raise_error(AASM::InvalidTransition)
        expect(form.aasm.current_state).to eq(:pending)
      end
    end
  end
end
