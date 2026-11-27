class ApprovalForm
  include AASM

  attr_reader :guard_args, :callback_args

  aasm do
    state :pending, :initial => true
    state :quick
    state :manual
    state :approved
    state :submitted

    event :review do
      transitions :from => :pending, :to => :quick,
                  :guard => :record_guard, :after => :record_callback
      transitions :from => :pending, :to => :manual,
                  :guard => :record_guard, :after => :record_callback
      transitions :from => :manual,  :to => :approved,
                  :guard => :record_guard, :after => :record_callback
    end

    event :submit do
      transitions :from => :pending, :to => :submitted,
                  :guard => :record_guard, :after => :record_callback
    end
  end

  def initialize
    @guard_args = []
    @callback_args = []
  end

  def record_guard(*args)
    @guard_args << args
    true
  end

  def record_callback(*args)
    @callback_args << args
  end
end
