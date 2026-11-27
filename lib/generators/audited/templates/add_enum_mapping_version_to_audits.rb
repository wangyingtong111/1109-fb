# frozen_string_literal: true

class <%= migration_class_name %> < <%= migration_parent %>
  def self.up
    add_column :audits, :enum_mapping_version, :string
  end

  def self.down
    remove_column :audits, :enum_mapping_version
  end
end
