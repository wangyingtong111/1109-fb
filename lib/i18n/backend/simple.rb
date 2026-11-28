# frozen_string_literal: true

require 'i18n/backend/base'

module I18n
  module Backend
    # A simple backend that reads translations from YAML files and stores them in
    # an in-memory hash. Relies on the Base backend.
    #
    # The implementation is provided by a Implementation module allowing to easily
    # extend Simple backend's behavior by including modules. E.g.:
    #
    #     module I18n::Backend::Pluralization
    #       def pluralize(*args)
    #         # extended pluralization logic
    #         super
    #       end
    #     end
    #     
    #     I18n::Backend::Simple.include(I18n::Backend::Pluralization)
    class Simple
      module Implementation
        include Base

        # Mutex to ensure that concurrent translations loading will be thread-safe
        MUTEX = Mutex.new

        # Options accepted by #store_translations to select the namespace(s)
        # that should be replaced instead of deep merged into.
        REPLACE_OPTION = :replace

        def initialized?
          @initialized ||= false
        end

        # Stores translations for the given locale in memory.
        # This uses a deep merge for the translations hash, so existing
        # translations will be overwritten by new ones only at the deepest
        # level of the hash.
        #
        # When the :replace option is given the listed namespaces are replaced
        # as a whole instead of being deep merged. That means keys belonging to
        # the namespace which are missing from +data+ are removed from the
        # current layer, so deleted translations really disappear:
        #
        #   store_translations(:en, { :notice => { :ready => 'Ready' } }, :replace => :notice)
        #
        # Translations of all other namespaces are left untouched (and are
        # still deep merged), so re-uploading a single dictionary does not
        # clobber the rest of the locale.
        #
        # :replace accepts a single namespace (String or Symbol; dots address a
        # nested namespace), an Array of namespaces or +true+ to replace every
        # top level namespace contained in +data+:
        #
        #   store_translations(:en, data, :replace => [:notice, :billing])
        #   store_translations(:en, data, :replace => true)
        #
        # Replacing is atomic: if storing raises, the previously stored
        # translations of the locale are left untouched, and concurrent lookups
        # see either the previous or the new version of the namespaces, never a
        # partially applied one.
        def store_translations(locale, data, options = EMPTY_HASH)
          if I18n.enforce_available_locales &&
            I18n.available_locales_initialized? &&
            !I18n.locale_available?(locale)
            return data
          end
          locale = locale.to_sym

          if (replace = replace_option(options))
            replace_translations(locale, data, replace, options)
          else
            translations[locale] ||= Concurrent::Hash.new
            data = Utils.deep_symbolize_keys(data) unless options.fetch(:skip_symbolize_keys, false)
            MUTEX.synchronize { Utils.deep_merge!(translations[locale], data) }
          end
        end

        # Get available locales from the translations hash
        def available_locales
          init_translations unless initialized?
          translations.inject([]) do |locales, (locale, data)|
            locales << locale unless data.size <= 1 && (data.empty? || data.has_key?(:i18n))
            locales
          end
        end

        # Clean up translations hash and set initialized to false on reload!
        def reload!
          @initialized = false
          @translations = nil
          super
        end

        def eager_load!
          init_translations unless initialized?
          super
        end

        def translations(do_init: false)
          # To avoid returning empty translations,
          # call `init_translations`
          init_translations if do_init && !initialized?

          @translations ||= Concurrent::Hash.new do |h, k|
            MUTEX.synchronize do
              h[k] = Concurrent::Hash.new
            end
          end
        end

      protected

        def init_translations
          load_translations
          @initialized = true
        end

        # Returns the normalized list of namespaces to replace for the given
        # +store_translations+ options, or nil when the translations should be
        # deep merged (the default behavior).
        def replace_option(options)
          return nil unless options.is_a?(Hash) && options.key?(REPLACE_OPTION)

          value = options[REPLACE_OPTION]
          return nil if value.nil? || value == false
          return true if value == true

          namespaces = Array(value).flatten.compact.map do |namespace|
            normalize_namespace(namespace, options)
          end
          namespaces.reject!(&:empty?)
          namespaces.empty? ? nil : namespaces
        end

        # Splits a namespace given to the :replace option into its keys, e.g.
        # :'notice.old' becomes ['notice', 'old']. Keys are compared as strings
        # so that symbolized and non-symbolized data behave the same way.
        def normalize_namespace(namespace, options)
          I18n.normalize_keys(nil, namespace, nil, options[:separator]).map(&:to_s)
        end

        # Stores +data+ for +locale+ while replacing the given +namespaces+ as a
        # whole: keys missing from +data+ are dropped from the current layer
        # instead of being kept alive by the regular deep merge. Everything
        # outside of the replaced namespaces is still deep merged.
        #
        # The new translations hash is built aside and only swapped in once it
        # is complete. A failure while importing therefore leaves the
        # previously stored translations untouched, and readers never observe a
        # partially applied update.
        def replace_translations(locale, data, namespaces, options)
          unless data.is_a?(Hash)
            raise InvalidLocaleData.new(locale, 'expects it to return a hash, but does not')
          end
          data = Utils.deep_symbolize_keys(data) unless options.fetch(:skip_symbolize_keys, false)

          namespaces = data.keys.map { |key| [key.to_s] } if namespaces == true

          # Make sure the bucket exists. The default proc of #translations
          # acquires MUTEX, so this has to happen before synchronizing.
          translations[locale]

          MUTEX.synchronize do
            updated = Concurrent::Hash.new
            deep_dup(translations[locale]).each { |key, value| updated[key] = value }
            namespaces.each { |namespace| delete_path(updated, namespace) }
            Utils.deep_merge!(updated, data)
            translations[locale] = updated
          end

          translations[locale]
        end

        # Removes the value stored at +path+ (if any) from +hash+.
        def delete_path(hash, path)
          *parents, key = path
          parent = parents.inject(hash) do |node, part|
            found, value = fetch_key(node, part)
            return unless found && value.is_a?(Hash)
            value
          end

          found, = fetch_key(parent, key)
          parent.delete(actual_key(parent, key)) if found
        end

        # Looks a key up supporting the string/symbol forms used across the
        # various backends (and numeric keys stored as integers). Returns
        # [found, value].
        def fetch_key(hash, key)
          stored = actual_key(hash, key)
          hash.key?(stored) ? [true, hash[stored]] : [false, nil]
        end

        # Returns the key actually stored in +hash+ for +key+, or +key+ itself
        # when it is not present.
        def actual_key(hash, key)
          return key if hash.key?(key)
          hash.keys.find { |candidate| candidate.to_s == key.to_s } || key
        end

        def deep_dup(value)
          case value
          when Hash
            value.each_with_object({}) { |(key, val), result| result[key] = deep_dup(val) }
          when Array
            value.map { |element| deep_dup(element) }
          else
            value
          end
        end

        # Looks up a translation from the translations hash. Returns nil if
        # either key is nil, or locale, scope or key do not exist as a key in the
        # nested translations hash. Splits keys or scopes containing dots
        # into multiple keys, i.e. <tt>currency.format</tt> is regarded the same as
        # <tt>%w(currency format)</tt>.
        def lookup(locale, key, scope = [], options = EMPTY_HASH)
          init_translations unless initialized?
          keys = I18n.normalize_keys(locale, key, scope, options[:separator])

          keys.inject(translations) do |result, _key|
            return nil unless result.is_a?(Hash)
            unless result.has_key?(_key)
              _key = _key.to_s.to_sym
              return nil unless result.has_key?(_key)
            end
            result = result[_key]
            result = resolve_entry(locale, _key, result, Utils.except(options.merge(:scope => nil), :count)) if result.is_a?(Symbol)
            result
          end
        end
      end

      include Implementation
    end
  end
end
