# frozen_string_literal: true

module Grape
  module Util
    # Picks the media type an Accept header asks for out of the ones the API
    # declared. The declaration order is the API's business preference:
    # Grape::ContentTypes keeps it and every +*/*+ a browser sends already
    # answers with the first type declared, so a header that names two types
    # without choosing between them -- equal quality, including an omitted q
    # (RFC 9110 §12.5.1) -- honors the same preference instead of the order
    # the client happened to list them in. A quality the client did set still
    # wins, and a quality of zero is a refusal: a wildcard that names nothing
    # in its own right never brings a refused type back.
    #
    # Rack::Utils.best_q_match answers neither: it sorts the matches and keeps
    # the last, so a tie is broken by the order the *header* lists the types,
    # and a specific type refused with +q=0+ is re-matched by a +*/*+ that
    # follows it.
    module AcceptNegotiation
      class << self
        # @param accept_header [String, nil] the scrubbed Accept header
        # @param available_media_types [Array<String>] the media types the API
        #   declared, in declaration order
        # @return [String, nil] the best match, or nil when the header names
        #   only types the API did not declare or refused every one of them
        def best_media_type(accept_header, available_media_types)
          return if accept_header.nil? || accept_header.empty?

          accepted = Rack::Utils.q_values(accept_header.downcase)
          best = available_media_types.inject(nil) { |chosen, media_type| prefer(chosen, media_type, accepted) }
          best&.first
        end

        private

        # The chosen pair +[media_type, quality]+ replaced by +media_type+ when
        # the header accepts it with a strictly better quality, else unchanged.
        # Iterating the declared types in order makes the first declared win a
        # tie, and skips a refused type (+q=0+) altogether.
        def prefer(chosen, media_type, accepted)
          quality = quality_for(media_type, accepted)
          return chosen if quality.nil? || quality.zero?
          return chosen if chosen && chosen.last >= quality

          [media_type, quality]
        end

        # The highest quality any accepted entry assigns to +media_type+, or
        # nil when none matches it. A more specific entry outranks a wildcard
        # that covers the same type, as RFC 9110 §12.5.3 requires.
        def quality_for(media_type, accepted)
          accepted.filter_map do |accepted_type, quality|
            next unless Rack::Mime.match?(media_type, accepted_type)

            [quality, specificity(accepted_type)]
          end.min_by(&:last)&.first
        end

        # Fewer wildcards name a type more precisely: a concrete type outranks
        # a subtype wildcard, which outranks +*/*+.
        def specificity(media_type)
          media_type.count('*')
        end
      end
    end
  end
end
