import decimal
from decimal import Decimal

import pytest
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.test import APIRequestFactory
from rest_framework.views import APIView


class PreRoundingDecimalField(serializers.DecimalField):
    """
    Explicit read-in strategy for decimal amounts.

    Incoming values are quantized with the declared ``rounding`` mode
    *before* the storable-range (max_digits / decimal_places) precision
    check runs, instead of DRF's default strict order (precision check
    on the raw input, quantize afterwards). As a result the value that
    passes validation is the value that will actually be written, and
    field validators, ``max_value``/``min_value`` checks and serializer
    validation all observe that same normalized value.

    Output behavior (``to_representation``) is untouched.
    """

    def validate_precision(self, value):
        try:
            value = self.quantize(value)
        except decimal.InvalidOperation:
            # Rounding pushed the value past what the column can store
            # (e.g. 99.995 -> 100.00 with max_digits=4).
            self.fail('max_digits', max_digits=self.max_digits)
        return super().validate_precision(value)


# Records every value the field-level validator receives.
seen_by_amount_validator = []
# Records every value the serializer-level validation receives.
seen_by_serializer = []


def recording_amount_validator(value):
    seen_by_amount_validator.append(value)


BUSINESS_MAX_AMOUNT = Decimal('50.00')


class QuoteSerializer(serializers.Serializer):
    amount = PreRoundingDecimalField(
        max_digits=4,
        decimal_places=2,
        rounding=decimal.ROUND_HALF_UP,
        max_value=BUSINESS_MAX_AMOUNT,
        validators=[recording_amount_validator],
    )

    def validate_amount(self, value):
        seen_by_serializer.append(value)
        return value


class QuoteView(APIView):
    def post(self, request):
        serializer = QuoteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.validated_data, status=status.HTTP_201_CREATED)


@pytest.fixture(autouse=True)
def clear_recorders():
    seen_by_amount_validator.clear()
    seen_by_serializer.clear()


class TestPreRoundingDecimalField:
    def test_half_up_rounds_positive_input_before_precision_check(self):
        serializer = QuoteSerializer(data={'amount': '5.125'})
        assert serializer.is_valid(), serializer.errors
        assert serializer.validated_data['amount'] == Decimal('5.13')

    def test_half_up_rounds_negative_input_away_from_zero(self):
        serializer = QuoteSerializer(data={'amount': '-5.125'})
        assert serializer.is_valid(), serializer.errors
        assert serializer.validated_data['amount'] == Decimal('-5.13')

    def test_rounding_past_storable_digits_is_a_storage_error(self):
        # 99.995 rounds to 100.00, which needs 5 digits > max_digits=4.
        serializer = QuoteSerializer(data={'amount': '99.995'})
        assert not serializer.is_valid()
        assert serializer.errors['amount'][0].code == 'max_digits'
        assert str(serializer.errors['amount'][0]) == (
            'Ensure that there are no more than 4 digits in total.'
        )

    def test_rounding_past_business_max_is_a_business_error(self):
        # 50.005 fits the column once rounded (50.01) but breaches the
        # caller-registered business cap of 50.00.
        serializer = QuoteSerializer(data={'amount': '50.005'})
        assert not serializer.is_valid()
        assert serializer.errors['amount'][0].code == 'max_value'
        assert str(serializer.errors['amount'][0]) == (
            'Ensure this value is less than or equal to 50.00.'
        )

    def test_validator_and_serializer_see_the_written_value(self):
        serializer = QuoteSerializer(data={'amount': '5.125'})
        assert serializer.is_valid(), serializer.errors

        written = serializer.validated_data['amount']
        assert written == Decimal('5.13')
        assert seen_by_amount_validator == [written]
        assert seen_by_serializer == [written]

    def test_output_configuration_is_unchanged(self):
        field = PreRoundingDecimalField(
            max_digits=4, decimal_places=2, rounding=decimal.ROUND_HALF_UP,
        )
        # Default coercion/quantized output, exactly like the stock field.
        assert field.to_representation(Decimal('5.13')) == '5.13'
        assert field.to_representation(Decimal('5.1')) == '5.10'


class TestQuoteAPI:
    def test_quote_endpoint_accepts_and_normalizes(self):
        factory = APIRequestFactory()
        request = factory.post('/quotes/', {'amount': '5.125'}, format='json')
        response = QuoteView.as_view()(request)
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data['amount'] == Decimal('5.13')
        assert seen_by_amount_validator == [Decimal('5.13')]
        assert seen_by_serializer == [Decimal('5.13')]

    def test_quote_endpoint_rejects_storage_overflow(self):
        factory = APIRequestFactory()
        request = factory.post('/quotes/', {'amount': '99.995'}, format='json')
        response = QuoteView.as_view()(request)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.data['amount'][0].code == 'max_digits'

    def test_quote_endpoint_rejects_business_overflow(self):
        factory = APIRequestFactory()
        request = factory.post('/quotes/', {'amount': '50.005'}, format='json')
        response = QuoteView.as_view()(request)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.data['amount'][0].code == 'max_value'


class TestStockDecimalFieldStaysStrict:
    """
    The original field keeps its strict behavior: multi-decimal input is
    rejected on the raw value, even when a rounding mode is declared.
    """

    @pytest.mark.parametrize('value,code', [
        ('5.125', 'max_decimal_places'),
        ('-5.125', 'max_decimal_places'),
        ('99.995', 'max_digits'),
        ('50.005', 'max_digits'),
    ])
    def test_strict_field_rejects_extra_decimal_input(self, value, code):
        field = serializers.DecimalField(
            max_digits=4, decimal_places=2, rounding=decimal.ROUND_HALF_UP,
        )
        with pytest.raises(serializers.ValidationError) as exc_info:
            field.run_validation(value)
        assert exc_info.value.detail[0].code == code
