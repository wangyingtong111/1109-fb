from django.test import TestCase

from django_filters.filters import BooleanFilter, CharFilter, ChoiceFilter, Filter
from django_filters.filterset import FilterSet

from .models import CaseMaterial, LegalCase


class SameMaterialFilterSet(FilterSet):
    """Conditions on materials must match the *same* material."""

    case_number = CharFilter()
    category = ChoiceFilter(
        field_name="materials__category",
        choices=CaseMaterial.CATEGORY_CHOICES,
        relation_group="materials",
    )
    confirmed = BooleanFilter(
        field_name="materials__confirmed",
        relation_group="materials",
    )

    class Meta:
        model = LegalCase
        fields = ["case_number"]


class SpanningMaterialFilterSet(FilterSet):
    """Legacy behavior: each condition may match a *different* material."""

    case_number = CharFilter()
    category = ChoiceFilter(
        field_name="materials__category",
        choices=CaseMaterial.CATEGORY_CHOICES,
    )
    confirmed = BooleanFilter(field_name="materials__confirmed")

    class Meta:
        model = LegalCase
        fields = ["case_number"]


class RelationGroupFilteringTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Case A: an unconfirmed scan and a confirmed photo. It matches each
        # condition individually, but no single material matches both.
        cls.case_a = LegalCase.objects.create(case_number="A-001")
        CaseMaterial.objects.create(
            case=cls.case_a, category=CaseMaterial.SCAN, confirmed=False
        )
        CaseMaterial.objects.create(
            case=cls.case_a, category=CaseMaterial.PHOTO, confirmed=True
        )

        # Case B: two confirmed scans (duplicate join rows) and one photo.
        cls.case_b = LegalCase.objects.create(case_number="B-002")
        CaseMaterial.objects.create(
            case=cls.case_b, category=CaseMaterial.SCAN, confirmed=True
        )
        CaseMaterial.objects.create(
            case=cls.case_b, category=CaseMaterial.SCAN, confirmed=True
        )
        CaseMaterial.objects.create(
            case=cls.case_b, category=CaseMaterial.PHOTO, confirmed=False
        )

        # Case C: no scans at all.
        cls.case_c = LegalCase.objects.create(case_number="C-003")
        CaseMaterial.objects.create(
            case=cls.case_c, category=CaseMaterial.PHOTO, confirmed=True
        )

    def test_same_relation_group_matches_single_material(self):
        f = SameMaterialFilterSet(
            {"category": "scan", "confirmed": "true"},
            queryset=LegalCase.objects.all(),
        )
        self.assertTrue(f.is_valid())
        self.assertEqual(list(f.qs), [self.case_b])

    def test_spanning_filters_match_across_materials(self):
        # The legacy behavior splits the conditions over two materials,
        # so case A (unconfirmed scan + confirmed photo) also matches.
        f = SpanningMaterialFilterSet(
            {"category": "scan", "confirmed": "true"},
            queryset=LegalCase.objects.all(),
        )
        self.assertTrue(f.is_valid())
        self.assertEqual(set(f.qs), {self.case_a, self.case_b})
        # ... and without distinct, the repeated joins duplicate case B.
        self.assertGreater(len(list(f.qs)), 2)

    def test_ungrouped_conditions_are_not_bound_to_relation(self):
        # The case number is an ordinary condition: it applies to the case
        # itself, independently of the material conditions.
        f = SameMaterialFilterSet(
            {"case_number": "B-002", "category": "scan", "confirmed": "true"},
            queryset=LegalCase.objects.all(),
        )
        self.assertTrue(f.is_valid())
        self.assertEqual(list(f.qs), [self.case_b])

        f = SameMaterialFilterSet(
            {"case_number": "A-001", "category": "scan", "confirmed": "true"},
            queryset=LegalCase.objects.all(),
        )
        self.assertTrue(f.is_valid())
        self.assertEqual(list(f.qs), [])

    def test_ungrouped_condition_only(self):
        f = SameMaterialFilterSet(
            {"case_number": "C-003"}, queryset=LegalCase.objects.all()
        )
        self.assertTrue(f.is_valid())
        self.assertEqual(list(f.qs), [self.case_c])

    def test_repeated_relation_does_not_duplicate_cases(self):
        # Case B has two confirmed scans; it must appear only once.
        f = SameMaterialFilterSet(
            {"category": "scan", "confirmed": "true"},
            queryset=LegalCase.objects.all(),
        )
        self.assertTrue(f.is_valid())
        self.assertEqual(list(f.qs), [self.case_b])

    def test_single_grouped_condition(self):
        f = SameMaterialFilterSet(
            {"category": "scan"}, queryset=LegalCase.objects.all()
        )
        self.assertTrue(f.is_valid())
        # Case B has two scans but must not be listed twice.
        self.assertCountEqual(list(f.qs), [self.case_a, self.case_b])

    def test_empty_group_values_do_not_filter(self):
        f = SameMaterialFilterSet(queryset=LegalCase.objects.all())
        self.assertFalse(f.is_bound)
        self.assertCountEqual(
            list(f.qs), [self.case_a, self.case_b, self.case_c]
        )

        f = SameMaterialFilterSet(
            {"category": "", "confirmed": ""}, queryset=LegalCase.objects.all()
        )
        self.assertTrue(f.is_valid())
        self.assertCountEqual(
            list(f.qs), [self.case_a, self.case_b, self.case_c]
        )

    def test_null_choice_value_skips_grouped_condition(self):
        class F(SameMaterialFilterSet):
            category = ChoiceFilter(
                field_name="materials__category",
                choices=CaseMaterial.CATEGORY_CHOICES,
                null_label="Any",
                relation_group="materials",
            )

        f = F(
            {"category": "null", "confirmed": "true"},
            queryset=LegalCase.objects.all(),
        )
        self.assertTrue(f.is_valid())
        # Only the confirmed condition applies: all cases have a confirmed
        # material of some category.
        self.assertCountEqual(list(f.qs), [self.case_a, self.case_b, self.case_c])


class RelationGroupDeclarationTests(TestCase):
    def test_method_not_allowed_with_relation_group(self):
        with self.assertRaises(ValueError):
            CharFilter(method="filter_it", relation_group="materials")

    def test_exclude_not_allowed_with_relation_group(self):
        with self.assertRaises(ValueError):
            CharFilter(exclude=True, relation_group="materials")

    def test_relation_group_defaults_to_none(self):
        self.assertIsNone(Filter().relation_group)
