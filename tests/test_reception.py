#  holidays
#  --------
#  A fast, efficient Python library for generating country, province and state
#  specific sets of holidays on the fly. It aims to make determining whether a
#  specific date is a holiday as fast and flexible as possible.
#
#  Authors: Vacanza Team and individual contributors (see AUTHORS file)
#           dr-prodigy <dr.prodigy.github@gmail.com> (c) 2017-2023
#           ryanss <ryanssdev@icloud.com> (c) 2014-2017
#  Website: https://github.com/vacanza/python-holidays
#  License: MIT (see LICENSE file)

from datetime import date, time
from unittest import TestCase

from holidays.reception import AM, PM, FULL_DAY, HALF_DAY, TIME_RANGE, ReceptionCalendar


class TestReceptionCalendar(TestCase):
    def setUp(self):
        self.cal = ReceptionCalendar()
        self.day = date(2026, 11, 30)

    def test_default_business_hours(self):
        self.assertEqual(self.cal.open_time, time(9, 0))
        self.assertEqual(self.cal.close_time, time(17, 0))
        self.assertEqual(self.cal.midday, time(13, 0))

    def test_unconfigured_day_follows_default_open_rule(self):
        self.assertEqual(self.cal.bookable_slots(self.day), [(time(9, 0), time(17, 0))])
        self.assertEqual(self.cal.closed_intervals(self.day), [])
        self.assertEqual(self.cal.closures_on(self.day), [])
        self.assertTrue(self.cal.is_bookable(self.day))

    def test_overlapping_time_range_closures_counted_once(self):
        self.cal.add_closure(self.day, "Official notice", "09:00", "12:00", source="official")
        self.cal.add_closure(
            self.day, "Local maintenance", time(11, 0), time(14, 0), source="maintenance"
        )

        # Overlap 11:00-12:00 is deducted only once.
        self.assertEqual(self.cal.closed_intervals(self.day), [(time(9, 0), time(14, 0))])
        self.assertEqual(self.cal.bookable_slots(self.day), [(time(14, 0), time(17, 0))])

        # Both sources stay visible.
        closures = self.cal.closures_on(self.day)
        self.assertEqual(len(closures), 2)
        self.assertEqual({c.source for c in closures}, {"official", "maintenance"})
        self.assertTrue(all(c.kind == TIME_RANGE for c in closures))

    def test_half_day_closures(self):
        self.cal.add_half_day_closure(self.day, AM, "Morning maintenance")
        self.assertEqual(self.cal.bookable_slots(self.day), [(time(13, 0), time(17, 0))])
        closure = self.cal.closures_on(self.day)[0]
        self.assertEqual(closure.kind, HALF_DAY)
        self.assertEqual((closure.start, closure.end), (time(9, 0), time(13, 0)))

        other = ReceptionCalendar()
        other.add_half_day_closure(self.day, PM, "Afternoon maintenance")
        self.assertEqual(other.bookable_slots(self.day), [(time(9, 0), time(13, 0))])

    def test_full_day_closure_keeps_holiday_behavior(self):
        self.cal.add_full_day_closure(self.day, "Christmas", source="official")
        # Stored as a regular holiday entry.
        self.assertIn(self.day, self.cal)
        self.assertEqual(self.cal[self.day], "Christmas")
        # Name lists keep working and merge as before.
        self.cal[self.day] = "Staff day"
        self.assertEqual(self.cal.get_list(self.day), ["Christmas", "Staff day"])
        # Whole business day is closed.
        self.assertEqual(self.cal.bookable_slots(self.day), [])
        self.assertEqual(self.cal.closed_intervals(self.day), [(time(9, 0), time(17, 0))])
        self.assertFalse(self.cal.is_bookable(self.day))
        closure = self.cal.closures_on(self.day)[0]
        self.assertEqual(closure.kind, FULL_DAY)
        self.assertTrue(closure.is_full_day)

    def test_add_closure_without_times_is_full_day(self):
        self.cal.add_closure(self.day, "Power outage")
        self.assertIn(self.day, self.cal)
        self.assertEqual(self.cal.bookable_slots(self.day), [])

    def test_is_bookable_with_time_range(self):
        self.cal.add_closure(self.day, "Official notice", "09:00", "12:00", source="official")
        self.cal.add_closure(self.day, "Local maintenance", "11:00", "14:00")
        self.assertTrue(self.cal.is_bookable(self.day, "14:00", "17:00"))
        self.assertTrue(self.cal.is_bookable(self.day, time(15, 0), time(16, 30)))
        self.assertFalse(self.cal.is_bookable(self.day, "13:00", "15:00"))
        self.assertFalse(self.cal.is_bookable(self.day, "10:00", "11:00"))

    def test_custom_business_hours(self):
        cal = ReceptionCalendar(open_time="08:30", close_time="12:30")
        self.assertEqual(cal.midday, time(10, 30))
        cal.add_half_day_closure(self.day, PM, "Closing early")
        self.assertEqual(cal.bookable_slots(self.day), [(time(8, 30), time(10, 30))])

    def test_invalid_configuration(self):
        self.assertRaises(ValueError, ReceptionCalendar, open_time="17:00", close_time="09:00")
        self.assertRaises(ValueError, ReceptionCalendar, midday="20:00")

    def test_invalid_closure(self):
        self.assertRaises(ValueError, self.cal.add_closure, self.day, "Bad", "12:00", "09:00")
        self.assertRaises(ValueError, self.cal.add_closure, self.day, "Bad", "20:00", "21:00")
        self.assertRaises(ValueError, self.cal.add_closure, self.day, "Bad", "09:00")
        self.assertRaises(ValueError, self.cal.add_half_day_closure, self.day, "noon", "Bad")
        self.assertRaises(ValueError, self.cal.is_bookable, self.day, "12:00", "09:00")

    def test_pop_and_delete_remove_closures(self):
        self.cal.add_closure(self.day, "Maintenance", "11:00", "14:00")
        self.cal.add_full_day_closure(self.day, "Holiday")
        self.cal.pop(self.day)
        self.assertEqual(self.cal.closures_on(self.day), [])
        self.assertNotIn(self.day, self.cal)

        self.cal.add_closure(self.day, "Maintenance", "11:00", "14:00")
        del self.cal[self.day]
        self.assertEqual(self.cal.closures_on(self.day), [])

    def test_years_and_populate_still_work(self):
        cal = ReceptionCalendar(years=2026)
        self.assertEqual(cal.bookable_slots(self.day), [(time(9, 0), time(17, 0))])
