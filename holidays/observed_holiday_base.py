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

from datetime import date
from datetime import timedelta as td
from typing import Dict, Optional, Tuple, Set

from holidays.calendars.gregorian import MON, TUE, WED, THU, FRI, SAT, SUN
from holidays.constants import HOLIDAY_NAME_DELIMITER
from holidays.holiday_base import DateArg, HolidayBase


class ObservedRule(Dict[int, int]):
    __slots__ = ()

    def __add__(self, other):
        return ObservedRule({**self, **other})


# Observance calculation rules: +7 - next workday, -7 - previous workday.
# Single days.
MON_TO_NEXT_TUE = ObservedRule({MON: +1})

TUE_TO_PREV_MON = ObservedRule({TUE: -1})
TUE_TO_PREV_FRI = ObservedRule({TUE: -4})

WED_TO_PREV_MON = ObservedRule({WED: -2})
WED_TO_NEXT_FRI = ObservedRule({WED: +2})

THU_TO_PREV_MON = ObservedRule({THU: -3})
THU_TO_PREV_WED = ObservedRule({THU: -1})
THU_TO_NEXT_MON = ObservedRule({THU: +4})
THU_TO_NEXT_FRI = ObservedRule({THU: +1})

FRI_TO_PREV_WED = ObservedRule({FRI: -2})
FRI_TO_PREV_THU = ObservedRule({FRI: -1})
FRI_TO_NEXT_MON = ObservedRule({FRI: +3})
FRI_TO_NEXT_TUE = ObservedRule({FRI: +4})
FRI_TO_NEXT_SAT = ObservedRule({FRI: +1})
FRI_TO_NEXT_WORKDAY = ObservedRule({FRI: +7})

SAT_TO_PREV_THU = ObservedRule({SAT: -2})
SAT_TO_PREV_FRI = ObservedRule({SAT: -1})
SAT_TO_PREV_WORKDAY = ObservedRule({SAT: -7})
SAT_TO_NEXT_MON = ObservedRule({SAT: +2})
SAT_TO_NEXT_TUE = ObservedRule({SAT: +3})
SAT_TO_NEXT_SUN = ObservedRule({SAT: +1})
SAT_TO_NEXT_WORKDAY = ObservedRule({SAT: +7})

SUN_TO_NEXT_MON = ObservedRule({SUN: +1})
SUN_TO_NEXT_TUE = ObservedRule({SUN: +2})
SUN_TO_NEXT_WED = ObservedRule({SUN: +3})
SUN_TO_NEXT_WORKDAY = ObservedRule({SUN: +7})

# Multiple days.
ALL_TO_NEAREST_MON = ObservedRule({TUE: -1, WED: -2, THU: -3, FRI: +3, SAT: +2, SUN: +1})
ALL_TO_NEAREST_MON_LATAM = ObservedRule({TUE: -1, WED: -2, THU: 4, FRI: +3, SAT: +2, SUN: +1})
ALL_TO_NEXT_MON = ObservedRule({TUE: +6, WED: +5, THU: +4, FRI: +3, SAT: +2, SUN: +1})
ALL_TO_NEXT_SUN = ObservedRule({MON: +6, TUE: +5, WED: +4, THU: +3, FRI: +2, SAT: +1})

WORKDAY_TO_NEAREST_MON = ObservedRule({TUE: -1, WED: -2, THU: -3, FRI: +3})
WORKDAY_TO_NEXT_MON = ObservedRule({TUE: +6, WED: +5, THU: +4, FRI: +3})
WORKDAY_TO_NEXT_WORKDAY = ObservedRule({MON: +7, TUE: +7, WED: +7, THU: +7, FRI: +7})

TUE_WED_TO_PREV_MON = ObservedRule({TUE: -1, WED: -2})
TUE_WED_THU_TO_PREV_MON = ObservedRule({TUE: -1, WED: -2, THU: -3})

WED_THU_TO_NEXT_FRI = ObservedRule({WED: +2, THU: +1})

THU_FRI_TO_NEXT_MON = ObservedRule({THU: +4, FRI: +3})
THU_FRI_TO_NEXT_WORKDAY = ObservedRule({THU: +7, FRI: +7})
THU_FRI_SUN_TO_NEXT_MON = ObservedRule({THU: +4, FRI: +3, SUN: +1})

FRI_SAT_TO_NEXT_WORKDAY = ObservedRule({FRI: +7, SAT: +7})
FRI_SUN_TO_NEXT_MON = ObservedRule({FRI: +3, SUN: +1})
FRI_SUN_TO_NEXT_SAT_MON = ObservedRule({FRI: +1, SUN: +1})

SAT_SUN_TO_PREV_FRI = ObservedRule({SAT: -1, SUN: -2})
SAT_SUN_TO_NEXT_MON = ObservedRule({SAT: +2, SUN: +1})
SAT_SUN_TO_NEXT_TUE = ObservedRule({SAT: +3, SUN: +2})
SAT_SUN_TO_NEXT_WED = ObservedRule({SAT: +4, SUN: +3})
SAT_SUN_TO_NEXT_MON_TUE = ObservedRule({SAT: +2, SUN: +2})
SAT_SUN_TO_NEXT_WORKDAY = ObservedRule({SAT: +7, SUN: +7})


class ObservedHolidayBase(HolidayBase):
    """Observed holidays implementation."""

    observed_label = "%s"

    def __init__(
        self,
        observed_rule: ObservedRule,
        observed_since: int = None,
        cross_year: bool = False,
        *args,
        **kwargs,
    ):
        self._observed_rule = observed_rule
        self._observed_since = observed_since
        # Whether an observed date is allowed to fall into the previous or
        # the next calendar year (e.g. a closing day on Sunday, December 31
        # observed on Tuesday, January 2 when January 1 is also a holiday).
        self.cross_year = cross_year
        # Dates that only hold an observed (moved) holiday. Used in the
        # cross-year mode so such entries are not treated as closing days
        # while looking for the next workday.
        self._observed_only_dates: Set[date] = set()

        super().__init__(*args, **kwargs)

    def _is_observed(self, *args, **kwargs) -> bool:
        return self._observed_since is None or self._year >= self._observed_since

    def _is_closing_day(self, dt: date, observed_only_dates: Set[date]) -> bool:
        """Return True for an actual closing day. Dates that only hold an
        observed (moved) holiday are skipped so they do not extend the
        next-workday lookup across the year boundary."""
        return dict.__contains__(self, dt) and dt not in observed_only_dates

    def _get_next_workday(self, dt: date, delta: int = +1) -> date:
        dt_work = dt + td(days=delta)
        if not self.cross_year:
            # Historical behavior: the observed date must stay within the
            # populated year; every holiday (observed ones included) counts.
            while dt_work.year == self._year:
                if dict.__contains__(self, dt_work) or self._is_weekend(dt_work):
                    dt_work += td(days=delta)
                else:
                    return dt_work
            return dt

        # Cross-year mode: keep going across the year boundary until the
        # first non-holiday workday is found. Neighboring calendar years are
        # populated on demand so closing marks just across the boundary
        # (e.g. January 1) are taken into account.
        neighbor_years = self._get_neighbor_years()
        observed_only_dates = self._collect_observed_only_dates(neighbor_years)
        while True:
            neighbor = neighbor_years.get(dt_work.year)
            if (
                self._is_closing_day(dt_work, observed_only_dates)
                or (
                    neighbor is not None
                    and neighbor._is_closing_day(dt_work, observed_only_dates)
                )
                or self._is_weekend(dt_work)
            ):
                dt_work += td(days=delta)
            else:
                return dt_work

    def _collect_observed_only_dates(
        self, neighbor_years: Dict[int, "ObservedHolidayBase"]
    ) -> Set[date]:
        """Collect dates that only hold an observed holiday across the
        populated year and its neighboring years."""
        observed_only_dates: Set[date] = set()
        for calendar in (self, *neighbor_years.values()):
            observed_only_dates.update(calendar._observed_only_dates)
        return observed_only_dates

    def _build_isolated_year(self, year: int, cross_year: bool = False) -> "ObservedHolidayBase":
        """Build an isolated single-year calendar without any cross-year
        side effects (neighbor lookups or previous-year back-filling)."""
        isolated = self.copy()
        isolated.clear()
        isolated.years = set()
        isolated.cross_year = cross_year
        isolated._observed_only_dates = set()
        isolated._skip_cross_year_backfill = True
        isolated._populate(year)
        return isolated

    def _get_neighbor_years(self) -> Dict[int, "ObservedHolidayBase"]:
        """Build isolated calendars for the years adjacent to the populated
        year when cross-year observed dates are enabled."""
        if not self.cross_year:
            return {}

        neighbor_years: Dict[int, "ObservedHolidayBase"] = {}
        for neighbor_year in (self._year - 1, self._year + 1):
            if neighbor_year in self.years:
                continue
            neighbor_years[neighbor_year] = self._build_isolated_year(neighbor_year)
        return neighbor_years

    def _get_observed_date(self, dt: date, rule: ObservedRule) -> date:
        delta = rule.get(dt.weekday(), 0)
        if delta != 0:
            if abs(delta) == 7:
                dt = self._get_next_workday(dt, delta // 7)
            else:
                dt += td(days=delta)
        return dt

    def _add_observed(
        self, dt: DateArg, name: Optional[str] = None, rule: Optional[ObservedRule] = None
    ) -> Tuple[bool, date]:
        dt = dt if isinstance(dt, date) else date(self._year, *dt)

        if not self.observed or not self._is_observed(dt):
            return False, dt

        dt_observed = self._get_observed_date(dt, rule or self._observed_rule)
        if dt_observed == dt:
            return False, dt

        estimated_label = self.tr(getattr(self, "estimated_label", ""))
        observed_label = self.tr(
            getattr(
                self,
                "observed_label_before" if dt_observed < dt else "observed_label",
                self.observed_label,
            )
        )

        estimated_label_text = estimated_label.strip("%s ()")
        # Use observed_estimated_label instead of observed_label for estimated dates.
        for name in (name,) if name else self.get_list(dt):
            holiday_name = self.tr(name)
            observed_estimated_label = None
            if len(estimated_label_text) > 0 and estimated_label_text in holiday_name:
                holiday_name = holiday_name.replace(f"({estimated_label_text})", "").strip()
                observed_estimated_label = self.tr(getattr(self, "observed_estimated_label"))

            observed_name = (observed_estimated_label or observed_label) % holiday_name
            if dt_observed.year == self._year:
                # Same calendar year: the regular add path (silently drops
                # nothing here as the year matches).
                super()._add_holiday(observed_name, dt_observed)
            elif self.cross_year:
                # Cross-year mode: keep the observed date and remember it is
                # an observed-only entry for the next-workday lookup.
                if not dict.__contains__(self, dt_observed):
                    self._observed_only_dates.add(dt_observed)
                dict.__setitem__(
                    self, dt_observed, self._merge_holiday_name(dt_observed, observed_name)
                )
            # In the non-cross-year mode out-of-year observed dates are
            # silently dropped, matching the historical `_add_holiday` behavior.

        return True, dt_observed

    def _merge_holiday_name(self, dt: date, name: str) -> str:
        """Build the value for a date that already holds one or more holidays."""
        existing_value = dict.get(self, dt)
        existing_names = (
            set(existing_value.split(HOLIDAY_NAME_DELIMITER)) if existing_value else set()
        )
        existing_names.add(name)
        return HOLIDAY_NAME_DELIMITER.join(sorted(existing_names))

    def _populate(self, year: int) -> None:
        super()._populate(year)

        if (
            not self.cross_year
            or not self.observed
            or getattr(self, "_skip_cross_year_backfill", False)
        ):
            return None

        # Merge observed dates that overflow from the adjacent calendar years
        # into the current one, so querying either calendar year gives the
        # same result (e.g. a Sunday December 31 observed on January 2, or a
        # Saturday January 1 observed on the previous December 31).
        for adjacent_year in (year - 1, year + 1):
            if adjacent_year in self.years:
                continue
            adjacent = self._build_isolated_year(adjacent_year, cross_year=True)
            for dt, name in tuple(adjacent.items()):
                if dt.year != year:
                    continue
                if not dict.__contains__(self, dt):
                    # The date has no actual holiday in the current year: it
                    # only holds the back-filled observed one.
                    self._observed_only_dates.add(dt)
                dict.__setitem__(self, dt, self._merge_holiday_name(dt, name))

    def _move_holiday(self, dt: date, rule: Optional[ObservedRule] = None) -> Tuple[bool, date]:
        is_observed, dt_observed = self._add_observed(dt, rule=rule)
        if is_observed:
            self.pop(dt)
        return is_observed, dt_observed if is_observed else dt

    def _populate_observed(self, dts: Set[date], multiple: bool = False) -> None:
        """
        When multiple is True, each holiday from a given date has its own observed date.
        """
        for dt in sorted(dts):
            if not self._is_observed(dt):
                continue
            if multiple:
                for name in self.get_list(dt):
                    self._add_observed(dt, name)
            else:
                self._add_observed(dt)

    def _populate_common_holidays(self):
        """Populate entity common holidays."""
        super()._populate_common_holidays()

        if not self.observed or not self.has_special_holidays:
            return None

        self._add_special_holidays(
            (f"special_{category}_holidays_observed" for category in self._sorted_categories),
            observed=True,
        )

    def _populate_subdiv_holidays(self):
        """Populate entity subdivision holidays."""
        super()._populate_subdiv_holidays()

        if not self.subdiv or not self.observed or not self.has_special_holidays:
            return None

        self._add_special_holidays(
            (
                f"special_{self._normalized_subdiv}_{category}_holidays_observed"
                for category in self._sorted_categories
            ),
            observed=True,
        )
