"""Reception desk service calendar with partial-day closure support.

This module provides :class:`ReceptionCalendar`, a calendar for reception /
front-desk services that are normally open during fixed business hours but
may be closed for a whole day, half a day, or a specific time range (e.g.
due to official notices or local maintenance).

Full-day closures keep the classic ``HolidayBase`` behavior: they are stored
as ``{date: "name; name"}`` entries, so existing name lists, ``get_list()``,
``pop_named()``, and friends keep working unchanged. Partial-day closures
(half-day and time-range) are recorded alongside with their source, and the
calendar can compute merged closed intervals and remaining bookable slots.
Overlapping closures from different sources are only deducted once while
every source stays visible.
"""

from dataclasses import dataclass
from datetime import date, time
from typing import Dict, List, Optional, Tuple, Union

from holidays.holiday_base import DateLike, HolidayBase

__all__ = (
    "AM",
    "PM",
    "FULL_DAY",
    "HALF_DAY",
    "TIME_RANGE",
    "Closure",
    "ReceptionCalendar",
)

# Closure kinds.
FULL_DAY = "full_day"
HALF_DAY = "half_day"
TIME_RANGE = "time_range"

# Half-day periods.
AM = "am"
PM = "pm"

TimeLike = Union[time, str]
"""A time of day expressed as :class:`datetime.time` or an ISO ``"HH:MM"`` string."""


@dataclass(frozen=True)
class Closure:
    """A single reception closure record.

    :param name:
        The closure name (e.g. the reason shown to users).

    :param source:
        Where the closure comes from (e.g. ``"official"``, ``"maintenance"``).
        Every source is kept so overlapping closures remain attributable.

    :param kind:
        One of ``FULL_DAY``, ``HALF_DAY`` or ``TIME_RANGE``.

    :param start:
        Closure start time; ``None`` for full-day closures.

    :param end:
        Closure end time; ``None`` for full-day closures.
    """

    name: str
    source: str
    kind: str
    start: Optional[time] = None
    end: Optional[time] = None

    @property
    def is_full_day(self) -> bool:
        """Return True if this is a full-day closure."""
        return self.kind == FULL_DAY


class ReceptionCalendar(HolidayBase):
    """Reception desk calendar supporting full-day, half-day and time-range closures.

    The calendar assumes the reception is normally open every day between
    ``open_time`` and ``close_time``. Dates without any closure configuration
    follow this default and are fully bookable.

    Full-day closures are stored as regular holiday entries (date -> name(s)),
    so existing behavior and name lists keep working. Half-day and time-range
    closures are recorded separately with their source.

    :param open_time:
        Daily opening time (default 09:00).

    :param close_time:
        Daily closing time (default 17:00).

    :param midday:
        Boundary between the AM and PM half-day periods. Defaults to the
        midpoint between ``open_time`` and ``close_time``.
    """

    default_open_time = time(9, 0)
    default_close_time = time(17, 0)

    def __init__(
        self,
        *args,
        open_time: Optional[TimeLike] = None,
        close_time: Optional[TimeLike] = None,
        midday: Optional[TimeLike] = None,
        **kwargs,
    ) -> None:
        self.open_time = self._to_time(open_time) if open_time else self.default_open_time
        self.close_time = self._to_time(close_time) if close_time else self.default_close_time
        if self.open_time >= self.close_time:
            raise ValueError(
                f"open_time ({self.open_time}) must be earlier than "
                f"close_time ({self.close_time})."
            )
        self.midday = (
            self._to_time(midday) if midday else self._midpoint(self.open_time, self.close_time)
        )
        if not self.open_time < self.midday < self.close_time:
            raise ValueError(
                f"midday ({self.midday}) must lie between open_time ({self.open_time}) "
                f"and close_time ({self.close_time})."
            )
        # Partial-day closures: {date: [Closure, ...]}.
        # Full-day closures are recorded here too (for source/kind visibility)
        # in addition to the regular HolidayBase dict entries.
        self._closures: Dict[date, List[Closure]] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _to_time(value: TimeLike) -> time:
        """Convert a TimeLike value to :class:`datetime.time`."""
        if isinstance(value, time):
            return value.replace(tzinfo=None)
        if isinstance(value, str):
            try:
                return time.fromisoformat(value)
            except ValueError:
                raise ValueError(f"Cannot parse time from string {value!r}") from None
        raise TypeError(f"Cannot convert type '{type(value)}' to time.")

    @staticmethod
    def _midpoint(start: time, end: time) -> time:
        """Return the midpoint between two times of day."""
        start_minutes = start.hour * 60 + start.minute
        end_minutes = end.hour * 60 + end.minute
        mid_minutes = (start_minutes + end_minutes) // 2
        return time(mid_minutes // 60, mid_minutes % 60)

    @staticmethod
    def _merge_intervals(
        intervals: List[Tuple[time, time]],
    ) -> List[Tuple[time, time]]:
        """Merge overlapping or touching intervals, counting overlaps only once."""
        merged: List[Tuple[time, time]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                if end > merged[-1][1]:
                    merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def _clip(self, start: time, end: time) -> Optional[Tuple[time, time]]:
        """Clip an interval to business hours; return None if it falls outside."""
        clipped_start = max(start, self.open_time)
        clipped_end = min(end, self.close_time)
        if clipped_start >= clipped_end:
            return None
        return clipped_start, clipped_end

    def add_closure(
        self,
        key: DateLike,
        name: str,
        start: Optional[TimeLike] = None,
        end: Optional[TimeLike] = None,
        source: str = "local",
    ) -> Closure:
        """Add a closure for a specific time range on a date.

        :param key:
            The date of the closure.

        :param name:
            The closure name (reason).

        :param start:
            Closure start time. If both ``start`` and ``end`` are omitted a
            full-day closure is recorded instead.

        :param end:
            Closure end time.

        :param source:
            The closure source (e.g. ``"official"``, ``"maintenance"``).

        :return:
            The recorded :class:`Closure`.
        """
        if (start is None) != (end is None):
            raise ValueError("Both start and end must be given, or neither.")
        if start is None:
            return self.add_full_day_closure(key, name, source=source)

        start_t, end_t = self._to_time(start), self._to_time(end)
        if start_t >= end_t:
            raise ValueError(f"Closure start ({start_t}) must be earlier than end ({end_t}).")
        if self._clip(start_t, end_t) is None:
            raise ValueError(
                f"Closure {start_t}-{end_t} falls outside business hours "
                f"{self.open_time}-{self.close_time}."
            )

        closure = Closure(name=name, source=source, kind=TIME_RANGE, start=start_t, end=end_t)
        self._closures.setdefault(self.__keytransform__(key), []).append(closure)
        return closure

    def add_half_day_closure(
        self,
        key: DateLike,
        period: str,
        name: str,
        source: str = "local",
    ) -> Closure:
        """Add a half-day (AM or PM) closure on a date.

        :param key:
            The date of the closure.

        :param period:
            ``AM`` (open time to midday) or ``PM`` (midday to close time).

        :param name:
            The closure name (reason).

        :param source:
            The closure source.

        :return:
            The recorded :class:`Closure`.
        """
        period = period.lower()
        if period == AM:
            start, end = self.open_time, self.midday
        elif period == PM:
            start, end = self.midday, self.close_time
        else:
            raise ValueError(f"Unknown half-day period: {period!r} (expected 'am' or 'pm').")

        closure = Closure(name=name, source=source, kind=HALF_DAY, start=start, end=end)
        self._closures.setdefault(self.__keytransform__(key), []).append(closure)
        return closure

    def add_full_day_closure(
        self,
        key: DateLike,
        name: str,
        source: str = "local",
    ) -> Closure:
        """Add a full-day closure on a date.

        The closure is stored as a regular holiday entry, so name lists and
        all existing ``HolidayBase`` behavior keep working.

        :param key:
            The date of the closure.

        :param name:
            The closure name (reason).

        :param source:
            The closure source.

        :return:
            The recorded :class:`Closure`.
        """
        self[key] = name
        closure = Closure(name=name, source=source, kind=FULL_DAY)
        self._closures.setdefault(self.__keytransform__(key), []).append(closure)
        return closure

    def closures_on(self, key: DateLike) -> List[Closure]:
        """Return all closure records for a date, keeping every source visible."""
        return list(self._closures.get(self.__keytransform__(key), ()))

    def closed_intervals(self, key: DateLike) -> List[Tuple[time, time]]:
        """Return merged closed intervals within business hours for a date.

        Overlapping closures (even from different sources) are only counted
        once. A full-day closure covers the whole business day.
        """
        intervals = []
        for closure in self._closures.get(self.__keytransform__(key), ()):
            if closure.is_full_day:
                return [(self.open_time, self.close_time)]
            clipped = self._clip(closure.start, closure.end)  # type: ignore[arg-type]
            if clipped is not None:
                intervals.append(clipped)
        return self._merge_intervals(intervals)

    def bookable_slots(self, key: DateLike) -> List[Tuple[time, time]]:
        """Return the remaining bookable time slots for a date.

        Dates without any closure configuration follow the default rule and
        are bookable for the whole business day.
        """
        bookable: List[Tuple[time, time]] = []
        cursor = self.open_time
        for start, end in self.closed_intervals(key):
            if cursor < start:
                bookable.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < self.close_time:
            bookable.append((cursor, self.close_time))
        return bookable

    def is_bookable(
        self,
        key: DateLike,
        start: Optional[TimeLike] = None,
        end: Optional[TimeLike] = None,
    ) -> bool:
        """Return whether a date (or a specific time range on it) is bookable.

        Without ``start``/``end`` this returns True if any bookable time
        remains on the date. With both given, it returns True only if the
        whole requested range fits inside a single bookable slot.
        """
        slots = self.bookable_slots(key)
        if start is None and end is None:
            return bool(slots)
        if start is None or end is None:
            raise ValueError("Both start and end must be given, or neither.")
        start_t, end_t = self._to_time(start), self._to_time(end)
        if start_t >= end_t:
            raise ValueError(f"Start ({start_t}) must be earlier than end ({end_t}).")
        return any(slot_start <= start_t and end_t <= slot_end for slot_start, slot_end in slots)

    def __delitem__(self, key: DateLike) -> None:
        transformed = self.__keytransform__(key)
        closures = self._closures.pop(transformed, None)
        if dict.__contains__(self, transformed):
            dict.__delitem__(self, transformed)
        elif closures is None:
            raise KeyError(key)

    def pop(self, key: DateLike, default=None):
        """Remove a date, dropping its closure records as well."""
        self._closures.pop(self.__keytransform__(key), None)
        return super().pop(key, default) if default is not None else super().pop(key)
