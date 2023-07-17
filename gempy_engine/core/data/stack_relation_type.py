from __future__ import annotations

import enum


class StackRelationType(enum.Enum):
    ERODE = enum.auto()
    ONLAP = enum.auto()
    FAULT = enum.auto()
    BASEMENT = enum.auto()
