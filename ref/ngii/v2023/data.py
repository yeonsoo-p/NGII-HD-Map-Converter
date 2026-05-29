from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from ngii.v2023.layer import BaseLayer

logger = logging.getLogger(__name__)

_FLOAT_TEXT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(slots=True, eq=False, kw_only=True)
class BaseData:
    id: str
    admin_code: str = ""
    maker: str = ""
    update_date: str | None = None
    version: str = ""
    remark: str | None = None
    hist_type: str | None = None
    hist_remark: str | None = None
    layer: BaseLayer[Any] | None = field(default=None, repr=False, compare=False)

    def __str__(self) -> str:
        return self.id


@dataclass(slots=True, eq=False, kw_only=True)
class UnresolvedData(BaseData):
    id: str = "-"
    layer_name: str = "-"
    reason: str = ""
    source_id: str | None = None


def parse_enum[T: StrEnum](
    enum_type: type[T],
    value: object,
    source_id: str,
    field_name: str,
) -> T | None:
    code = optional_code(value)
    if code is None:
        log_row_issue(
            "warning",
            source_id,
            f"Missing required {field_name}; storing None.",
        )
        return None
    member = enum_member(enum_type, code)
    if member is not None:
        return member
    log_row_issue(
        "warning",
        source_id,
        f"Invalid required {field_name}: {code}; storing None.",
    )
    return None


def parse_optional_enum[T: StrEnum](enum_type: type[T], value: object) -> T | None:
    code = optional_code(value)
    if code is None:
        return None
    return enum_member(enum_type, code)


def enum_member[T: StrEnum](enum_type: type[T], code: str) -> T | None:
    for member in enum_type:
        if member.value == code:
            return member
    return None


def row_id(row: Any) -> str:
    value = optional_text(row_value(row, "ID"))
    return value if value is not None else "-"


def row_value(row: Any, *names: str) -> object:
    if isinstance(row, Mapping):
        for name in names:
            value = row.get(name)
            if not is_null(value):
                return value
        casefolded = {name.casefold() for name in names}
        for column, value in row.items():
            if str(column).casefold() in casefolded and not is_null(value):
                return value
        return None

    row_index = getattr(row, "index", None)
    if row_index is None:
        return None

    columns = list(row_index)
    for name in names:
        if name in columns:
            value = row[name]
            if not is_null(value):
                return value

    casefolded = {name.casefold() for name in names}
    for column in columns:
        if str(column).casefold() not in casefolded:
            continue
        value = row[column]
        if not is_null(value):
            return value
    return None


def text(row: Any, field_name: str) -> str:
    return optional_text(row_value(row, field_name)) or ""


def optional_text(value: object) -> str | None:
    if is_null(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip() or None
    value_text = str(value).strip()
    return value_text or None


def optional_code(value: object) -> str | None:
    if is_null(value):
        return None
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return str(int(value)) if value.is_integer() else str(value).strip()
    value_text = optional_text(value)
    if value_text is None:
        return None
    if _is_integral_decimal_text(value_text):
        return value_text[:-2]
    return value_text


def optional_int(value: object) -> int | None:
    code = optional_code(value)
    if code is None:
        return None
    if _is_int_text(code):
        return int(code)
    if _is_integral_decimal_text(code):
        return int(code[:-2])
    return None


def int_or_zero(value: object) -> int:
    return optional_int(value) or 0


def optional_float(value: object) -> float | None:
    if is_null(value):
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, Real):
        number = float(value)
    else:
        value_text = optional_text(value)
        if value_text is None or _FLOAT_TEXT_PATTERN.fullmatch(value_text) is None:
            return None
        number = float(value_text)
    if math.isnan(number):
        return None
    return number


def float_or_zero(value: object) -> float:
    return optional_float(value) or 0.0


def is_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float | np.floating):
        return math.isnan(value)
    value_type = type(value)
    return value_type.__module__.startswith("pandas") and value_type.__name__ in {
        "NAType",
        "NaTType",
    }


def log_row_issue(
    level: Literal["warning", "error"],
    source_id: str,
    message: str,
) -> None:
    formatted = f"{source_id}: {message}"
    if level == "error":
        logger.error(formatted)
    else:
        logger.warning(formatted)


def _is_int_text(value: str) -> bool:
    return value.removeprefix("+").removeprefix("-").isdigit()


def _is_integral_decimal_text(value: str) -> bool:
    if not value.endswith(".0"):
        return False
    return _is_int_text(value[:-2])
