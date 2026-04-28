#!/usr/bin/env python3
"""Armenian text frontend for TTS.

The acoustic model should see pronounceable text, not written shortcuts such as
``02.02.2026`` or ``25%``.  This module keeps that logic in one place so the
same frontend can be used for manifest preparation and inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, List, Optional


ARMENIAN_CHAR_RE = re.compile(r"[\u0530-\u058f]")
SPACE_RE = re.compile(r"\s+")

REPLACEMENTS = {
    "\u00a0": " ",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}

LANGUAGE_IDS_ARMENIAN = {
    "hy",
    "hye",
    "hyw",
    "hye-east",
    "hyw-west",
    "armenian",
    "eastern armenian",
    "western armenian",
}

ONES = {
    0: "զրո",
    1: "մեկ",
    2: "երկու",
    3: "երեք",
    4: "չորս",
    5: "հինգ",
    6: "վեց",
    7: "յոթ",
    8: "ութ",
    9: "ինը",
}

TEENS = {
    10: "տասը",
    11: "տասնմեկ",
    12: "տասներկու",
    13: "տասներեք",
    14: "տասնչորս",
    15: "տասնհինգ",
    16: "տասնվեց",
    17: "տասնյոթ",
    18: "տասնութ",
    19: "տասնինը",
}

TENS = {
    20: "քսան",
    30: "երեսուն",
    40: "քառասուն",
    50: "հիսուն",
    60: "վաթսուն",
    70: "յոթանասուն",
    80: "ութսուն",
    90: "իննսուն",
}

MONTHS_GENITIVE = {
    1: "հունվարի",
    2: "փետրվարի",
    3: "մարտի",
    4: "ապրիլի",
    5: "մայիսի",
    6: "հունիսի",
    7: "հուլիսի",
    8: "օգոստոսի",
    9: "սեպտեմբերի",
    10: "հոկտեմբերի",
    11: "նոյեմբերի",
    12: "դեկտեմբերի",
}

DAY_DATIVE = {
    1: "մեկին",
    2: "երկուսին",
    3: "երեքին",
    4: "չորսին",
    5: "հինգին",
    6: "վեցին",
    7: "յոթին",
    8: "ութին",
    9: "իննին",
    10: "տասին",
    11: "տասնմեկին",
    12: "տասներկուսին",
    13: "տասներեքին",
    14: "տասնչորսին",
    15: "տասնհինգին",
    16: "տասնվեցին",
    17: "տասնյոթին",
    18: "տասնութին",
    19: "տասնիննին",
    20: "քսանին",
    21: "քսանմեկին",
    22: "քսաներկուսին",
    23: "քսաներեքին",
    24: "քսանչորսին",
    25: "քսանհինգին",
    26: "քսանվեցին",
    27: "քսանյոթին",
    28: "քսանութին",
    29: "քսանիննին",
    30: "երեսունին",
    31: "երեսունմեկին",
}

ORDINALS = {
    1: "առաջին",
    2: "երկրորդ",
    3: "երրորդ",
    4: "չորրորդ",
    5: "հինգերորդ",
    6: "վեցերորդ",
    7: "յոթերորդ",
    8: "ութերորդ",
    9: "իններորդ",
    10: "տասներորդ",
    20: "քսաներորդ",
    30: "երեսուներորդ",
    40: "քառասուներորդ",
    50: "հիսուներորդ",
    60: "վաթսուներորդ",
    70: "յոթանասուներորդ",
    80: "ութսուներորդ",
    90: "իննսուներորդ",
    100: "հարյուրերորդ",
    1000: "հազարերորդ",
}

CURRENCY_NAMES = {
    "֏": "դրամ",
    "դրամ": "դրամ",
    "դր": "դրամ",
    "դր.": "դրամ",
    "amd": "դրամ",
    "$": "դոլար",
    "usd": "դոլար",
    "€": "եվրո",
    "eur": "եվրո",
    "£": "ֆունտ",
    "gbp": "ֆունտ",
    "₽": "ռուբլի",
    "rub": "ռուբլի",
}

LETTER_DIGIT = r"A-Za-zԱ-Ֆա-ֆևԵՎՙ՚՛՜՝՞՟\d_"
LEFT_BOUNDARY = rf"(?<![{LETTER_DIGIT}])"
RIGHT_BOUNDARY = rf"(?![{LETTER_DIGIT}])"

DATE_SUFFIX = r"(?:\s*-?\s*(?:ին|թ\.?|թվականին))?"
DATE_DMY_RE = re.compile(
    rf"{LEFT_BOUNDARY}([0-3]?\d)[./-]([01]?\d)[./-]((?:18|19|20|21)\d{{2}}){DATE_SUFFIX}{RIGHT_BOUNDARY}"
)
DATE_YMD_RE = re.compile(
    rf"{LEFT_BOUNDARY}((?:18|19|20|21)\d{{2}})[./-]([01]?\d)[./-]([0-3]?\d){DATE_SUFFIX}{RIGHT_BOUNDARY}"
)
TIME_RE = re.compile(
    rf"{LEFT_BOUNDARY}(?:ժամը\s*)?([0-2]?\d):([0-5]\d)(?::([0-5]\d))?{RIGHT_BOUNDARY}"
)
PERCENT_RE = re.compile(
    rf"{LEFT_BOUNDARY}([+-]?(?:\d{{1,3}}(?:[ ,]\d{{3}})+|\d+)(?:[.,]\d+)?)\s*(?:%|տոկոս){RIGHT_BOUNDARY}",
    re.IGNORECASE,
)
CURRENCY_PREFIX_RE = re.compile(
    rf"{LEFT_BOUNDARY}([$€£₽֏])\s*([+-]?(?:\d{{1,3}}(?:[ ,]\d{{3}})+|\d+)(?:[.,]\d+)?)"
)
CURRENCY_SUFFIX_RE = re.compile(
    rf"{LEFT_BOUNDARY}([+-]?(?:\d{{1,3}}(?:[ ,]\d{{3}})+|\d+)(?:[.,]\d+)?)\s*(֏|դրամ|դր\.?|AMD|USD|EUR|GBP|RUB|\$|€|£|₽){RIGHT_BOUNDARY}",
    re.IGNORECASE,
)
ORDINAL_RE = re.compile(rf"{LEFT_BOUNDARY}(\d+)\s*[-\u2010-\u2015]?\s*(?:րդ|ին){RIGHT_BOUNDARY}")
RANGE_RE = re.compile(rf"{LEFT_BOUNDARY}(\d+)\s*[-\u2010-\u2015]\s*(\d+){RIGHT_BOUNDARY}")
PLAIN_NUMBER_RE = re.compile(
    rf"{LEFT_BOUNDARY}([+-]?(?:\d{{1,3}}(?:[ ,]\d{{3}})+|\d+)(?:[.,]\d+)?)"
    rf"{RIGHT_BOUNDARY}"
)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SYMBOL_RE = re.compile(r"[@#&=+*/\\|<>_~^]")
LATIN_TOKEN_RE = re.compile(rf"{LEFT_BOUNDARY}[A-Za-z]{{2,}}{RIGHT_BOUNDARY}")


@dataclass(frozen=True)
class TextIssue:
    kind: str
    value: str
    start: int
    end: int

    def as_dict(self) -> dict:
        return asdict(self)


def looks_armenian_context(text: str, language: Optional[str] = None) -> bool:
    if language and language.strip().lower() in LANGUAGE_IDS_ARMENIAN:
        return True
    return bool(ARMENIAN_CHAR_RE.search(text))


def normalize_unicode_text(text: str) -> str:
    s = str(text)
    for old, new in REPLACEMENTS.items():
        s = s.replace(old, new)
    s = SPACE_RE.sub(" ", s.strip())
    return s


def integer_to_armenian(value: int) -> str:
    if value < 0:
        return "մինուս " + integer_to_armenian(abs(value))
    if value < 10:
        return ONES[value]
    if value < 20:
        return TEENS[value]
    if value < 100:
        tens = value // 10 * 10
        ones = value % 10
        return TENS[tens] + (ONES[ones] if ones else "")
    if value < 1000:
        hundreds = value // 100
        rest = value % 100
        prefix = "հարյուր" if hundreds == 1 else f"{integer_to_armenian(hundreds)} հարյուր"
        return prefix if rest == 0 else f"{prefix} {integer_to_armenian(rest)}"

    for scale, name in (
        (1_000_000_000, "միլիարդ"),
        (1_000_000, "միլիոն"),
        (1000, "հազար"),
    ):
        if value >= scale:
            head = value // scale
            rest = value % scale
            if scale == 1000 and head == 1:
                prefix = name
            else:
                prefix = f"{integer_to_armenian(head)} {name}"
            return prefix if rest == 0 else f"{prefix} {integer_to_armenian(rest)}"

    raise ValueError(f"Unsupported integer value: {value}")


def _strip_group_separators(value: str) -> str:
    return re.sub(r"(?<=\d)[ ,](?=\d{3}(?:\D|$))", "", value)


def number_to_armenian(value: str | int) -> str:
    raw = str(value).strip()
    if not raw:
        return raw

    sign = ""
    if raw[0] in "+-":
        sign = "մինուս " if raw[0] == "-" else ""
        raw = raw[1:]

    raw = _strip_group_separators(raw)

    decimal_match = re.fullmatch(r"(\d+)[.,](\d+)", raw)
    if decimal_match:
        whole, frac = decimal_match.groups()
        frac_words = " ".join(ONES[int(ch)] for ch in frac)
        return f"{sign}{integer_to_armenian(int(whole))} ամբողջ {frac_words}".strip()

    return f"{sign}{integer_to_armenian(int(raw))}".strip()


def ordinal_to_armenian(value: int) -> str:
    if value in ORDINALS:
        return ORDINALS[value]
    if value < 100:
        return integer_to_armenian(value) + "երորդ"

    words = integer_to_armenian(value).split()
    words[-1] = ordinal_to_armenian(int_to_last_component(value))
    return " ".join(words)


def int_to_last_component(value: int) -> int:
    if value % 100:
        return value % 100
    if value % 1000:
        return value % 1000
    if value % 1_000_000:
        return value % 1_000_000
    return value


def day_to_date_armenian(day: int) -> str:
    if day in DAY_DATIVE:
        return DAY_DATIVE[day]
    if not 1 <= day <= 31:
        raise ValueError(f"Invalid day: {day}")
    raise ValueError(f"Invalid day: {day}")


def expand_numeric_date(day: int, month: int, year: int) -> str:
    if month not in MONTHS_GENITIVE or not 1 <= day <= 31:
        raise ValueError(f"Invalid date: {day}.{month}.{year}")
    # Keep validation lightweight; this frontend is a normalizer, not a calendar.
    return f"{MONTHS_GENITIVE[month]} {day_to_date_armenian(day)}, {integer_to_armenian(year)} թվականին"


def _replace_dmy(match: re.Match[str]) -> str:
    day, month, year = (int(x) for x in match.groups())
    try:
        return expand_numeric_date(day, month, year)
    except ValueError:
        return match.group(0)


def _replace_ymd(match: re.Match[str]) -> str:
    year, month, day = (int(x) for x in match.groups())
    try:
        return expand_numeric_date(day, month, year)
    except ValueError:
        return match.group(0)


def _replace_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3)) if match.group(3) is not None else None
    if hour > 23:
        return match.group(0)
    text = f"ժամը {integer_to_armenian(hour)}"
    if minute:
        text += f" անց {integer_to_armenian(minute)}"
    if second:
        text += f" և {integer_to_armenian(second)} վայրկյան"
    return text


def _replace_percent(match: re.Match[str]) -> str:
    return f"{number_to_armenian(match.group(1))} տոկոս"


def _replace_currency_prefix(match: re.Match[str]) -> str:
    currency = CURRENCY_NAMES[match.group(1).lower()]
    return f"{number_to_armenian(match.group(2))} {currency}"


def _replace_currency_suffix(match: re.Match[str]) -> str:
    currency = CURRENCY_NAMES[match.group(2).lower()]
    return f"{number_to_armenian(match.group(1))} {currency}"


def _replace_ordinal(match: re.Match[str]) -> str:
    return ordinal_to_armenian(int(match.group(1)))


def _replace_range(match: re.Match[str]) -> str:
    return f"{number_to_armenian(match.group(1))}ից {number_to_armenian(match.group(2))}"


def _replace_number(match: re.Match[str]) -> str:
    try:
        return number_to_armenian(match.group(1))
    except ValueError:
        return match.group(0)


def expand_armenian_text(text: str) -> str:
    s = normalize_unicode_text(text)
    s = DATE_DMY_RE.sub(_replace_dmy, s)
    s = DATE_YMD_RE.sub(_replace_ymd, s)
    s = TIME_RE.sub(_replace_time, s)
    s = CURRENCY_PREFIX_RE.sub(_replace_currency_prefix, s)
    s = CURRENCY_SUFFIX_RE.sub(_replace_currency_suffix, s)
    s = PERCENT_RE.sub(_replace_percent, s)
    s = ORDINAL_RE.sub(_replace_ordinal, s)
    s = RANGE_RE.sub(_replace_range, s)
    s = PLAIN_NUMBER_RE.sub(_replace_number, s)
    return cleanup_spacing(s)


def cleanup_spacing(text: str) -> str:
    s = SPACE_RE.sub(" ", text.strip())
    s = re.sub(r"\s+([,.;:!?։՝՞՜])", r"\1", s)
    s = re.sub(r"([,;:!?։])(?=\S)", r"\1 ", s)
    return SPACE_RE.sub(" ", s).strip()


def normalize_for_tts(text: str, language: Optional[str] = "hy") -> str:
    s = normalize_unicode_text(text)
    if looks_armenian_context(s, language=language):
        return expand_armenian_text(s)
    return cleanup_spacing(s)


def _iter_issue_matches(text: str) -> Iterable[TextIssue]:
    patterns = [
        ("url", URL_RE),
        ("email", EMAIL_RE),
        ("date", DATE_DMY_RE),
        ("date", DATE_YMD_RE),
        ("time", TIME_RE),
        ("currency", CURRENCY_PREFIX_RE),
        ("currency", CURRENCY_SUFFIX_RE),
        ("percent", PERCENT_RE),
        ("ordinal", ORDINAL_RE),
        ("range", RANGE_RE),
        ("number", PLAIN_NUMBER_RE),
        ("symbol", SYMBOL_RE),
        ("latin_token", LATIN_TOKEN_RE),
    ]
    occupied: list[tuple[int, int]] = []
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < old_end and end > old_start for old_start, old_end in occupied):
                continue
            occupied.append((start, end))
            yield TextIssue(kind=kind, value=match.group(0), start=start, end=end)


def find_text_frontend_issues(
    text: str,
    language: Optional[str] = "hy",
) -> List[TextIssue]:
    s = normalize_unicode_text(text)
    if not looks_armenian_context(s, language=language):
        return []
    return sorted(_iter_issue_matches(s), key=lambda issue: (issue.start, issue.end))


def issues_as_dicts(issues: Iterable[TextIssue]) -> list[dict]:
    return [issue.as_dict() for issue in issues]
