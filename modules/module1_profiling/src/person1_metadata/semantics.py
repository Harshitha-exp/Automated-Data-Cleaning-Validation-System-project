"""
semantics.py
-------------
Person 1 — Metadata Extraction Engine: semantic detection.

Public function: detect_semantics(df, types=None) -> dict[str, dict]

Detects what a column MEANS — name, email, phone, date, ID, URL, currency,
zip (the brief's required list), plus a few more that fall naturally out
of the same approach (address, ip_address, national_id, credit_card,
categorical, numeric, boolean, text).

CONTRACT WITH PERSON 3's RULE ENGINE (engine.py / column_rules.py / pii.py):
This module's output feeds the "metadata" argument of their run_rules(df,
metadata). Their code reads it as:

    metadata["columns"][col]["semantic"]      # a lowercase string, or None

Two of their rules key off the literal string "id" (near_unique and
extreme_skew both skip a column when semantic == "id" — ID columns are
*supposed* to be unique / have weird distributions, so flagging them is
noise). Their PII module also recognises a specific vocabulary as a 4th
corroborating signal (see SEMANTIC_TO_TYPE in their pii.py): "email",
"phone"/"phone_number", "name"/"person_name"/"full_name", "ssn"/
"national_id", "credit_card", "ip_address"/"ip", "dob"/"date_of_birth",
"address". This module deliberately emits "id", "email", "phone", and
"person_name" to match that vocabulary exactly.

One known, intentional gap: this module can emit "zip" as its own
semantic (more precise than lumping it into "address", and the brief asks
for "zip" specifically) but Person 3's SEMANTIC_TO_TYPE map only has
"address", not "zip" — so a pure zip-code column won't get PII's bonus
"semantic label" signal (it'll still very likely get flagged via its own
column-name + regex signals independently). Worth a one-line addition on
their side if exact parity matters; not a functional break either way.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .type_inference import infer_types

# ---------------------------------------------------------------------------
# Regex patterns for content-based detection. Compiled once at import time.
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"^[\w.+\-]+@[\w\-]+(\.[\w\-]+)+$")
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://\S+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().\-]{7,20}$")
_US_ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")
_UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.IGNORECASE)
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_IPV4_RE = re.compile(r"^((25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(25[0-5]|2[0-4]\d|1?\d?\d)$")
_CREDIT_CARD_RE = re.compile(r"^[\d \-]{13,23}$")
_CURRENCY_RE = re.compile(r"^[\$£€]\s?-?\d[\d,]*(\.\d+)?$|^-?\d[\d,]*(\.\d+)?\s?(USD|GBP|EUR)$", re.IGNORECASE)
_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y",
    "%Y-%m-%d %H:%M:%S", "%d %b %Y", "%b %d, %Y", "%B %d, %Y",
)

# Column-name WHOLE-WORD tokens. Matched against tokens split on any
# non-alphanumeric character (see _tokenize) — never raw substrings, which
# would false-match e.g. "id" inside "Subsidy" or "name" inside "username"
# vs "county name" ambiguity. This is the same bug class Person 3 and I
# both had to fix independently; tokenizing first avoids it.
_ID_HINTS = {"id", "code", "fips", "key", "uuid"}
_EMAIL_HINTS = {"email", "mail"}
_PHONE_HINTS = {"phone", "mobile", "tel", "telephone", "fax", "cell"}
_URL_HINTS = {"url", "link", "website", "uri"}
# Deliberately narrow: an explicit currency UNIT token in the column name
# (e.g. "(US$)" tokenizes to "us"), not vague money-adjacent words like
# "payment"/"amount"/"cost" — those false-positive badly (e.g. "Payment
# acreage" is a land-area measure, not money, but contains "payment").
# Content-based corroboration isn't available here either: by the time a
# column is numeric dtype, pandas/CSV parsing has already stripped any "$"
# formatting from the original values, so the column name is genuinely
# the only signal left.
_CURRENCY_HINTS = {"usd", "gbp", "eur", "us", "uk", "dollar", "dollars",
                    "pound", "pounds", "euro", "euros", "currency"}
# Geographic/administrative code columns: int-typed but semantically an
# ID, not a continuous measure (FIPS state/county codes, zone/district
# numbers). Kept SEPARATE from the general _ID_HINTS set used by
# _name_hint() for text columns — "State name"/"County name" must stay
# "address"/"categorical", not get pulled into "id" by the word "state".
_GEO_CODE_HINTS = {"state", "county", "district", "region", "zone", "fips"}
_ZIP_HINTS = {"zip", "zipcode", "postcode", "postal"}
_ADDRESS_HINTS = {"address", "street"}
_DATE_HINTS = {"date", "dob", "timestamp"}
_IP_HINTS = {"ip", "ipaddress"}
_NATIONAL_ID_HINTS = {"ssn", "nin", "passport", "nino"}
_CREDIT_CARD_HINTS = {"creditcard", "cardnumber", "ccnum"}

# "name" alone is ambiguous: "State name" / "County name" / "Commodity
# name" are place/entity names, not a *person's* name. Only call it
# person_name when "name" is unqualified or paired with a person-ish
# qualifier — same guard Person 3 built independently in pii.py
# (_person_name_hint), kept consistent here on purpose.
_PERSON_NAME_QUALIFIERS = {
    "first", "last", "full", "sur", "given", "family", "middle", "fore",
    "person", "customer", "client", "farmer", "holder", "policyholder", "insured",
}
_NON_PERSON_NAME_QUALIFIERS = {
    "state", "county", "city", "country", "place", "district", "region",
    "company", "product", "item", "account", "bank", "branch", "store",
    "school", "team", "brand", "crop", "commodity", "category",
}

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(col_name: str) -> set[str]:
    """'Payment indemnity (US$)' -> {'payment','indemnity','us'}."""
    return {t for t in _TOKEN_SPLIT_RE.split(str(col_name).lower()) if t}


def _is_person_name_column(tokens: set[str]) -> bool:
    if "name" not in tokens:
        return False
    if tokens & _NON_PERSON_NAME_QUALIFIERS:
        return False
    # bare "name", or "name" + a person-ish qualifier
    return len(tokens) == 1 or bool(tokens & _PERSON_NAME_QUALIFIERS)


def _classify_date(value: str) -> bool:
    from datetime import datetime
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False


def _name_hint(tokens: set[str]) -> str | None:
    """Column-name-only semantic guess, checked before content sampling."""
    if tokens & _EMAIL_HINTS:
        return "email"
    if tokens & _PHONE_HINTS:
        return "phone"
    if tokens & _URL_HINTS:
        return "url"
    if tokens & _IP_HINTS:
        return "ip_address"
    if tokens & _NATIONAL_ID_HINTS:
        return "national_id"
    if tokens & _CREDIT_CARD_HINTS:
        return "credit_card"
    if tokens & _ZIP_HINTS:
        return "zip"
    if tokens & _ADDRESS_HINTS:
        return "address"
    if tokens & _DATE_HINTS:
        return "date"
    if _is_person_name_column(tokens):
        return "person_name"
    if tokens & _ID_HINTS:
        return "id"
    if tokens & _CURRENCY_HINTS:
        return "currency"
    return None


def _content_sample_semantic(series: pd.Series, sample_size: int = 500) -> tuple[str | None, float]:
    """Classify a column by sampling its values against the regex patterns.
    Returns (semantic_or_None, share_of_sample_matching)."""
    non_null = series.dropna()
    if non_null.empty:
        return None, 0.0
    sample = non_null.sample(min(sample_size, len(non_null)), random_state=42).astype(str).str.strip()
    n = len(sample)
    if n == 0:
        return None, 0.0

    checks = (
        ("email", lambda v: bool(_EMAIL_RE.match(v))),
        ("url", lambda v: bool(_URL_RE.match(v))),
        ("ip_address", lambda v: bool(_IPV4_RE.match(v))),
        ("national_id", lambda v: bool(_SSN_RE.match(v))),
        ("zip", lambda v: bool(_US_ZIP_RE.match(v)) or bool(_UK_POSTCODE_RE.match(v))),
        ("currency", lambda v: bool(_CURRENCY_RE.match(v))),
        ("date", _classify_date),
        ("phone", lambda v: bool(_PHONE_RE.match(v)) and bool(re.search(r"[\s()+\-.]", v))
                             and 7 <= sum(c.isdigit() for c in v) <= 15),
    )

    best_semantic, best_share = None, 0.0
    for semantic, fn in checks:
        matches = sum(1 for v in sample if fn(v))
        share = matches / n
        if share > best_share:
            best_semantic, best_share = semantic, share

    return best_semantic, best_share


def detect_semantics(df: pd.DataFrame, types: dict[str, dict] | None = None, sample_size: int = 500) -> dict[str, dict[str, Any]]:
    """
    Infer the semantic meaning of every column in `df`.

    Parameters
    ----------
    df : pd.DataFrame
    types : dict | None
        Output of infer_types(df). Computed internally if not supplied
        (pass it in from metadata.build_metadata to avoid recomputing).
    sample_size : int
        How many non-null values to content-sample per column for the
        regex-based checks (email/phone/date/etc.). Column-name hints are
        checked first and don't need sampling at all.

    Returns
    -------
    {
        "<column_name>": {
            "semantic": "<string, see module docstring for vocabulary, or None>",
            "confidence": <float 0-1>,
            "basis": ["column_name" | "content_sample", ...],
        },
        ...
    }

    `semantic` is intentionally `None` (not "unknown" or "text") when
    nothing — name or content — gives a confident signal, so downstream
    code (Person 3's `get_semantic`) can tell "no opinion" apart from "we
    looked and it's genuinely unclassifiable text".
    """
    types = types or infer_types(df)
    result: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        tokens = _tokenize(col)
        structural_type = types.get(col, {}).get("type")
        series = df[col]

        name_guess = _name_hint(tokens)

        # Structural type overrides/short-circuits some cases outright —
        # a real datetime64 column IS a date, full stop, no need to sample.
        if structural_type == "datetime":
            result[col] = {"semantic": "date", "confidence": 0.99, "basis": ["dtype"]}
            continue
        if structural_type == "boolean":
            result[col] = {"semantic": "boolean", "confidence": 0.99, "basis": ["dtype"]}
            continue

        # Numeric columns: identifier vs currency vs plain numeric measure.
        # Float dtype is essentially never an identifier in practice (a
        # continuous measure like a dollar amount is naturally almost all
        # unique — that's normal, not an ID signal) so only integer dtype
        # is eligible for the high-cardinality-implies-id heuristic.
        if structural_type in ("integer", "float"):
            cardinality_ratio = types.get(col, {}).get("cardinality_ratio", 0.0)
            has_currency_unit = bool(tokens & _CURRENCY_HINTS)

            if name_guess == "id":
                result[col] = {"semantic": "id", "confidence": 0.85, "basis": ["column_name"]}
                continue
            # Currency unit token wins over everything else for numeric
            # columns — "Payment indemnity (US$)" has both a vague money
            # word ("payment"/"indemnity") AND an explicit unit ("us");
            # only the unit token is trusted (see _CURRENCY_HINTS comment).
            if has_currency_unit:
                result[col] = {"semantic": "currency", "confidence": 0.75, "basis": ["column_name"]}
                continue
            if structural_type == "integer" and tokens & _GEO_CODE_HINTS:
                # Low-cardinality int code with a geographic/administrative
                # name (State, County, FIPS, ...) — an identifier, not a
                # continuous measure, regardless of its cardinality ratio.
                result[col] = {"semantic": "id", "confidence": 0.8, "basis": ["column_name"]}
                continue
            if structural_type == "integer" and cardinality_ratio > 0.9 and len(df) > 1:
                result[col] = {"semantic": "id", "confidence": 0.55, "basis": ["cardinality"]}
                continue
            result[col] = {"semantic": "numeric", "confidence": 0.8, "basis": ["dtype"]}
            continue

        # Text/categorical columns: name hint first, then content sample;
        # agreement between the two boosts confidence, disagreement keeps
        # whichever is stronger rather than silently picking one.
        content_guess, content_share = _content_sample_semantic(series, sample_size)

        if name_guess and content_guess and name_guess == content_guess:
            result[col] = {"semantic": name_guess, "confidence": min(1.0, 0.7 + content_share * 0.3),
                            "basis": ["column_name", "content_sample"]}
            continue
        if content_guess and content_share >= 0.7:
            basis = ["content_sample"] + (["column_name"] if name_guess else [])
            result[col] = {"semantic": content_guess, "confidence": content_share, "basis": basis}
            continue
        if name_guess:
            result[col] = {"semantic": name_guess, "confidence": 0.6, "basis": ["column_name"]}
            continue

        # No strong signal either way: fall back to the structural type's
        # own category rather than guessing.
        if structural_type == "categorical":
            result[col] = {"semantic": "categorical", "confidence": 0.5, "basis": ["cardinality"]}
        else:
            result[col] = {"semantic": None, "confidence": 0.0, "basis": []}

    return result
