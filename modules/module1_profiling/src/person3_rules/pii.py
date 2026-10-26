"""PII detection (Person 3).

Finds columns that probably hold personal data and gives each a confidence score.
Three signals are combined:
  1. name  : the column name hints at it (e.g. 'email', 'phone', 'dob')
  2. regex : the values match a known pattern (email, phone, card number...)
  3. presidio : Microsoft Presidio recognises the values (checked on a small sample)
A fourth, weaker signal is Person 1's semantic label, if one was given.

No single weak signal gives high confidence: Presidio alone never flags a column.
Flags name the column and the TYPE of PII only. Real values are never put in a flag.
"""
import importlib.util
import re
import warnings

import pandas as pd

from column_rules import make_flag, name_words, get_semantic
from formats import looks_like_phone

RULE_NAME = "pii"

EMAIL_RE = re.compile(r"^[\w.+\-]+@[\w\-]+(\.[\w\-]+)+$")
SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
UK_NI_RE = re.compile(r"^[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]$")
IPV4_RE = re.compile(r"^((25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(25[0-5]|2[0-4]\d|1?\d?\d)$")


def _luhn_ok(value):
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19 or not re.fullmatch(r"[\d \-]+", value):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_national_id(value):
    return bool(SSN_RE.match(value) or UK_NI_RE.match(value.replace(" ", "").upper()))


def _person_name_hint(col):
    """'name', 'full_name', 'first_name'... but not 'county_name' or 'crop_name'."""
    words = name_words(col)
    compact = re.sub(r"[^a-z0-9]", "", str(col).lower())
    if compact in {"name", "fullname", "firstname", "lastname", "surname", "forename"}:
        return True
    qualifiers = {"first", "last", "full", "sur", "given", "family", "middle", "fore",
                  "person", "customer", "client", "farmer", "holder", "policyholder", "insured"}
    return "name" in words and bool(words & qualifiers)


# What we know about each kind of PII.
# hints = whole words in the column name; compact = text found inside the squashed name.
PII_TYPES = {
    "email": {"hints": {"email", "emailaddress"}, "compact": ["email"],
              "regex": lambda v: bool(EMAIL_RE.match(v)), "presidio": "EMAIL_ADDRESS"},
    "phone": {"hints": {"phone", "mobile", "tel", "telephone", "fax", "cell"},
              "compact": ["phone", "mobile"],
              "regex": looks_like_phone, "presidio": "PHONE_NUMBER"},
    "person_name": {"hint_fn": _person_name_hint, "regex": None, "presidio": "PERSON"},
    "national_id": {"hints": {"ssn", "nin", "passport", "nhs"},
                    "compact": ["socialsecurity", "nationalinsurance", "passport"],
                    "regex": _is_national_id, "presidio": "US_SSN"},
    "credit_card": {"hints": set(), "compact": ["creditcard", "cardnumber"],
                    "regex": _luhn_ok, "presidio": "CREDIT_CARD"},
    "ip_address": {"hints": {"ip"}, "compact": ["ipaddress"],
                   "regex": lambda v: bool(IPV4_RE.match(v)), "presidio": "IP_ADDRESS"},
    "date_of_birth": {"hints": {"dob", "birthday", "birthdate", "birth"},
                      "compact": ["dateofbirth", "birthdate"], "regex": None, "presidio": None},
    "address": {"hints": {"address", "street", "postcode", "postal", "zip", "zipcode"},
                "compact": [], "regex": None, "presidio": None},
}

# Person 1's possible labels -> our PII types
SEMANTIC_TO_TYPE = {
    "email": "email", "phone": "phone", "phone_number": "phone",
    "name": "person_name", "person_name": "person_name", "full_name": "person_name",
    "ssn": "national_id", "national_id": "national_id",
    "credit_card": "credit_card", "ip_address": "ip_address", "ip": "ip_address",
    "dob": "date_of_birth", "date_of_birth": "date_of_birth", "address": "address",
}

_analyzer_cache = {}


def _get_cfg(cfg):
    if cfg is not None:
        return cfg
    from engine import load_config   # imported here to avoid a circular import
    return load_config()["rules"].get(RULE_NAME, {})


def _get_analyzer(model_name):
    """Load Presidio once (it is slow to start). Returns None if it can't be used."""
    if model_name in _analyzer_cache:
        return _analyzer_cache[model_name]
    analyzer = None
    try:
        # Check the spaCy model is installed BEFORE Presidio sees it: if it is missing,
        # Presidio tries to download it and then shuts the whole program down.
        if importlib.util.find_spec(model_name) is None:
            raise OSError(f"spaCy model '{model_name}' is not installed "
                          f"(python -m spacy download {model_name})")
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        nlp = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    except (Exception, SystemExit) as e:   # SystemExit: libraries that quit on failure
        warnings.warn(f"Presidio unavailable ({type(e).__name__}: {e}); PII detection "
                      f"will use column names and regex only.")
    _analyzer_cache[model_name] = analyzer
    return analyzer


def _name_hint(col, spec):
    if "hint_fn" in spec:
        return spec["hint_fn"](col)
    compact = re.sub(r"[^a-z0-9]", "", str(col).lower())
    return bool(name_words(col) & spec["hints"]) or any(c in compact for c in spec["compact"])


def _presidio_shares(values, analyzer, cfg):
    """Share of sampled values in which Presidio found each entity type."""
    wanted = sorted({s["presidio"] for s in PII_TYPES.values() if s["presidio"]})
    min_score = cfg.get("presidio_min_score", 0.5)
    sample = values.drop_duplicates()
    sample = sample[~sample.str.fullmatch(r"-?\d+(\.\d+)?")]      # skip plain numbers
    if len(sample) == 0:
        return {}
    sample = sample.sample(min(len(sample), cfg.get("presidio_sample", 50)), random_state=0)
    hits = {}
    for text in sample:
        found = {r.entity_type for r in analyzer.analyze(text=text, language="en",
                                                         entities=wanted)
                 if r.score >= min_score}
        for entity in found:
            hits[entity] = hits.get(entity, 0) + 1
    return {entity: n / len(sample) for entity, n in hits.items()}


def detect_pii(df, cfg=None, metadata=None):
    """Find columns that probably contain personal data.

    df       : the dataset
    cfg      : this rule's settings (optional; read from rules.yaml if missing)
    metadata : Person 1's column information (optional extra signal)
    Returns a list of flags in the standard format (one per PII column).
    """
    cfg = _get_cfg(cfg)
    severity = cfg.get("severity", "high")
    weights = {"name": 0.55, "regex": 0.6, "presidio": 0.45, "semantic": 0.3}
    weights.update(cfg.get("weights", {}))
    min_conf = cfg.get("min_confidence", 0.5)
    regex_min = cfg.get("regex_min_share", 0.5)
    presidio_min = cfg.get("presidio_min_share", 0.3)
    sample_size = cfg.get("sample_size", 5000)
    hint_required = set(cfg.get("name_hint_required", ["person_name"]))
    min_rows = cfg.get("min_rows", 5)

    analyzer = _get_analyzer(cfg.get("presidio_model", "en_core_web_sm")) \
        if cfg.get("use_presidio", True) else None

    flags = []
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        if len(values) > sample_size:                  # random sample, not the first rows
            values = values.sample(sample_size, random_state=0)
        has_rows = len(values) >= min_rows

        is_text = not pd.api.types.is_numeric_dtype(df[col])
        presidio = _presidio_shares(values, analyzer, cfg) if (analyzer and has_rows and is_text) else {}
        semantic = SEMANTIC_TO_TYPE.get(str(get_semantic(metadata, col) or "").lower())

        best = None                                     # (confidence, type, signals)
        for pii_type, spec in PII_TYPES.items():
            signals = []
            if _name_hint(col, spec):
                signals.append("column name")
            if has_rows and spec["regex"]:
                share = values.map(spec["regex"]).mean()
                if share >= regex_min:
                    signals.append(f"regex ({share:.0%} of values match)")
            if spec["presidio"] and presidio.get(spec["presidio"], 0) >= presidio_min:
                signals.append(f"presidio ({presidio[spec['presidio']]:.0%} of sampled values)")
            if semantic == pii_type:
                signals.append("semantic label")
            if not signals:
                continue
            if pii_type in hint_required and "column name" not in signals:
                continue                                # e.g. place names look like people

            key = {"column name": "name", "semantic label": "semantic"}
            miss = 1.0
            for s in signals:
                miss *= 1 - weights[key.get(s, s.split(" ")[0])]
            conf = 1 - miss
            if conf >= min_conf and (best is None or conf > best[0]):
                best = (conf, pii_type, signals)

        if best:
            conf, pii_type, signals = best
            flags.append(make_flag(
                col, RULE_NAME, severity, conf,
                f"likely {pii_type}; signals: {', '.join(signals)}",
                "mask, hash or drop this column before sharing or modelling"))
    return flags