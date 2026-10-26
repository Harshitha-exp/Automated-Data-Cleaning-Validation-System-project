"""
Person 3's original tests use flat imports (import engine / import
column_rules as cr / import formats / import pii) exactly as they wrote
them against profiling_rules_to_send/. This conftest puts
src/person3_rules on sys.path so those unmodified test files still work
inside the combined repo, without changing a single line of their code.
"""
import os
import sys

PERSON3_PKG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "person3_rules",
)
if PERSON3_PKG not in sys.path:
    sys.path.insert(0, PERSON3_PKG)
