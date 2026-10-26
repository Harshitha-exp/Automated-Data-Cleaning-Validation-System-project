"""
Root test configuration for Module 1.

Puts `src/` on sys.path so every test file can import the three person
packages (person1_metadata, person2_profiling, person3_rules) plus the
orchestrator and api modules the same way the shipped code does.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
