"""
tests/unit/test_searcher_unit.py
---------------------------------
Unit tests for pure-Python helpers in src/searcher.py.
These tests do NOT require a CLIP model or GPU and run in milliseconds.

Scenarios covered
-----------------
QN-01  "find the <X>"  → "<X>"  (basic imperative stripped)
QN-02  "show me <X>"   → "<X>"
QN-03  "search for <X>" → "<X>"
QN-04  "find me the <X>" → "the <X>"  (regex matches "me" first; iterative)
QN-05  Plain query without prefix → unchanged
QN-06  Prefix-only input (no noun) → never returns empty string
QN-07  Mixed-case prefix is stripped (case-insensitive)
QN-08  Leading/trailing whitespace stripped from result
QN-09  Five CLIP templates produced from a cleaned query
QN-10  Every template in _CLIP_TEMPLATES contains exactly one "{}" placeholder
QN-11  "locate <X>" stripped
QN-12  "can you find <X>" stripped
"""

from __future__ import annotations

import pytest

from src.searcher import _normalize_query, _CLIP_TEMPLATES

pytestmark = pytest.mark.unit


# ── QN-01  "find the" ────────────────────────────────────────────────────

def test_QN01_find_the_stripped():
    assert _normalize_query("find the man waving hand") == "man waving hand"


# ── QN-02  "show me" ─────────────────────────────────────────────────────

def test_QN02_show_me_stripped():
    assert _normalize_query("show me a red car") == "a red car"


# ── QN-03  "search for" ───────────────────────────────────────────────────

def test_QN03_search_for_stripped():
    assert _normalize_query("search for running person") == "running person"


# ── QN-04  "find me the" — iterative stripping ───────────────────────────

def test_QN04_find_me_the_strips_find_me():
    # The regex matches "me" (not "me the") from the alternation.
    # First pass: "find me " removed → "the bicycle"
    # Second pass: "the bicycle" has no prefix → returned as-is
    result = _normalize_query("find me the bicycle")
    assert "find" not in result.lower()
    assert "bicycle" in result


# ── QN-05  Plain query unchanged ─────────────────────────────────────────

def test_QN05_plain_query_unchanged():
    q = "person climbing fence"
    assert _normalize_query(q) == q


# ── QN-06  Prefix-only input → never empty ───────────────────────────────

def test_QN06_prefix_only_not_empty():
    # "find the" with nothing after — must not produce ""
    result = _normalize_query("find the")
    assert isinstance(result, str)
    assert len(result) > 0


# ── QN-07  Case-insensitive prefix ───────────────────────────────────────

def test_QN07_case_insensitive():
    assert _normalize_query("FIND THE bicycle") == "bicycle"
    assert _normalize_query("Show Me a dog") == "a dog"


# ── QN-08  Whitespace trimmed from result ────────────────────────────────

def test_QN08_result_has_no_leading_trailing_whitespace():
    result = _normalize_query("  find  the   dog  ")
    assert result == result.strip()


# ── QN-09  Five CLIP templates produced ──────────────────────────────────

def test_QN09_five_templates_produced():
    cleaned = "person on roof"
    templates = [t.format(cleaned) for t in _CLIP_TEMPLATES]
    assert len(templates) == 5
    assert cleaned in templates        # "{}" identity template must be present


# ── QN-10  Each template has exactly one "{}" placeholder ────────────────

def test_QN10_each_template_has_one_placeholder():
    for t in _CLIP_TEMPLATES:
        assert t.count("{}") == 1, (
            f"Template should have exactly 1 placeholder, got: {t!r}"
        )


# ── QN-11  "locate" stripped ─────────────────────────────────────────────

def test_QN11_locate_stripped():
    assert _normalize_query("locate the red bus") == "the red bus"


# ── QN-12  "can you find" stripped ───────────────────────────────────────

def test_QN12_can_you_find_stripped():
    result = _normalize_query("can you find the cat on the roof")
    assert "can you" not in result.lower()
    assert "cat" in result
