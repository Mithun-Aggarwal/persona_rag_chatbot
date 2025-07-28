"""Unit tests for QueryClassifier.

Run with:
    pytest -q tests/planner/test_query_classifier.py
"""
from src.planner.query_classifier import QueryClassifier, QueryMeta, Intent, Complexity


def test_basic_classification_stub(monkeypatch):
    """The stub Gemini client always returns the same JSON → make sure mapping works."""

    qc = QueryClassifier()
    q = "What is the PBAC outcome for belantamab mafodotin?"
    meta: QueryMeta = qc(q)

    assert isinstance(meta, QueryMeta)
    assert meta.intent in {"factual", "comparative", "procedural", "citation", "other"}
    assert meta.complexity in {"simple", "moderate", "complex"}
    assert 0 <= meta.confidence_needed <= 1

    # Stub response in query_classifier.GeminiClient.call_gemini → 'factual', 'simple', 0.75
    assert meta.intent == "factual"
    assert meta.complexity == "simple"
    assert meta.confidence_needed == 0.75


def test_cache(monkeypatch):
    """Ensure LRU cache hits (no second Gemini call)."""

    calls = []

    def fake_call(payload):
        calls.append(payload)
        return {"intent": "other", "complexity": "moderate", "confidence": 0.5}

    monkeypatch.setattr("src.planner.query_classifier.GeminiClient.call_gemini", fake_call)

    qc = QueryClassifier(cache_size=2)
    q = "Explain indirect costs in PBAC submissions"
    first = qc(q)
    second = qc(q)  # cached

    assert first == second
    # Gemini called exactly once
    assert len(calls) == 1
