"""Tests for the LoCoMo importer (raw locomo10.json -> TYPHON samples)."""

from __future__ import annotations

from typhon.benchmarks.locomo_importer import build_locomo_samples, conversation_context

RAW = [
    {
        "sample_id": "c1",
        "conversation": {
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
                {"speaker": "B", "dia_id": "D1:2", "text": "yo"},
            ],
            "session_1_date_time": "1pm on 1 Jan, 2023",
            "session_2": [{"speaker": "A", "dia_id": "D2:1", "text": "bye"}],
            "session_2_date_time": "2pm on 2 Jan, 2023",
        },
        "qa": [
            {"question": "q1?", "answer": "a1", "evidence": ["D1:1"], "category": 2},
            {"question": "qnum?", "answer": 3, "evidence": ["D2:1"], "category": 1},
            {"question": "qadv?", "adversarial_answer": "not mentioned", "category": 5, "evidence": []},
        ],
    }
]


def test_context_orders_sessions_with_markers():
    ctx = conversation_context(RAW[0]["conversation"])
    assert ctx.startswith("Session 1: (1pm on 1 Jan, 2023) A: hi B: yo")
    assert "Session 2: (2pm on 2 Jan, 2023) A: bye" in ctx
    assert ctx.index("Session 1:") < ctx.index("Session 2:")


def test_build_samples_maps_qa_and_coerces_answers():
    samples = build_locomo_samples(RAW)
    assert len(samples) == 3
    s0 = samples[0]
    assert s0["sample_id"] == "c1_q000"
    assert s0["question"] == "q1?"
    assert s0["reference_answer"] == "a1"
    assert s0["metadata"]["category"] == 2
    assert samples[1]["reference_answer"] == "3"  # int coerced to str


def test_adversarial_uses_adversarial_answer_and_flag():
    adv = build_locomo_samples(RAW)[2]
    assert adv["reference_answer"] == "not mentioned"
    assert adv["metadata"]["adversarial"] is True


def test_exclude_adversarial():
    samples = build_locomo_samples(RAW, include_adversarial=False)
    assert len(samples) == 2
    assert all(not s["metadata"]["adversarial"] for s in samples)


def test_max_per_conversation_caps():
    assert len(build_locomo_samples(RAW, max_per_conversation=1)) == 1


def test_numeric_session_sort_not_lexical():
    conv = {
        "session_10": [{"speaker": "A", "text": "ten"}],
        "session_2": [{"speaker": "A", "text": "two"}],
        "session_10_date_time": "",
        "session_2_date_time": "",
    }
    ctx = conversation_context(conv)
    assert ctx.index("Session 2:") < ctx.index("Session 10:")
