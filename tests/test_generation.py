"""Tests for typhon.eval.generation (generative reader + LLM-as-judge).

Pure builders/parser + artifact field extraction need no model. ``decorate_artifact`` and the env
gate are tested with a fake ChatModel, so nothing here hits the network.
"""

from __future__ import annotations

from typhon.eval.generation import (
    build_judge_messages,
    build_reader_messages,
    decorate_artifact,
    make_chat_from_env,
    parse_judge_verdict,
    references_of,
    retrieved_texts,
)


class FakeChat:
    """A scripted ChatModel: reader prompts -> a fixed answer, judge prompts -> a fixed verdict."""

    def __init__(self, *, reader_answer="Lisbon", judge_raw='{"correct": true}'):
        self.model = "fake-model"
        self.reader_answer = reader_answer
        self.judge_raw = judge_raw
        self.calls: list[str] = []

    def complete(self, messages):
        system = messages[0]["content"]
        self.calls.append(system)
        return self.judge_raw if "impartial grader" in system else self.reader_answer


# --- pure prompt builders -----------------------------------------------------------

def test_reader_messages_include_question_and_facts():
    msgs = build_reader_messages("Where did Dana move?", ["Dana moved to Lisbon", "Pixel phone"])
    assert msgs[0]["role"] == "system"
    assert "Where did Dana move?" in msgs[1]["content"]
    assert "Dana moved to Lisbon" in msgs[1]["content"]


def test_reader_messages_handle_no_facts():
    msgs = build_reader_messages("Q?", [])
    assert "no facts" in msgs[1]["content"].lower()


def test_judge_messages_include_refs_and_prediction():
    msgs = build_judge_messages("Q?", "my answer", ["gold1", "gold2"])
    body = msgs[1]["content"]
    assert "my answer" in body and "gold1" in body and "gold2" in body


# --- verdict parsing (robust + fail-closed) -----------------------------------------

def test_parse_verdict_json_true_false():
    assert parse_judge_verdict('{"correct": true, "reason": "x"}') is True
    assert parse_judge_verdict('{"correct": false}') is False


def test_parse_verdict_json_embedded_in_prose():
    assert parse_judge_verdict('Sure: {"correct": true} done') is True


def test_parse_verdict_keyword_fallback():
    assert parse_judge_verdict("This is CORRECT.") is True
    assert parse_judge_verdict("Verdict: WRONG") is False


def test_parse_verdict_fails_closed_on_garbage():
    assert parse_judge_verdict("") is False
    assert parse_judge_verdict("uh, maybe?") is False


def test_parse_verdict_wrong_beats_stray_correct_word():
    # "incorrect" must not be read as a pass even though it contains "correct".
    assert parse_judge_verdict("That is incorrect") is False


# --- artifact field extraction ------------------------------------------------------

def test_retrieved_texts_reads_any_preview_key():
    assert retrieved_texts({"retrieval_preview": {"cross_episode": ["a", "b"]}}) == ["a", "b"]
    assert retrieved_texts({"retrieval_preview": {"local_exact": ["x"]}}) == ["x"]
    assert retrieved_texts({"retrieval_preview": {}}) == []
    assert retrieved_texts({}) == []


def test_references_of_merges_fixture_and_prediction():
    art = {
        "fixture": {"reference_answers": ["gold"]},
        "prediction": {"reference_answer": "single", "reference_answers": ["gold"]},
    }
    refs = references_of(art)
    assert "gold" in refs and "single" in refs


# --- env gate + decoration ----------------------------------------------------------

def test_make_chat_from_env_none_without_key(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert make_chat_from_env() is None


def test_decorate_artifact_generates_and_judges():
    art = {
        "fixture": {"question": "Where did Dana move?", "reference_answers": ["Lisbon"]},
        "retrieval_preview": {"cross_episode": ["Dana moved to Lisbon in May"]},
        "prediction": {"predicted_answer": "(extractive)", "metrics": {}},
    }
    chat = FakeChat(reader_answer="Lisbon", judge_raw='{"correct": true}')
    decorate_artifact(art, chat, generate=True, judge=True, judge_model="m", reader_model="m")
    gen = art["prediction"]["generative"]
    assert gen["predicted_answer"] == "Lisbon"
    assert gen["answer_mode"] == "generative"
    assert art["prediction"]["llm_judge"]["correct"] is True
    assert art["prediction"]["llm_judge"]["target"] == "generative"
    # reader then judge -> two calls.
    assert len(chat.calls) == 2


def test_decorate_artifact_judge_only_uses_extractive():
    art = {
        "fixture": {"question": "Q?", "reference_answers": ["gold"]},
        "retrieval_preview": {"local_exact": ["some fact"]},
        "prediction": {"predicted_answer": "gold", "metrics": {}},
    }
    chat = FakeChat(judge_raw='{"correct": false}')
    decorate_artifact(art, chat, generate=False, judge=True)
    assert "generative" not in art["prediction"]
    assert art["prediction"]["llm_judge"]["target"] == "extractive"
    assert art["prediction"]["llm_judge"]["correct"] is False
    assert len(chat.calls) == 1  # judge only