"""Tests for the model comparison. These use fake answers and do not call the network."""

import pytest

from arena import (
    BAD_KEY,
    DEFAULT_PROMPT,
    GEMINI_MODEL,
    GROK_MODEL,
    LABEL_GEMINI,
    LABEL_GROK,
    MISSING_KEY,
    MODEL_ERROR,
    RATE_LIMITED_OR_NO_CREDIT,
    UNREACHABLE,
    ModelResult,
    ask_gemini,
    ask_grok,
    compare,
    failure_for_status,
    main,
    prompt_from_args,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail if a test constructs a real model client."""

    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not use the network")

    monkeypatch.setattr("arena.OpenAI", refuse)
    monkeypatch.setattr("arena.genai.Client", refuse)


def _result(label, model, answer, seconds, input_tokens=None, output_tokens=None):
    return ModelResult(
        label=label,
        model=model,
        answer=answer,
        seconds=seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_side_by_side_fake_answers(monkeypatch):
    def grok(prompt):
        assert prompt == "What is a token?"
        return _result(LABEL_GROK, GROK_MODEL, "Grok answer about tokens.", 1.25, 11, 22)

    def gemini(prompt):
        return _result(LABEL_GEMINI, GEMINI_MODEL, "Gemini answer about tokens.", 0.5, 33, 44)

    monkeypatch.setattr("arena.ask_grok", grok)
    monkeypatch.setattr("arena.ask_gemini", gemini)

    text = compare("What is a token?")

    assert "Prompt: What is a token?" in text
    assert "Grok (grok-build-0.1)" in text
    assert "Gemini (gemini-3.5-flash-lite)" in text
    assert text.index("Grok (") < text.index("Gemini (")
    assert "Grok answer about tokens." in text
    assert "Gemini answer about tokens." in text
    assert "Time: 1.25s" in text
    assert "Time: 0.50s" in text
    assert "Input tokens: 11" in text
    assert "Output tokens: 22" in text
    assert "Input tokens: 33" in text
    assert "Output tokens: 44" in text


def test_one_model_failure_still_shows_the_other(monkeypatch):
    def grok(prompt):
        raise MISSING_KEY("XAI_API_KEY", label=LABEL_GROK, model=GROK_MODEL)

    def gemini(prompt):
        return _result(LABEL_GEMINI, GEMINI_MODEL, "Gemini still answered.", 0.5, 3, 4)

    monkeypatch.setattr("arena.ask_grok", grok)
    monkeypatch.setattr("arena.ask_gemini", gemini)

    text = compare("hello")

    assert "MISSING_KEY:" in text
    assert "XAI_API_KEY" in text
    assert "Gemini still answered." in text
    assert "Time: 0.50s" in text
    assert "Input tokens: 3" in text
    assert "Output tokens: 4" in text


def test_both_failures_are_printed(monkeypatch):
    def grok(prompt):
        raise BAD_KEY("XAI_API_KEY", label=LABEL_GROK, model=GROK_MODEL)

    def gemini(prompt):
        raise RATE_LIMITED_OR_NO_CREDIT(label=LABEL_GEMINI, model=GEMINI_MODEL)

    monkeypatch.setattr("arena.ask_grok", grok)
    monkeypatch.setattr("arena.ask_gemini", gemini)

    text = compare("hello")

    assert "BAD_KEY:" in text
    assert "XAI_API_KEY" in text
    assert "RATE_LIMITED_OR_NO_CREDIT:" in text
    assert "rate limited" in text


def test_omits_token_counts_when_the_api_does_not_return_them(monkeypatch):
    def grok(prompt):
        return _result(LABEL_GROK, GROK_MODEL, "Grok with no usage.", 1.0)

    def gemini(prompt):
        return _result(LABEL_GEMINI, GEMINI_MODEL, "Gemini with no usage.", 1.0)

    monkeypatch.setattr("arena.ask_grok", grok)
    monkeypatch.setattr("arena.ask_gemini", gemini)

    text = compare("hello")

    assert "Grok with no usage." in text
    assert "Gemini with no usage." in text
    assert "Time: 1.00s" in text
    assert "Input tokens:" not in text
    assert "Output tokens:" not in text


def test_does_not_print_api_keys(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-test-secret-value")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-secret-value")

    def grok(prompt):
        return _result(LABEL_GROK, GROK_MODEL, "leak sk-test-secret-value", 1.0, 1, 1)

    def gemini(prompt):
        return _result(LABEL_GEMINI, GEMINI_MODEL, "leak google-test-secret-value", 1.0, 1, 1)

    monkeypatch.setattr("arena.ask_grok", grok)
    monkeypatch.setattr("arena.ask_gemini", gemini)

    text = compare("hello")

    assert "sk-test-secret-value" not in text
    assert "google-test-secret-value" not in text
    assert "[redacted]" in text


def test_missing_key_names_the_variable_and_does_not_call_out(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(MISSING_KEY) as grok_failure:
        ask_grok("hello")
    with pytest.raises(MISSING_KEY) as gemini_failure:
        ask_gemini("hello")

    assert "XAI_API_KEY" in str(grok_failure.value)
    assert "GOOGLE_API_KEY" in str(gemini_failure.value)


def test_status_failures():
    bad_key = failure_for_status(LABEL_GROK, GROK_MODEL, "XAI_API_KEY", 401, "nope")
    rate_limited = failure_for_status(LABEL_GEMINI, GEMINI_MODEL, "GOOGLE_API_KEY", 429, "quota")
    model_error = failure_for_status(
        LABEL_GROK, GROK_MODEL, "XAI_API_KEY", 400, "model grok-x is missing"
    )
    bad_gemini_key = failure_for_status(
        LABEL_GEMINI,
        GEMINI_MODEL,
        "GOOGLE_API_KEY",
        400,
        "API key not valid",
        status="INVALID_ARGUMENT",
    )

    assert isinstance(bad_key, BAD_KEY)
    assert "XAI_API_KEY" in str(bad_key)
    assert isinstance(rate_limited, RATE_LIMITED_OR_NO_CREDIT)
    assert isinstance(model_error, MODEL_ERROR)
    assert "model grok-x is missing" in str(model_error)
    assert isinstance(bad_gemini_key, BAD_KEY)
    assert "GOOGLE_API_KEY" in str(bad_gemini_key)
    assert "API key not valid" not in str(bad_gemini_key)


def test_model_error_redacts_the_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-secret-value")
    failure = failure_for_status(
        LABEL_GEMINI,
        GEMINI_MODEL,
        "GOOGLE_API_KEY",
        400,
        "rejected google-test-secret-value",
    )
    assert isinstance(failure, MODEL_ERROR)
    assert "google-test-secret-value" not in str(failure)
    assert "[redacted]" in str(failure)


def test_unreachable_message():
    failure = UNREACHABLE(label=LABEL_GEMINI, model=GEMINI_MODEL)
    message = str(failure)
    assert "could not be reached" in message
    assert "timed out" in message
    assert "Check the network" in message


def test_default_prompt(capsys, monkeypatch):
    seen = {}

    def fake_compare(prompt):
        seen["prompt"] = prompt
        return "comparison"

    monkeypatch.setattr("arena.compare", fake_compare)
    assert prompt_from_args([]) == DEFAULT_PROMPT
    main([])
    assert seen["prompt"] == DEFAULT_PROMPT
    assert capsys.readouterr().out.strip() == "comparison"


def test_prompt_from_the_command_line():
    assert prompt_from_args(["Explain", "tokens"]) == "Explain tokens"
