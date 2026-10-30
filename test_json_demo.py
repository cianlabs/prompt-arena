"""Offline tests: a fake client stands in for Grok, so no network and no key."""

from types import SimpleNamespace

import pytest

import json_demo

GOOD = '{"name": "Aoife", "city": "Cork", "temperature": 12.5}'


class FakeClient:
    """Returns the scripted replies one by one and remembers the prompts it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages):
        self.prompts.append(messages[0]["content"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_valid_json_needs_one_try():
    client = FakeClient([GOOD])
    data, tries = json_demo.get_json(client, "p")
    assert data == {"name": "Aoife", "city": "Cork", "temperature": 12.5}
    assert tries == 1
    assert len(client.prompts) == 1


def test_json_in_markdown_fences_is_accepted_without_retry():
    client = FakeClient(["```json\n" + GOOD + "\n```"])
    data, tries = json_demo.get_json(client, "p")
    assert data["city"] == "Cork"
    assert tries == 1
    assert len(client.prompts) == 1


def test_invalid_json_triggers_one_stricter_retry():
    client = FakeClient(["Sure! Here you go: name Aoife, city Cork", GOOD])
    data, tries = json_demo.get_json(client, "p")
    assert data["name"] == "Aoife"
    assert tries == 2
    assert len(client.prompts) == 2
    assert "Return only valid JSON" in client.prompts[1]
    assert "Return only valid JSON" not in client.prompts[0]


def test_still_invalid_after_retry_is_named_failure():
    client = FakeClient(["nope", "still nope"])
    with pytest.raises(json_demo.BAD_JSON) as info:
        json_demo.get_json(client, "p")
    assert str(info.value).startswith("BAD_JSON:")
    assert len(client.prompts) == 2


def test_json_that_is_not_an_object_counts_as_invalid():
    client = FakeClient(["[1, 2, 3]", GOOD])
    data, tries = json_demo.get_json(client, "p")
    assert tries == 2
    assert data["city"] == "Cork"


def test_model_request_failure_is_named_failure():
    client = FakeClient([RuntimeError("boom")])
    with pytest.raises(json_demo.MODEL_ERROR):
        json_demo.get_json(client, "p")


def test_strip_fences_leaves_plain_text_alone():
    assert json_demo.strip_fences('  {"a": 1}  ') == '{"a": 1}'
