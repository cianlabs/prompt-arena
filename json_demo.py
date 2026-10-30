"""Ask Grok for a JSON object, check it really is JSON, and retry once if it isn't."""

import json
import os
import sys

from openai import OpenAI

XAI_BASE_URL = "https://api.x.ai/v1"
MODEL = "grok-build-0.1"

PROMPT = (
    "Give me a JSON object describing a made-up traveler with exactly these fields: "
    "name (string), city (string), temperature (number, degrees Celsius)."
)
STRICT_SUFFIX = (
    "\n\nReturn only valid JSON with no markdown, no code fences and no explanation. "
    "Your whole reply must be one JSON object."
)


class Failure(Exception):
    """Base class: every failure has a name and a plain-English message."""

    name = "FAILURE"

    def __str__(self) -> str:
        return f"{self.name}: {self.args[0]}"


class MISSING_KEY(Failure):
    name = "MISSING_KEY"


class MODEL_ERROR(Failure):
    name = "MODEL_ERROR"


class BAD_JSON(Failure):
    name = "BAD_JSON"


def strip_fences(text: str) -> str:
    """Remove ```json ... ``` wrapping if the model added it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop the opening fence line, e.g. ```json
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_json_object(text: str) -> dict:
    """Return the JSON object in text, or raise ValueError."""
    data = json.loads(strip_fences(text))
    if not isinstance(data, dict):
        raise ValueError("the JSON is not an object")
    return data


def ask(client, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # the real client raises several kinds of errors
        raise MODEL_ERROR(f"the model request failed ({type(exc).__name__}).") from None
    return response.choices[0].message.content or ""


def get_json(client, prompt: str) -> tuple[dict, int]:
    """Return (data, number_of_tries). Tries at most twice."""
    first = ask(client, prompt)
    try:
        return parse_json_object(first), 1
    except ValueError:
        pass
    second = ask(client, prompt + STRICT_SUFFIX)
    try:
        return parse_json_object(second), 2
    except ValueError:
        raise BAD_JSON("the model did not return valid JSON, even after one retry.") from None


def main() -> int:
    key = (os.environ.get("XAI_API_KEY") or "").strip()
    if not key:
        print(MISSING_KEY("XAI_API_KEY is not set. Set it and run again."))
        return 1
    client = OpenAI(api_key=key, base_url=XAI_BASE_URL, max_retries=0, timeout=60)
    try:
        data, tries = get_json(client, PROMPT)
    except Failure as failure:
        print(failure)
        return 1
    print(json.dumps(data, indent=2))
    print(f"(valid JSON after {tries} {'try' if tries == 1 else 'tries'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
