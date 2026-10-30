"""Send one prompt to Grok and Gemini and print a side-by-side comparison."""

import argparse
import os
import textwrap
import time
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

DEFAULT_PROMPT = "Explain what a context window is in two sentences for a beginner."
XAI_BASE_URL = "https://api.x.ai/v1"
GROK_MODEL = "grok-build-0.1"
GEMINI_MODEL = "gemini-3.5-flash-lite"
LABEL_GROK = "Grok"
LABEL_GEMINI = "Gemini"
COLUMN_WIDTH = 40
REQUEST_TIMEOUT_SECONDS = 60

# Dollars per 1 million tokens: (input price, output price).
PRICES = {
    GROK_MODEL: (1.00, 2.00),
    GEMINI_MODEL: (0.10, 0.40),
}


class Failure(Exception):
    """A known failure. The comparison prints the class name and this message."""

    def __init__(self, message: str, *, label: str, model: str) -> None:
        super().__init__(message)
        self.label = label
        self.model = model


class MISSING_KEY(Failure):
    def __init__(self, key_name: str, *, label: str, model: str) -> None:
        super().__init__(
            f"{key_name} is not set. Set that environment variable and run the script again.",
            label=label,
            model=model,
        )


class BAD_KEY(Failure):
    def __init__(self, key_name: str, *, label: str, model: str) -> None:
        super().__init__(
            f"{key_name} was rejected. Check that environment variable and try again.",
            label=label,
            model=model,
        )


class RATE_LIMITED_OR_NO_CREDIT(Failure):
    def __init__(self, *, label: str, model: str) -> None:
        super().__init__(
            f"{label} is rate limited or the account is out of credit. "
            "Wait and try again, or check billing.",
            label=label,
            model=model,
        )


class UNREACHABLE(Failure):
    def __init__(self, *, label: str, model: str) -> None:
        super().__init__(
            f"{label} could not be reached because the connection failed or the request timed out. "
            "Check the network and try again.",
            label=label,
            model=model,
        )


class MODEL_ERROR(Failure):
    def __init__(self, detail: str, *, label: str, model: str) -> None:
        super().__init__(
            f"{label} returned an error: {detail}. Try again later.",
            label=label,
            model=model,
        )


@dataclass
class ModelResult:
    label: str
    model: str
    answer: str
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


def _redact(text: str) -> str:
    for name in ("XAI_API_KEY", "GOOGLE_API_KEY"):
        raw = os.environ.get(name) or ""
        for key in {raw, raw.strip()}:
            if key:
                text = text.replace(key, "[redacted]")
    return text


def _public_detail(exc: BaseException) -> str:
    message = _redact(" ".join(str(exc).split()))
    lowered = message.lower()
    if any(word in lowered for word in ("api_key", "authorization", "bearer", "x-goog-api-key")):
        return "the request failed before the model returned an answer"
    return message or "the request failed before the model returned an answer"


def require_key(key_name: str, label: str, model: str) -> str:
    key = (os.environ.get(key_name) or "").strip()
    if not key:
        raise MISSING_KEY(key_name, label=label, model=model)
    return key


def failure_for_status(
    label: str,
    model: str,
    key_name: str,
    status_code: int,
    server_message: str,
    status: str = "",
) -> Failure:
    blob = f"{status} {server_message}".lower()
    if status_code == 429 or "resource_exhausted" in blob or "resource exhausted" in blob:
        return RATE_LIMITED_OR_NO_CREDIT(label=label, model=model)
    if status_code in (401, 403) or _is_bad_key(blob):
        return BAD_KEY(key_name, label=label, model=model)
    detail = _redact(" ".join(server_message.split())) or "the server did not include a message"
    return MODEL_ERROR(detail, label=label, model=model)


def _is_bad_key(blob: str) -> bool:
    if "api_key_invalid" in blob or "unauthenticated" in blob:
        return True
    return "api key" in blob and any(word in blob for word in ("invalid", "not valid", "rejected"))


def _count(usage: object, *names: str) -> int | None:
    if usage is None:
        return None
    for name in names:
        value = getattr(usage, name, None)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        return value
    return None


def _openai_server_message(exc: APIStatusError) -> str:
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif isinstance(error, str):
            message = error
    elif isinstance(body, str):
        message = body
    if not message:
        message = str(getattr(exc, "message", "") or "")
    return " ".join(message.split())


def ask_grok(prompt: str) -> ModelResult:
    key = require_key("XAI_API_KEY", LABEL_GROK, GROK_MODEL)
    client = OpenAI(
        api_key=key,
        base_url=XAI_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIStatusError as exc:
        raise failure_for_status(
            LABEL_GROK,
            GROK_MODEL,
            "XAI_API_KEY",
            exc.status_code,
            _openai_server_message(exc),
        ) from None
    except (APITimeoutError, APIConnectionError):
        raise UNREACHABLE(label=LABEL_GROK, model=GROK_MODEL) from None
    seconds = time.perf_counter() - started
    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return ModelResult(
        label=LABEL_GROK,
        model=GROK_MODEL,
        answer=answer,
        seconds=seconds,
        input_tokens=_count(usage, "prompt_tokens", "input_tokens"),
        output_tokens=_count(usage, "completion_tokens", "output_tokens"),
    )


def ask_gemini(prompt: str) -> ModelResult:
    key = require_key("GOOGLE_API_KEY", LABEL_GEMINI, GEMINI_MODEL)
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_SECONDS * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    started = time.perf_counter()
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except genai_errors.APIError as exc:
        raise failure_for_status(
            LABEL_GEMINI,
            GEMINI_MODEL,
            "GOOGLE_API_KEY",
            int(exc.code or 0),
            str(exc.message or ""),
            str(exc.status or ""),
        ) from None
    except (httpx.TimeoutException, httpx.NetworkError):
        raise UNREACHABLE(label=LABEL_GEMINI, model=GEMINI_MODEL) from None
    seconds = time.perf_counter() - started
    usage = getattr(response, "usage_metadata", None)
    return ModelResult(
        label=LABEL_GEMINI,
        model=GEMINI_MODEL,
        answer=response.text or "",
        seconds=seconds,
        input_tokens=_count(usage, "prompt_token_count"),
        output_tokens=_count(usage, "candidates_token_count"),
    )


def _attempt(ask, prompt: str) -> ModelResult | Failure:
    try:
        return ask(prompt)
    except Failure as failure:
        return failure
    except Exception as exc:
        label = LABEL_GROK if ask is ask_grok else LABEL_GEMINI
        model = GROK_MODEL if ask is ask_grok else GEMINI_MODEL
        return MODEL_ERROR(_public_detail(exc), label=label, model=model)


def _header(outcome: ModelResult | Failure) -> str:
    return f"{outcome.label} ({outcome.model})"


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def _body_lines(outcome: ModelResult | Failure, width: int) -> list[str]:
    if isinstance(outcome, Failure):
        return _wrap(f"{type(outcome).__name__}: {outcome}", width)
    return _wrap(outcome.answer.strip() or "(no text returned)", width)


def estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Estimated dollars for one call, or None if the price or token counts are unknown."""
    if model not in PRICES or input_tokens is None or output_tokens is None:
        return None
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _footer(outcome: ModelResult | Failure) -> list[str]:
    if isinstance(outcome, Failure):
        return []
    lines = [f"Time: {outcome.seconds:.2f}s"]
    if outcome.input_tokens is not None:
        lines.append(f"Input tokens: {outcome.input_tokens}")
    if outcome.output_tokens is not None:
        lines.append(f"Output tokens: {outcome.output_tokens}")
    cost = estimate_cost(outcome.model, outcome.input_tokens, outcome.output_tokens)
    if cost is not None:
        lines.append(f"Cost: ${cost:.4f}")
    return lines


def render(prompt: str, left: ModelResult | Failure, right: ModelResult | Failure) -> str:
    width = max(COLUMN_WIDTH, len(_header(left)), len(_header(right)))
    left_body = _body_lines(left, width)
    right_body = _body_lines(right, width)
    body_height = max(len(left_body), len(right_body))
    left_body += [""] * (body_height - len(left_body))
    right_body += [""] * (body_height - len(right_body))
    left_lines = [_header(left), "-" * width, *left_body, "", *_footer(left)]
    right_lines = [_header(right), "-" * width, *right_body, "", *_footer(right)]
    rows = max(len(left_lines), len(right_lines))
    left_lines += [""] * (rows - len(left_lines))
    right_lines += [""] * (rows - len(right_lines))
    table = "\n".join(f"{a:<{width}}    {b}" for a, b in zip(left_lines, right_lines))
    return _redact(f"Prompt: {_redact(prompt)}\n\n{table}")


def compare(prompt: str) -> str:
    return render(prompt, _attempt(ask_grok, prompt), _attempt(ask_gemini, prompt))


def prompt_from_args(argv: list[str] | None) -> str:
    parser = argparse.ArgumentParser(description="Compare Grok and Gemini on the same prompt.")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to both models.")
    args = parser.parse_args(argv)
    text = " ".join(args.prompt).strip()
    return text or DEFAULT_PROMPT


def main(argv: list[str] | None = None) -> None:
    print(compare(prompt_from_args(argv)))


if __name__ == "__main__":
    main()
