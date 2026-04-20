from __future__ import annotations

import json
import os
from typing import Iterator
from urllib import error, parse, request


def _build_prompt(persona_header: str, strategy: str, memory_context: str, user_input: str) -> str:
    return (
        f"{persona_header}\n"
        f"Memory Strategy: {strategy}\n"
        "Use the memory context to stay consistent with prior conversation details. "
        "Keep replies practical and concise unless the user asks for deep detail. "
        "Match the user's mood and tone naturally (friendly, formal, excited, concerned) while staying helpful. "
        "Do not use meta labels or scaffolding like 'Assumptions:', 'Thinking:', or 'Analysis:' unless the user explicitly asks for them. "
        "Start with a direct, natural answer. "
        "Strictly follow any explicit output constraints from the user (format, count, length, style). "
        "If the user asks for N points, return exactly N points.\n\n"
        f"Memory Context:\n{memory_context[:5000]}\n\n"
        f"User Message:\n{user_input}"
    )


def _normalize_model_name(model: str) -> str:
    name = (model or "").strip()
    if name.startswith("models/"):
        return name.split("models/", 1)[1]
    return name


def _candidate_models() -> list[str]:
    configured = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
    configured_list = [m.strip() for m in configured.split(",") if m.strip()]
    fallbacks = [
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-lite-latest",
    ]
    candidates: list[str] = []
    for model in configured_list + fallbacks:
        normalized = _normalize_model_name(model)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _extract_text_from_response(data: dict) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()


def stream_gemini_reply(
    *,
    persona_header: str,
    strategy: str,
    memory_context: str,
    user_input: str,
    temperature: float = 0.4,
) -> Iterator[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return

    models = _candidate_models()
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip()

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_prompt(persona_header, strategy, memory_context, user_input),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }
    body = json.dumps(payload).encode("utf-8")

    for model in models:
        endpoint = (
            f"{base_url.rstrip('/')}/models/{parse.quote(model, safe='')}:streamGenerateContent"
            f"?alt=sse&key={parse.quote(api_key, safe='')}"
        )
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=60) as response:
                cumulative_text = ""
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue

                    try:
                        event = json.loads(data_str)
                    except ValueError:
                        continue

                    event_text = _extract_text_from_response(event)
                    if not event_text:
                        continue

                    if event_text.startswith(cumulative_text):
                        delta = event_text[len(cumulative_text):]
                        cumulative_text = event_text
                    else:
                        delta = event_text
                        cumulative_text += event_text

                    if delta:
                        yield delta

                if cumulative_text:
                    return
        except error.HTTPError as http_err:
            if http_err.code in (400, 404, 429, 500, 503):
                continue
            return
        except (error.URLError, TimeoutError, OSError):
            return


def generate_gemini_reply(
    *,
    persona_header: str,
    strategy: str,
    memory_context: str,
    user_input: str,
    temperature: float = 0.4,
) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    models = _candidate_models()
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip()

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_prompt(persona_header, strategy, memory_context, user_input),
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }
    body = json.dumps(payload).encode("utf-8")

    for model in models:
        endpoint = (
            f"{base_url.rstrip('/')}/models/{parse.quote(model, safe='')}:generateContent?key={parse.quote(api_key, safe='')}"
        )
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as http_err:
            if http_err.code in (400, 404, 429, 500, 503):
                # Try another model alias if one is unsupported, quota-limited, or temporarily unavailable.
                continue
            return None
        except (error.URLError, TimeoutError, OSError):
            return None

        try:
            data = json.loads(raw)
            text = _extract_text_from_response(data)
            if text:
                return text
        except (ValueError, TypeError, KeyError):
            continue

    return None