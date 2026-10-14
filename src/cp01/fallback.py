"""Explicit opt-in paid fallback. No keys or raw responses enter logs or graph state."""

import json

from .providers import OllamaProvider, RoleProvider
from .transport import RetryingTransport, TerminalProviderFailure, TransientExhausted


class PaidAPIProvider(RoleProvider):
    def __init__(self, provider: str, api_key: str, model: str, transport: RetryingTransport):
        if provider not in {"openai", "anthropic"} or not api_key.strip() or not model.strip():
            raise ValueError(
                "The selected fallback requires its private API key and model setting."
            )
        self.provider, self._api_key, self.model, self.transport = (
            provider,
            api_key,
            model,
            transport,
        )
        self.name = "OpenAI API fallback" if provider == "openai" else "Claude API fallback"

    def _chat(self, system: str, data: dict) -> dict:
        if self.provider == "openai":
            result = self.transport.post(
                self.name,
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                payload={
                    "model": self.model,
                    "temperature": 0,
                    "max_completion_tokens": 1600,
                    "response_format": {"type": "json_object"},
                    "store": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(data)},
                    ],
                },
            )
            try:
                output = json.loads(result["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError, ValueError):
                raise TerminalProviderFailure(
                    "OpenAI returned an invalid structured response. No retry or further fallback was used."
                ) from None
        else:
            # A forced formatting tool returns structured data; it executes no external action.
            if "sources" in data:
                schema = {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "maxLength": 1200},
                        "claims": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    k: {"type": "string"} for k in ("text", "citation_id", "quote")
                                },
                                "required": ["text", "citation_id", "quote"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["answer", "claims"],
                }
            elif "excerpts" in data:
                schema = {
                    "type": "object",
                    "properties": {"approved": {"type": "boolean"}},
                    "required": ["approved"],
                }
            else:
                schema = {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {"type": "string", "maxLength": 300},
                        }
                    },
                    "required": ["queries"],
                }
            schema["additionalProperties"] = False
            result = self.transport.post(
                self.name,
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                payload={
                    "model": self.model,
                    "max_tokens": 1600,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": json.dumps(data)}],
                    "tools": [
                        {
                            "name": "submit_result",
                            "description": "Return structured role output only; no external action is performed.",
                            "input_schema": schema,
                        }
                    ],
                    "tool_choice": {"type": "tool", "name": "submit_result"},
                },
            )
            blocks = result.get("content", [])
            outputs = (
                [
                    b.get("input")
                    for b in blocks
                    if isinstance(b, dict)
                    and b.get("type") == "tool_use"
                    and b.get("name") == "submit_result"
                ]
                if isinstance(blocks, list)
                else []
            )
            if len(outputs) != 1:
                raise TerminalProviderFailure(
                    "Claude returned an invalid structured response. No retry or further fallback was used."
                )
            output = outputs[0]
        if not isinstance(output, dict):
            raise TerminalProviderFailure(
                "Fallback returned invalid structured output. No retry or further fallback was used."
            )
        return output


class ResilientProvider(RoleProvider):
    """Reuse role validation, switch transport only after transient primary exhaustion."""

    def __init__(self, primary: OllamaProvider, fallback: PaidAPIProvider | None = None):
        if fallback is not None and fallback.transport is not primary.transport:
            raise ValueError("Primary and fallback must share one request budget.")
        self.primary, self.fallback, self.active = primary, fallback, primary
        self.transport = primary.transport
        self.name = "Hosted LLM API"

    def start_question(self):
        self.active = self.primary
        self.transport.reset_question()

    @property
    def active_name(self):
        return self.active.name

    def _chat(self, system: str, data: dict) -> dict:
        try:
            return self.active._chat(system, data)
        except TransientExhausted:
            if self.active is not self.primary or self.fallback is None:
                raise
            self.transport.events.append(
                {"provider": self.fallback.name, "outcome": "opt_in_fallback_activated"}
            )
            self.active = self.fallback
            return self.active._chat(system, data)
