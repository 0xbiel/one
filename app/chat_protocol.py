"""Bounded OpenAI-compatible chat protocol; never expose analysis channels."""
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ID = r"^[A-Za-z0-9_-]{1,64}$"
MAX_TOOL_BYTES = 8192
MAX_OUTPUT_BYTES = 32768


class ChatFailure(Exception):
    def __init__(self, code: str, retryable: bool = False):
        self.code, self.retryable = code, retryable
        super().__init__(code)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QuestionOption(StrictModel):
    id: str = Field(pattern=ID)
    label: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=240)


class UserQuestion(StrictModel):
    id: str = Field(pattern=ID)
    header: str = Field(min_length=1, max_length=12)
    question: str = Field(min_length=1, max_length=240)
    category: Literal["preference", "schedule", "organization"] = "preference"
    options: list[QuestionOption] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def unique_options(self):
        if len({option.id for option in self.options}) != len(self.options):
            raise ValueError("Duplicate option IDs")
        return self


# Defense in depth, not a universal semantic classifier. This UI tool is
# restricted to routine organizational preferences; context consent does not
# authorize eliciting health details, secrets, identity, or protected traits.
SENSITIVE = re.compile(
    r"\b(password|passcode|api[ _-]?key|secret|social security|ssn|credit card|bank account|"
    r"diagnos\w*|symptom\w*|medical history|medication|medicine|dose|dosage|"
    r"religio\w*|ethnic\w*|race|racial|sexual\w*|political|immigration|"
    r"disabilit\w*|mental health|suicid\w*|self[ -]?harm|pregnan\w*|"
    r"contrase\w*|diagnóst\w*|síntoma\w*|medicamento\w*|dosis|"
    r"religión|etnia|orientación sexual|salud mental)\b", re.IGNORECASE,
)


class QuestionBatch(StrictModel):
    questions: list[UserQuestion] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def safe_batch(self):
        if len({question.id for question in self.questions}) != len(self.questions):
            raise ValueError("Duplicate question IDs")
        if SENSITIVE.search(json.dumps(self.model_dump(), ensure_ascii=False)):
            raise ValueError("Sensitive questions are unavailable")
        return self


ASK_USER_QUESTIONS = {
    "type": "function",
    "function": {
        "name": "ask_user_questions",
        "description": "Ask one batch of 1-3 routine organization, schedule, or preference questions, only when needed to fulfill the user's request. Never ask for sensitive personal details, health details, medication choices, financial information, credentials, or protected traits. Each question has 2-3 choices; the UI also offers custom text. Wait for the whole batch before continuing.",
        # Providers differ in strict-schema restrictions. Enforcement is always
        # server-side; no provider-specific strict or reasoning flags here.
        "parameters": QuestionBatch.model_json_schema(),
    },
}


class ChatAccumulator:
    """Assemble one tool call and require a truthful terminal finish reason."""
    def __init__(self, *, allow_tools: bool = True):
        self.allow_tools = allow_tools
        self.text = ""
        self.call = None
        self.finished = False
        self.reason = None

    def push(self, chunk: dict) -> str | None:
        if not isinstance(chunk, dict) or chunk.get("error"):
            raise ChatFailure("invalid_provider_stream")
        choices = chunk.get("choices", [])
        if not choices:  # Usage-only chunks may occur after the terminal choice.
            return None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or choices[0].get("index", 0) != 0 or self.finished:
            raise ChatFailure("invalid_provider_stream")
        choice = choices[0]
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise ChatFailure("invalid_provider_stream")
        # Ignore reasoning, analysis, and log probabilities entirely. Raw
        # Harmony serialization is not an API response or safe UI content.
        content = delta.get("content")
        visible = delta.get("channel") in (None, "final", "commentary")
        if content is not None and not isinstance(content, str):
            raise ChatFailure("invalid_provider_stream")
        if content and visible:
            if self.call is not None or "<|" in self.text + content:
                raise ChatFailure("invalid_provider_stream")
            self.text += content
            if len(self.text.encode("utf-8")) > MAX_OUTPUT_BYTES:
                raise ChatFailure("output_budget_exceeded")
        fragments = delta.get("tool_calls") or []
        if not isinstance(fragments, list) or (fragments and (not visible or not self.allow_tools)):
            raise ChatFailure("invalid_tool_call")
        for fragment in fragments:
            if not isinstance(fragment, dict) or fragment.get("index") != 0 or fragment.get("type", "function") != "function":
                raise ChatFailure("invalid_tool_call")
            if self.call is None:
                self.call = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            function = fragment.get("function", {})
            if not isinstance(function, dict) or set(function) - {"name", "arguments"}:
                raise ChatFailure("invalid_tool_call")
            for target, value in [(self.call, {"id": fragment.get("id")}), (self.call["function"], function)]:
                for key in ("id", "name", "arguments"):
                    part = value.get(key)
                    if part is not None:
                        if not isinstance(part, str):
                            raise ChatFailure("invalid_tool_call")
                        target[key] = target.get(key, "") + part
            if len(json.dumps(self.call).encode("utf-8")) > MAX_TOOL_BYTES:
                raise ChatFailure("tool_budget_exceeded")
        reason = choice.get("finish_reason")
        if reason is not None:
            self.finished, self.reason = True, reason
        return content if content and visible else None

    def complete(self) -> tuple[dict, dict | None]:
        if not self.finished:
            raise ChatFailure("incomplete_provider_stream", True)
        if self.call is None:
            if self.reason != "stop" or not self.text.strip():
                raise ChatFailure("incomplete_provider_stream", self.reason == "length")
            return {"role": "assistant", "content": self.text}, None
        if self.reason != "tool_calls" or not re.fullmatch(ID, self.call["id"]) or self.call["function"]["name"] != "ask_user_questions":
            raise ChatFailure("invalid_tool_call")
        try:
            batch = QuestionBatch.model_validate_json(self.call["function"]["arguments"])
        except (ValidationError, ValueError):
            raise ChatFailure("invalid_question_batch")
        return {"role": "assistant", "content": self.text or None, "tool_calls": [self.call]}, {"tool_call_id": self.call["id"], **batch.model_dump()}


def provider_chat_support(settings, metadata: dict) -> dict:
    """Documented API support, not a probe or model availability guarantee.

    Checked 2026-10-02: https://inference-docs.cerebras.ai/capabilities/tool-use
    and /api-reference/chat-completions; https://lmstudio.ai/docs/developer/
    openai-compat/tools and /chat-completions. LM Studio's selected template
    may use its default parser; only explicit documented examples are native.
    Unknown compatible gateways/models fail closed rather than guessing.
    """
    provider = metadata.get("provider")
    model = settings.effective_llm_model.strip().lower()
    support = {"streaming": False, "tools_supported": False, "tool_support": "unverified"}
    if not model:
        return support
    if provider in {"lmstudio", "lm_studio"}:
        native_examples = ("qwen2.5-7b-instruct", "llama-3.1-8b-instruct", "ministral-8b-instruct-2410")
        support.update(streaming=True, tools_supported=True,
                       tool_support="native" if any(name in model for name in native_examples) else "default")
    elif provider == "cerebras" and model in {"gpt-oss-120b", "qwen-3.8-27b", "kimi-k2.7-code"}:
        support.update(streaming=True, tools_supported=True, tool_support="native")
    elif provider in {"openai_compatible", "openrouter"}:
        support.update(streaming=True, tool_support="unverified")
    return support
