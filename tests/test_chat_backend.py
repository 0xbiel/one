"""Mock-only normal chat: disposable DB, no provider requests or real keys."""
import io
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.chat_protocol import ChatAccumulator, ChatFailure, QuestionBatch
from app.config import Settings
from app.integrations import LMStudioAdapter, MAX_LLM_REQUEST_BYTES
from app.main import make_app


def uid():
    return str(uuid.uuid4())


def batch():
    return {"questions": [{"id": "format", "header": "Format", "question": "Which format would help?", "options": [
        {"id": "brief", "label": "Brief", "description": "A concise summary."},
        {"id": "list", "label": "List", "description": "A checklist."}]}]}


def chunk(delta=None, finish=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def tool_chunks(arguments=None, call_id="call_one"):
    text = json.dumps(arguments or batch())
    yield chunk({"reasoning_content": "PRIVATE SHOULD NEVER APPEAR"})
    yield chunk({"tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": "ask_user_", "arguments": text[:10]}}]})
    yield chunk({"tool_calls": [{"index": 0, "function": {"name": "questions", "arguments": text[10:]}}]}, "tool_calls")


def parse(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def client(tmp_path, **kwargs):
    settings = Settings(_env_file=None, database_url="sqlite:///:memory:", object_store_path=tmp_path / "objects", bootstrap_secret="test", env="test", **kwargs)
    app = make_app(settings)
    c = TestClient(app)
    started = c.post("/api/v1/pairing/start", json={"display_name": "Synthetic Admin", "home_name": "Disposable Home"}).json()
    session = c.post("/api/v1/pairing/complete", json={"code": started["pairing_code"]}).json()
    home = session["home_id"]
    headers = {"Authorization": "Bearer " + session["access_token"]}
    recipient = c.post(f"/api/v1/homes/{home}/care-recipients", headers=headers, json={"display_name": "Synthetic Recipient"}).json()["data"]["id"]
    assert c.post(f"/api/v1/homes/{home}/consents", headers=headers, json={"purpose": "analytics", "policy_version": "test", "care_recipient_id": recipient}).status_code == 200
    return c, f"/api/v1/homes/{home}/chat", headers, recipient


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.delenv("ONE_PROVIDER_CONFIG_DIR", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Real provider network forbidden")
    monkeypatch.setattr(LMStudioAdapter, "_open_request", forbidden)


def test_fragment_assembly_and_private_reasoning_suppressed():
    accumulator = ChatAccumulator()
    for part in tool_chunks():
        assert accumulator.push(part) is None
    assistant, pending = accumulator.complete()
    assert assistant["tool_calls"][0]["function"]["name"] == "ask_user_questions"
    assert pending["questions"][0]["id"] == "format"
    assert "PRIVATE" not in json.dumps((assistant, pending))


@pytest.mark.parametrize("mutation", ["duplicate", "one_option", "sensitive", "unknown_category", "extra"])
def test_question_boundary_rejects_invalid_or_sensitive_batches(mutation):
    arguments = batch()
    q = arguments["questions"][0]
    if mutation == "duplicate": arguments["questions"].append(q.copy())
    if mutation == "one_option": q["options"] = q["options"][:1]
    if mutation == "sensitive": q["question"] = "What is your diagnosis?"
    if mutation == "unknown_category": q["category"] = "health"
    if mutation == "extra": q["consent_grant"] = True
    accumulator = ChatAccumulator()
    for part in tool_chunks(arguments): accumulator.push(part)
    with pytest.raises(ChatFailure, match="invalid_question_batch"):
        accumulator.complete()


def test_incomplete_and_unbounded_tool_calls_are_not_dispatched():
    accumulator = ChatAccumulator()
    accumulator.push(chunk({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "ask_user_questions", "arguments": "{"}}]}))
    with pytest.raises(ChatFailure, match="incomplete_provider_stream"): accumulator.complete()
    with pytest.raises(ChatFailure, match="tool_budget_exceeded"):
        accumulator.push(chunk({"tool_calls": [{"index": 0, "function": {"arguments": "a" * 9000}}]}))
    with pytest.raises(ChatFailure, match="invalid_tool_call"):
        ChatAccumulator().push(chunk({"tool_calls": [{"index": 1, "function": {}}]}))


def test_unconfigured_debug_batch_custom_answers_and_idempotency(tmp_path):
    c, url, headers, recipient = client(tmp_path)
    assert c.get(url + "/capability", headers=headers, params={"care_recipient_id": recipient}).json()["mode"] == "debug"
    body = {"request_id": uid(), "message": "Help organize this week", "care_recipient_id": recipient}
    events = parse(c.post(url + "/stream", headers=headers, json=body))
    assert events == parse(c.post(url + "/stream", headers=headers, json=body))
    pending = next(event for event in events if event["type"] == "questions.required")
    conversation = pending["conversation_id"]
    answer = {"request_id": uid(), "expected_revision": 1, "tool_call_id": pending["tool_call_id"], "answers": [
        {"question_id": "focus", "option_id": "week"}, {"question_id": "format", "custom_text": "A short calendar"}]}
    continued = parse(c.post(url + f"/conversations/{conversation}/answers/stream", headers=headers, json=answer))
    assert continued == parse(c.post(url + f"/conversations/{conversation}/answers/stream", headers=headers, json=answer))
    assert "A short calendar" in next(e["text"] for e in continued if e["type"] == "assistant.completed")
    conv = c.app.state.chat_store.conversations[conversation]
    assert len([message for message in conv.history if message["role"] == "tool"]) == 1
    answer["request_id"] = uid()
    assert c.post(url + f"/conversations/{conversation}/answers/stream", headers=headers, json=answer).status_code == 409
    assert c.get(url + f"/conversations/{conversation}", headers=headers).json()["revision"] == 2


def test_answers_must_be_whole_unique_valid_and_correlated(tmp_path):
    c, url, headers, recipient = client(tmp_path)
    pending = next(e for e in parse(c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Help plan", "care_recipient_id": recipient})) if e["type"] == "questions.required")
    path = url + f"/conversations/{pending['conversation_id']}/answers/stream"
    body = {"request_id": uid(), "expected_revision": 1, "tool_call_id": pending["tool_call_id"], "answers": [{"question_id": "focus", "option_id": "week"}]}
    assert c.post(path, headers=headers, json=body).status_code == 422
    body["answers"].append({"question_id": "focus", "option_id": "today"})
    assert c.post(path, headers=headers, json=body).status_code == 422
    body["answers"][1] = {"question_id": "format", "option_id": "unknown"}
    assert c.post(path, headers=headers, json=body).status_code == 422
    body["tool_call_id"] = "wrong"
    assert c.post(path, headers=headers, json=body).status_code == 409


def test_scope_auth_cancellation_stale_revision_and_no_history_injection(tmp_path):
    c, url, headers, recipient = client(tmp_path)
    assert c.get(url + "/capability").status_code == 401
    assert c.get(url.replace(url.split("/")[4], uid()) + "/capability", headers=headers).status_code == 403
    body = {"request_id": uid(), "message": "Help plan", "care_recipient_id": recipient}
    assert c.post(url + "/stream", headers=headers, json={**body, "messages": []}).status_code == 422
    pending = next(e for e in parse(c.post(url + "/stream", headers=headers, json=body)) if e["type"] == "questions.required")
    conversation = pending["conversation_id"]
    cancel = {"request_id": uid(), "expected_revision": 1}
    cancelled = c.post(url + f"/conversations/{conversation}/cancel", headers=headers, json=cancel)
    assert cancelled.status_code == 200
    assert cancelled.json() == c.post(url + f"/conversations/{conversation}/cancel", headers=headers, json=cancel).json()
    assert cancelled.json()["revision"] == 2
    assert c.app.state.chat_store.conversations[conversation].history == []
    body.update(request_id=uid(), conversation_id=conversation, expected_revision=1)
    assert c.post(url + "/stream", headers=headers, json=body).status_code == 409


def test_configured_provider_failure_never_falls_back_to_debug(tmp_path, monkeypatch):
    c, url, headers, recipient = client(tmp_path, lm_studio_url="http://127.0.0.1:9/v1")
    calls = []
    def failed(self, *args):
        calls.append(1)
        raise ChatFailure("connection_error", True)
        yield
    monkeypatch.setattr(LMStudioAdapter, "stream_chat", failed)
    assert c.get(url + "/capability", headers=headers, params={"care_recipient_id": recipient}).json()["mode"] == "provider"
    body = {"request_id": uid(), "message": "Hello", "care_recipient_id": recipient}
    events = parse(c.post(url + "/stream", headers=headers, json=body))
    assert [e["type"] for e in events] == ["conversation.started", "stream.error", "stream.done"]
    assert "Debug" not in json.dumps(events)
    assert events == parse(c.post(url + "/stream", headers=headers, json=body))
    assert len(calls) == 1
    cap = c.get(url + "/capability", headers=headers, params={"care_recipient_id": recipient}).json()
    assert cap["state"] == "unavailable" and cap["configured"]


def test_provider_tool_response_correlated_one_time_and_boundary_closed(tmp_path, monkeypatch):
    c, url, headers, recipient = client(tmp_path, lm_studio_url="http://127.0.0.1:9/v1")
    calls, closed = [], []
    def transport(self, context, messages, tools):
        calls.append(messages)
        try:
            if messages[-1]["role"] == "tool":
                yield chunk({"content": "Final answer"}, "stop")
            else:
                yield from tool_chunks()
                raise AssertionError("Must stop at tool boundary")
        finally:
            closed.append(True)
    monkeypatch.setattr(LMStudioAdapter, "stream_chat", transport)
    pending = next(e for e in parse(c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Help plan", "care_recipient_id": recipient})) if e["type"] == "questions.required")
    answer = {"request_id": uid(), "expected_revision": 1, "tool_call_id": "call_one", "answers": [{"question_id": "format", "option_id": "brief"}]}
    path = url + f"/conversations/{pending['conversation_id']}/answers/stream"
    events = parse(c.post(path, headers=headers, json=answer))
    assert next(e["text"] for e in events if e["type"] == "assistant.completed") == "Final answer"
    assert len(calls) == 2 and len(closed) == 2
    assert calls[1][-2]["tool_calls"][0]["id"] == calls[1][-1]["tool_call_id"] == "call_one"
    assert calls[0][0]["role"] == "system" and "Current bounded context" in calls[0][0]["content"]


def test_external_config_is_not_consent_and_keys_are_not_returned(tmp_path):
    c, url, headers, recipient = client(tmp_path, llm_provider="cerebras", llm_base_url="https://api.cerebras.ai/v1", llm_model="gpt-oss-120b", llm_api_key="test-only-not-credential", llm_external_data_enabled=True)
    cap = c.get(url + "/capability", headers=headers, params={"care_recipient_id": recipient}).json()
    assert cap["configured"] and cap["requires_consent"] and cap["mode"] == "unavailable"
    assert "test-only-not-credential" not in json.dumps(cap)
    response = c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Hello", "care_recipient_id": recipient})
    assert response.status_code == 403


def test_bounded_standard_sse_transport_and_gate(monkeypatch):
    settings = Settings(_env_file=None, lm_studio_url="http://127.0.0.1:9/v1")
    adapter = LMStudioAdapter(settings)
    seen = []
    content = ("data: " + json.dumps(chunk({"content": "Hello"})) + "\n\ndata: " + json.dumps(chunk({}, "stop")) + "\n\ndata: [DONE]\n\n").encode()
    def mocked(request):
        seen.append(request)
        return io.BytesIO(content)
    monkeypatch.setattr(adapter, "_open_request", mocked)
    result = list(adapter.stream_chat({"privacy_boundary": {}}, [{"role": "user", "content": "Hello"}], []))
    assert len(result) == 2
    payload = json.loads(seen[0].data)
    assert seen[0].full_url.endswith("/v1/chat/completions")
    assert payload["stream"] is True and "reasoning_effort" not in payload
    assert len(seen[0].data) <= MAX_LLM_REQUEST_BYTES
    with pytest.raises(ChatFailure, match="context_budget_exceeded"):
        list(adapter.stream_chat({}, [{"role": "user", "content": "x" * 40000}], []))


@pytest.mark.parametrize("model,provider,streaming,tools,level", [
    ("gpt-oss-120b", "cerebras", True, True, "native"),
    ("retired-or-unknown", "cerebras", False, False, "unverified"),
    ("qwen3.6-35b-a3b", "lmstudio", True, True, "default"),
    ("lmstudio-community/Qwen2.5-7B-Instruct-GGUF", "lmstudio", True, True, "native"),
    ("custom-model", "openai_compatible", True, False, "unverified"),
])
def test_capabilities_derive_from_model_transport(model, provider, streaming, tools, level):
    from app.chat_protocol import provider_chat_support
    settings = Settings(_env_file=None, llm_model=model)
    support = provider_chat_support(settings, {"provider": provider})
    assert support == {"streaming": streaming, "tools_supported": tools, "tool_support": level}


def test_unconfigured_flags_and_completed_wire_status(tmp_path):
    c, url, headers, recipient = client(tmp_path)
    cap = c.get(url + "/capability", headers=headers, params={"care_recipient_id": recipient}).json()
    assert not cap["streaming"] and not cap["tools_supported"] and not cap["live_connection_verified"]
    events = parse(c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Hello", "care_recipient_id": recipient}))
    assert events[-1]["status"] == "completed"
    conv = c.get(url + f"/conversations/{events[0]['conversation_id']}", headers=headers).json()
    assert conv["status"] == "ready"


@pytest.mark.parametrize("delta", [
    {"tool_calls": [{"index": 0, "function": None}]},
    {"tool_calls": {"index": 0}},
    {"channel": "analysis", "tool_calls": [{"index": 0, "function": {}}]},
])
def test_malformed_or_private_tool_fragments_fail_closed(delta):
    with pytest.raises(ChatFailure, match="invalid_tool_call"):
        ChatAccumulator().push(chunk(delta))


def test_raw_serialization_split_across_deltas_is_rejected():
    accumulator = ChatAccumulator()
    accumulator.push(chunk({"content": "<"}))
    with pytest.raises(ChatFailure, match="invalid_provider_stream"):
        accumulator.push(chunk({"content": "|analysis|>private"}))
    with pytest.raises(ChatFailure, match="invalid_tool_call"):
        ChatAccumulator(allow_tools=False).push(next(part for part in tool_chunks() if part["choices"][0]["delta"].get("tool_calls")))


def test_cancel_wire_snapshot_decodes_and_removes_pending_tool(tmp_path):
    c, url, headers, recipient = client(tmp_path)
    pending = next(e for e in parse(c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Help plan", "care_recipient_id": recipient})) if e["type"] == "questions.required")
    path = url + f"/conversations/{pending['conversation_id']}"
    result = c.post(path + "/cancel", headers=headers, json={"request_id": uid(), "expected_revision": 1}).json()
    assert result["pending"] is None and result["messages"] == []
    assert result["status"] == "cancelled" and result["revision"] == 2


def test_failed_answer_retries_same_batch_without_duplicate_result(tmp_path, monkeypatch):
    c, url, headers, recipient = client(tmp_path, lm_studio_url="http://127.0.0.1:9/v1")
    attempts = []
    def transport(self, context, messages, tools):
        if messages[-1]["role"] != "tool":
            yield from tool_chunks()
        else:
            attempts.append(messages)
            if len(attempts) == 1:
                raise ChatFailure("connection_error", True)
            yield chunk({"content": "Resumed answer"}, "stop")
    monkeypatch.setattr(LMStudioAdapter, "stream_chat", transport)
    pending = next(e for e in parse(c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Help plan", "care_recipient_id": recipient})) if e["type"] == "questions.required")
    path = url + f"/conversations/{pending['conversation_id']}"
    body = {"request_id": uid(), "expected_revision": 1, "tool_call_id": "call_one", "answers": [{"question_id": "format", "option_id": "brief"}]}
    failed = parse(c.post(path + "/answers/stream", headers=headers, json=body))
    assert failed[-1]["status"] == "failed"
    restored = c.get(path, headers=headers).json()
    assert restored["pending"]["tool_call_id"] == "call_one"
    body.update(request_id=uid(), expected_revision=restored["revision"])
    resumed = parse(c.post(path + "/answers/stream", headers=headers, json=body))
    assert resumed[-1]["status"] == "completed"
    history = c.app.state.chat_store.conversations[pending["conversation_id"]].history
    assert len([item for item in history if item["role"] == "tool"]) == 1
    assert all(message[-2]["tool_calls"][0]["id"] == message[-1]["tool_call_id"] == "call_one" for message in attempts)


def test_plain_compatible_transport_omits_tool_parameters(monkeypatch):
    settings = Settings(_env_file=None, lm_studio_url="http://127.0.0.1:9/v1")
    adapter = LMStudioAdapter(settings)
    captured = []
    def mocked(request):
        captured.append(json.loads(request.data))
        return io.BytesIO(("data: " + json.dumps(chunk({"content": "Hello"}, "stop")) + "\n\n").encode())
    monkeypatch.setattr(adapter, "_open_request", mocked)
    list(adapter.stream_chat({}, [{"role": "user", "content": "Hello"}], []))
    assert "tools" not in captured[0] and "tool_choice" not in captured[0] and "parallel_tool_calls" not in captured[0]
    list(adapter.stream_chat({}, [{"role": "user", "content": "Hello"}], [{"type": "function"}]))
    assert captured[1]["parallel_tool_calls"] is False


def test_unknown_compatible_model_keeps_plain_chat_without_tools(tmp_path, monkeypatch):
    c, url, headers, recipient = client(tmp_path, llm_provider="openai_compatible", llm_base_url="https://api.example.com/v1", llm_model="custom-model", llm_api_key="test-only-not-credential", llm_external_data_enabled=True)
    home = url.split("/")[4]
    assert c.post(f"/api/v1/homes/{home}/consents", headers=headers, json={"purpose": "external_ai_processing", "policy_version": "test", "care_recipient_id": recipient}).status_code == 200
    seen = []
    def transport(self, context, messages, tools):
        seen.append(tools)
        yield chunk({"content": "Ordinary answer"}, "stop")
    monkeypatch.setattr(LMStudioAdapter, "stream_chat", transport)
    cap = c.get(url + "/capability", headers=headers, params={"care_recipient_id": recipient}).json()
    assert cap["state"] == "configured" and cap["streaming"] and not cap["tools_supported"]
    assert not cap["live_connection_verified"]
    events = parse(c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Hello", "care_recipient_id": recipient}))
    assert events[-1]["status"] == "completed" and seen == [[]]
    def unexpected(self, context, messages, tools):
        yield from tool_chunks()
    monkeypatch.setattr(LMStudioAdapter, "stream_chat", unexpected)
    events = parse(c.post(url + "/stream", headers=headers, json={"request_id": uid(), "message": "Help plan", "care_recipient_id": recipient}))
    assert not any(event["type"] == "questions.required" for event in events)
    assert next(event["code"] for event in events if event["type"] == "stream.error") == "invalid_tool_call"
