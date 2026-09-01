from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from agentpm import (
    AfterKnowledgeRetrievalDecision,
    AfterKnowledgeRetrievalInput,
    BeforeKnowledgeRequestDecision,
    BeforeMemoryOperationDecision,
    BeforeMemoryReadDecision,
    BeforeMemoryWriteDecision,
    BeforeModelRequestDecision,
    BeforeToolCallDecision,
    BeforeToolSelectionDecision,
    EmbeddingProviderRequest,
    EmbeddingProviderResult,
    HarnessClient,
    HarnessProtocolError,
    HookDecision,
    KnowledgeRuntimeRequest,
    KnowledgeRuntimeResult,
)


def _write_fake_harness(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake_harness.py"
    script.write_text(body, encoding="utf-8")
    return str(script)


def _write_fake_agentpm_command(tmp_path: Path, body: str) -> str:
    script = tmp_path / "agentpm"
    script.write_text(f"#!/usr/bin/env python3\n{body}", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def _common_harness(body: str) -> str:
    body = textwrap.dedent(body).strip()
    prefix = (
        "import json\n"
        "import sys\n\n"
        'PROTOCOL = "agentpm-harness-machine"\n\n'
        "def write(frame):\n"
        '    frame = {"protocol": PROTOCOL, "version": 1, **frame}\n'
        '    sys.stdout.write(json.dumps(frame) + "\\n")\n'
        "    sys.stdout.flush()\n\n"
        'write({"kind": "event", "method": "preflight", "payload": {"status": "ready"}})\n\n'
    )
    return prefix + body + "\n"


def test_harness_client_initializes_streams_events_runs_and_shuts_down(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "preflight":
                    write({"kind": "response", "id": frame["id"], "payload": {"status": "ready_with_warnings", "diagnostics": []}})
                elif frame.get("method") == "start_run":
                    write({"kind": "event", "method": "harness_event", "payload": {"event_type": "run_started", "payload": {"fields": {"input": frame["payload"]["input"]}}}})
                    write({"kind": "response", "id": frame["id"], "payload": {"status": "ended", "output": {"message": "done"}, "report": {"trace_path": "events.jsonl"}}})
                elif frame.get("method") == "shutdown":
                    write({"kind": "response", "id": frame["id"], "payload": {"shutdown": True}})
                    break
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])
    assert client.wait_for_event(lambda event: event.get("status") == "ready")["status"] == "ready"
    assert client.initialize()["session"] == {"protocol": "agentpm-harness-machine", "version": 1}
    assert client.preflight()["status"] == "ready_with_warnings"
    assert client.run("hello")["output"] == {"message": "done"}
    assert client.wait_for_event(lambda event: event.get("event_type") == "run_started")[
        "payload"
    ] == {"fields": {"input": "hello"}}
    assert client.shutdown() == {"shutdown": True}


def test_harness_client_passes_installed_agent_refs_as_positional_selector(
    tmp_path: Path,
) -> None:
    command = _write_fake_agentpm_command(
        tmp_path,
        (
            "import json\n"
            "import sys\n\n"
            'PROTOCOL = "agentpm-harness-machine"\n\n'
            "def write(frame):\n"
            '    frame = {"protocol": PROTOCOL, "version": 1, **frame}\n'
            '    sys.stdout.write(json.dumps(frame) + "\\n")\n'
            "    sys.stdout.flush()\n\n"
            "for line in sys.stdin:\n"
            "    frame = json.loads(line)\n"
            '    if frame.get("method") == "initialize":\n'
            '        write({"kind": "response", "id": frame["id"], "payload": {"argv": sys.argv[1:], "session": {"protocol": PROTOCOL, "version": 1}}})\n'
        ),
    )
    client = HarnessClient(
        agentpm_path=command,
        agent="@zack/support-agent@0.1.0",
    )

    assert client.initialize()["argv"] == [
        "harness",
        "@zack/support-agent@0.1.0",
        "--machine",
    ]
    client.stop()


def test_harness_client_iterates_buffered_and_future_events_once(
    tmp_path: Path,
) -> None:
    script = _write_fake_harness(
        tmp_path,
        (
            "import json\n"
            "import sys\n"
            "import time\n\n"
            'PROTOCOL = "agentpm-harness-machine"\n\n'
            "def write(frame):\n"
            '    frame = {"protocol": PROTOCOL, "version": 1, **frame}\n'
            '    sys.stdout.write(json.dumps(frame) + "\\n")\n'
            "    sys.stdout.flush()\n\n"
            'write({"kind": "event", "method": "harness_event", "payload": {"event_type": "A"}})\n'
            "time.sleep(0.02)\n"
            'write({"kind": "event", "method": "harness_event", "payload": {"event_type": "B"}})\n'
        ),
    )
    client = HarnessClient(
        agentpm_path=sys.executable,
        args=[script],
        request_timeout_seconds=0.5,
    )
    events = client.events()

    assert next(events)["event_type"] == "A"
    assert next(events)["event_type"] == "B"
    with pytest.raises(StopIteration):
        next(events)


def test_harness_client_routes_model_hook_and_approval_callbacks(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            start_run_id = None
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    write({"kind": "response", "id": frame["id"], "payload": {"registered": True, "service": {"role": frame["payload"]["role"], "registry_id": frame["payload"]["registry_id"]}}})
                elif frame.get("method") == "start_run":
                    start_run_id = frame["id"]
                    write({"kind": "request", "id": "host-model-1", "method": "host_service", "payload": {"role": "model", "registry_id": "company-model", "method": "generate", "payload": {"request": {"phase_id": "classify"}}}})
                elif frame.get("kind") == "response" and frame.get("id") == "host-model-1":
                    write({"kind": "request", "id": "host-hook-1", "method": "host_service", "payload": {"role": "hook", "registry_id": "sdk-hooks", "method": "before_tool_call", "payload": {"hook": "before_tool_call", "input": {"arguments": {"body": "original"}}}}})
                elif frame.get("kind") == "response" and frame.get("id") == "host-hook-1":
                    write({"kind": "request", "id": "host-approval-1", "method": "host_service", "payload": {"role": "approval", "registry_id": "controller", "method": "request_approval", "payload": {"checkpoint": {"id": "gate"}}}})
                elif frame.get("kind") == "response" and frame.get("id") == "host-approval-1":
                    write({"kind": "response", "id": start_run_id, "payload": {"status": "ended", "output": {"approval": frame["payload"]["decision"]}, "report": {}}})
            """),
    )
    calls: list[str] = []
    client = HarnessClient(agentpm_path=sys.executable, args=[script])

    def model_provider(payload: Any) -> Any:
        calls.append(f"model:{payload}")
        return {
            "assistant_content": None,
            "actions": [],
            "usage": {},
            "finish_reason": "stop",
            "provider_metadata": {},
        }

    def tool_hook(payload: Any) -> BeforeToolCallDecision:
        calls.append(f"hook:{payload}")
        return {"decision": "continue", "patch": {"arguments": {"body": "patched"}}}

    def approval(payload: Any) -> str:
        calls.append(f"approval:{payload}")
        return "approve"

    client.register_model_provider("company-model", model_provider).on_before_tool_call(
        tool_hook
    ).on_approval(approval)

    result = client.run("use host services")
    assert result["status"] == "ended"
    assert result["output"] == {"approval": "approve"}
    assert len(calls) == 3
    client.stop()


def test_harness_client_maps_callback_timeouts_to_error_frames(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            start_run_id = None
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    write({"kind": "response", "id": frame["id"], "payload": {"registered": True}})
                elif frame.get("method") == "start_run":
                    start_run_id = frame["id"]
                    write({"kind": "request", "id": "host-hook-timeout", "method": "host_service", "payload": {"role": "hook", "registry_id": "sdk-hooks", "method": "before_tool_call", "payload": {"input": {"arguments": {}}}}})
                elif frame.get("kind") == "error" and frame.get("id") == "host-hook-timeout":
                    write({"kind": "response", "id": start_run_id, "payload": {"status": "ended", "output": {"code": frame["error"]["code"]}, "report": {}}})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])

    def slow_hook(_: Any) -> HookDecision:
        time.sleep(0.05)
        return {"decision": "continue"}

    client.register_hook("before_tool_call", slow_hook, request_timeout_ms=5)

    result = client.run("timeout hook")
    assert result["output"] == {"code": "host_service_callback_failed"}
    client.stop()


def test_harness_client_registers_repeated_hooks_as_ordered_bindings(
    tmp_path: Path,
) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            registrations = []
            start_run_id = None
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    registrations.append(frame["payload"]["registry_id"])
                    write({"kind": "response", "id": frame["id"], "payload": {"registered": True}})
                elif frame.get("method") == "start_run":
                    start_run_id = frame["id"]
                    write({"kind": "request", "id": "hook-a", "method": "host_service", "payload": {"role": "hook", "registry_id": registrations[0], "method": "before_tool_selection", "payload": {"input": {"candidates": [{"canonical_id": "t1"}, {"canonical_id": "t2"}]}}}})
                elif frame.get("kind") == "response" and frame.get("id") == "hook-a":
                    write({"kind": "request", "id": "hook-b", "method": "host_service", "payload": {"role": "hook", "registry_id": registrations[1], "method": "before_tool_selection", "payload": {"input": {"candidates": [{"canonical_id": "t1"}]}}}})
                elif frame.get("kind") == "response" and frame.get("id") == "hook-b":
                    write({"kind": "response", "id": start_run_id, "payload": {"status": "ended", "output": {"registrations": registrations, "second": frame["payload"]["patch"]["candidate_ids"]}, "report": {}}})
            """),
    )
    seen: list[Any] = []
    client = HarnessClient(agentpm_path=sys.executable, args=[script])

    def first_hook(_: Any) -> BeforeToolSelectionDecision:
        return {"decision": "continue", "patch": {"candidate_ids": ["t1"]}}

    def second_hook(payload: Any) -> BeforeToolSelectionDecision:
        seen.append(payload)
        return {"decision": "continue", "patch": {"candidate_ids": ["t1"]}}

    client.on_before_tool_selection(first_hook).on_before_tool_selection(second_hook)

    result = client.run("compose hooks")
    assert result["output"] == {
        "registrations": ["sdk-hooks", "sdk-hooks-1"],
        "second": ["t1"],
    }
    assert seen == [{"candidates": [{"canonical_id": "t1"}]}]
    client.stop()


def test_harness_client_advertises_typed_hook_helpers(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            registrations = []
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    registrations.append({
                        "registry_id": frame["payload"]["registry_id"],
                        "hooks": frame["payload"]["hooks"],
                        "capabilities": frame["payload"]["capabilities"],
                    })
                    write({"kind": "response", "id": frame["id"], "payload": {"registered": True}})
                elif frame.get("method") == "start_run":
                    write({"kind": "response", "id": frame["id"], "payload": {"status": "ended", "output": {"registrations": registrations}, "report": {}}})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])

    def before_knowledge_request(_: Any) -> BeforeKnowledgeRequestDecision:
        return {"decision": "continue", "patch": {"query": "narrowed", "top_k": 2}}

    def after_knowledge_retrieval(
        _: AfterKnowledgeRetrievalInput,
    ) -> AfterKnowledgeRetrievalDecision:
        return {
            "decision": "continue",
            "patch": {
                "results": [
                    {
                        "source_id": "src_1",
                        "chunk_id": "chunk_1",
                        "text": "model-visible text",
                    }
                ]
            },
        }

    def before_memory_read(_: Any) -> BeforeMemoryReadDecision:
        return {"decision": "continue", "patch": {"limit": 1}}

    def before_memory_write(_: Any) -> BeforeMemoryWriteDecision:
        return {"decision": "continue", "patch": {"content": {"safe": True}}}

    def before_memory_operation(_: Any) -> BeforeMemoryOperationDecision:
        return {
            "decision": "continue",
            "patch": {"model_guidance": "Prefer recent records."},
        }

    client.on_before_tool_call(
        lambda _: {"decision": "continue"},
        registry_id="a",
    ).on_before_model_request(
        lambda _: {"decision": "continue"},
        registry_id="b",
    ).on_before_knowledge_request(
        before_knowledge_request,
        registry_id="c",
    ).on_after_knowledge_retrieval(
        after_knowledge_retrieval,
        registry_id="d",
    ).on_before_memory_read(
        before_memory_read,
        registry_id="e",
    ).on_before_memory_write(
        before_memory_write,
        registry_id="f",
    ).on_before_memory_operation(
        before_memory_operation,
        registry_id="g",
    )

    result = client.run("advertise hooks")
    assert result["output"] == {
        "registrations": [
            {
                "registry_id": "a",
                "hooks": ["before_tool_call"],
                "capabilities": {"hooks": ["before_tool_call"]},
            },
            {
                "registry_id": "b",
                "hooks": ["before_model_request"],
                "capabilities": {"hooks": ["before_model_request"]},
            },
            {
                "registry_id": "c",
                "hooks": ["before_knowledge_request"],
                "capabilities": {"hooks": ["before_knowledge_request"]},
            },
            {
                "registry_id": "d",
                "hooks": ["after_knowledge_retrieval"],
                "capabilities": {"hooks": ["after_knowledge_retrieval"]},
            },
            {
                "registry_id": "e",
                "hooks": ["before_memory_read"],
                "capabilities": {"hooks": ["before_memory_read"]},
            },
            {
                "registry_id": "f",
                "hooks": ["before_memory_write"],
                "capabilities": {"hooks": ["before_memory_write"]},
            },
            {
                "registry_id": "g",
                "hooks": ["before_memory_operation"],
                "capabilities": {"hooks": ["before_memory_operation"]},
            },
        ]
    }
    client.stop()


def test_harness_client_advertises_role_specific_host_service_capabilities(
    tmp_path: Path,
) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            registrations = []
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    registrations.append({
                        "role": frame["payload"]["role"],
                        "registry_id": frame["payload"]["registry_id"],
                        "capabilities": frame["payload"]["capabilities"],
                        "hooks": frame["payload"]["hooks"],
                    })
                    write({"kind": "response", "id": frame["id"], "payload": {"registered": True}})
                elif frame.get("method") == "start_run":
                    write({"kind": "response", "id": frame["id"], "payload": {"status": "ended", "output": {"registrations": registrations}, "report": {}}})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])
    client.register_model_provider(
        "company-model",
        lambda _: {},
        {
            "model": "model-1",
            "context_window_tokens": 4096,
            "semantic_actions": True,
            "structured_output": True,
            "multimodal_input": False,
            "usage_reporting": True,
        },
    ).register_host_provider(
        "embedding",
        "embedder",
        lambda _: {},
        {
            "embedding_spaces": [
                {
                    "provider": "embedder",
                    "model": "embed-1",
                    "dimensions": 1536,
                    "normalized": True,
                }
            ]
        },
    ).on_before_tool_call(
        lambda _: {"decision": "continue"},
        registry_id="hook-a",
    ).on_approval(
        lambda _: "approve",
        {"approval": True, "cancellation": True},
    )

    result = client.run("advertise capabilities")
    assert result["output"] == {
        "registrations": [
            {
                "role": "model",
                "registry_id": "company-model",
                "capabilities": {
                    "provider": "company-model",
                    "model": "model-1",
                    "semantic_actions": True,
                    "structured_output": True,
                    "multimodal_input": False,
                    "context_window_tokens": 4096,
                    "usage_reporting": True,
                },
                "hooks": [],
            },
            {
                "role": "embedding",
                "registry_id": "embedder",
                "capabilities": {
                    "embedding_spaces": [
                        {
                            "provider": "embedder",
                            "model": "embed-1",
                            "dimensions": 1536,
                            "normalized": True,
                        }
                    ]
                },
                "hooks": [],
            },
            {
                "role": "hook",
                "registry_id": "hook-a",
                "capabilities": {"hooks": ["before_tool_call"]},
                "hooks": ["before_tool_call"],
            },
            {
                "role": "approval",
                "registry_id": "controller",
                "capabilities": {"approval": True, "cancellation": True},
                "hooks": [],
            },
        ]
    }
    client.stop()


def test_harness_client_registers_typed_embedding_and_knowledge_providers(
    tmp_path: Path,
) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            registrations = []
            start_run_id = None
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    registrations.append({
                        "role": frame["payload"]["role"],
                        "registry_id": frame["payload"]["registry_id"],
                        "capabilities": frame["payload"]["capabilities"],
                    })
                    write({"kind": "response", "id": frame["id"], "payload": {
                        "registered": True,
                        "service": {"role": frame["payload"]["role"], "registry_id": frame["payload"]["registry_id"]},
                        "active": True,
                    }})
                elif frame.get("method") == "start_run":
                    start_run_id = frame["id"]
                    write({"kind": "request", "id": "embed-1", "method": "host_service", "payload": {
                        "role": "embedding",
                        "registry_id": "embedder",
                        "method": "embed",
                        "payload": {"provider": "openai", "model": "text-embedding-3-small", "dimensions": 3, "normalized": True, "text": "hello"},
                    }})
                elif frame.get("kind") == "response" and frame.get("id") == "embed-1":
                    write({"kind": "request", "id": "knowledge-1", "method": "host_service", "payload": {
                        "role": "knowledge",
                        "registry_id": "kb",
                        "method": "retrieve",
                        "payload": {"request": {"package": "@zack/docs", "version": "0.1.0", "mode": "vector_query", "query": "hello", "top_k": 1, "return_citations": True}},
                    }})
                elif frame.get("kind") == "response" and frame.get("id") == "knowledge-1":
                    write({"kind": "response", "id": start_run_id, "payload": {
                        "status": "ended",
                        "output": {"registrations": registrations, "knowledge": frame["payload"]},
                        "report": {},
                    }})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])
    calls: list[str] = []

    def embedding_provider(request: EmbeddingProviderRequest) -> EmbeddingProviderResult:
        calls.append(f"embedding:{request['provider']}:{request['model']}:{request['text']}")
        return {
            "vector": [1.0, 0.0, 0.0],
            "provider": request["provider"],
            "model": request["model"],
            "dimensions": 3,
        }

    def knowledge_runtime(request: KnowledgeRuntimeRequest) -> KnowledgeRuntimeResult:
        query = request.get("query") or ""
        calls.append(f"knowledge:{request['package']}:{request['mode']}:{query}")
        return {
            "ok": True,
            "package": request["package"],
            "version": request["version"],
            "mode": request["mode"],
            "query": query,
            "results": [
                {
                    "rank": 1,
                    "score": 0.9,
                    "chunk_id": "chunk-1",
                    "source_id": "source-1",
                    "text": "answer",
                }
            ],
        }

    client.register_embedding_provider(
        "embedder",
        embedding_provider,
        {
            "embedding_spaces": [
                {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "dimensions": 3,
                    "normalized": True,
                }
            ]
        },
    ).register_knowledge_runtime(
        "kb",
        knowledge_runtime,
        {
            "modes": ["vector_query"],
            "features": ["citations"],
            "packages": [{"package": "@zack/docs", "version": "0.1.0", "ready": True}],
        },
    )

    result = client.run("use typed knowledge providers")
    assert calls == [
        "embedding:openai:text-embedding-3-small:hello",
        "knowledge:@zack/docs:vector_query:hello",
    ]
    assert result["output"] == {
        "registrations": [
            {
                "role": "embedding",
                "registry_id": "embedder",
                "capabilities": {
                    "embedding_spaces": [
                        {
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "dimensions": 3,
                            "normalized": True,
                        }
                    ]
                },
            },
            {
                "role": "knowledge",
                "registry_id": "kb",
                "capabilities": {
                    "modes": ["vector_query"],
                    "features": ["citations"],
                    "packages": [{"package": "@zack/docs", "version": "0.1.0", "ready": True}],
                },
            },
        ],
        "knowledge": {
            "ok": True,
            "package": "@zack/docs",
            "version": "0.1.0",
            "mode": "vector_query",
            "query": "hello",
            "results": [
                {
                    "rank": 1,
                    "score": 0.9,
                    "chunk_id": "chunk-1",
                    "source_id": "source-1",
                    "text": "answer",
                }
            ],
        },
    }
    embedding_registration = client.host_service_registration("embedding", "embedder")
    knowledge_registration = client.host_service_registration("knowledge", "kb")
    assert embedding_registration is not None
    assert knowledge_registration is not None
    assert embedding_registration["active"] is True
    assert knowledge_registration["active"] is True
    client.stop()


def test_harness_client_flushes_registrations_added_after_initialize(
    tmp_path: Path,
) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            registrations = []
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    registrations.append(frame["payload"]["registry_id"])
                    write({"kind": "response", "id": frame["id"], "payload": {"registered": True}})
                elif frame.get("method") == "start_run":
                    write({"kind": "response", "id": frame["id"], "payload": {"status": "ended", "output": {"registrations": registrations}, "report": {}}})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])

    def late_hook(_: Any) -> BeforeToolCallDecision:
        return {"decision": "continue"}

    client.initialize()
    client.on_before_tool_call(late_hook)

    assert client.run("late hook")["output"] == {"registrations": ["sdk-hooks"]}
    client.stop()


def test_harness_client_stores_inactive_host_registration_reason(
    tmp_path: Path,
) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "register_host_service":
                    write({"kind": "response", "id": frame["id"], "payload": {
                        "registered": True,
                        "service": {"role": frame["payload"]["role"], "registry_id": frame["payload"]["registry_id"]},
                        "active": False,
                        "reason": "configured KnowledgeRuntime could not attest the requested package",
                    }})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])

    client.register_host_provider("knowledge", "kb", lambda _: {"ok": True})
    client.initialize()

    assert client.host_service_registration("knowledge", "kb") == {
        "registered": True,
        "service": {"role": "knowledge", "registry_id": "kb"},
        "active": False,
        "reason": "configured KnowledgeRuntime could not attest the requested package",
    }
    assert len(client.host_service_registrations()) == 1
    client.stop()


def test_harness_client_cancellation_and_memory_operation_errors(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            for line in sys.stdin:
                frame = json.loads(line)
                if frame.get("method") == "initialize":
                    write({"kind": "response", "id": frame["id"], "payload": {"session": {"protocol": PROTOCOL, "version": 1}, "preflight": {"status": "ready"}, "required_host_services": []}})
                elif frame.get("method") == "cancel_run":
                    write({"kind": "response", "id": frame["id"], "payload": {"accepted": True, "status": "cancelled"}})
                elif frame.get("method") == "memory_operation":
                    write({"kind": "error", "id": frame["id"], "error": {"code": "memory_operation_unavailable", "message": "not live yet"}})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])
    client.initialize()
    assert client.cancel_run() == {"accepted": True, "status": "cancelled"}
    with pytest.raises(HarnessProtocolError) as err:
        client.invoke_memory_operation({"operation": "compact"})
    assert err.value.code == "memory_operation_unavailable"
    client.stop()


def test_harness_client_rejects_pending_requests_on_process_exit(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        "import time, sys\ntime.sleep(0.02)\nsys.exit(7)\n",
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])
    with pytest.raises(RuntimeError, match="Harness process exited"):
        client.initialize()


def test_harness_client_fails_fast_after_malformed_stdout(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        (
            "import sys\n"
            "import time\n"
            "for line in sys.stdin:\n"
            '    sys.stdout.write("not-json\\n")\n'
            "    sys.stdout.flush()\n"
            "    while True:\n"
            "        time.sleep(1)\n"
        ),
    )
    client = HarnessClient(
        agentpm_path=sys.executable,
        args=[script],
        request_timeout_seconds=0.5,
    )

    with pytest.raises(json.JSONDecodeError):
        client.initialize()
    with pytest.raises(json.JSONDecodeError):
        client.preflight()


def test_harness_client_exposes_protocol_error_codes(tmp_path: Path) -> None:
    script = _write_fake_harness(
        tmp_path,
        _common_harness("""
            for line in sys.stdin:
                frame = json.loads(line)
                write({"kind": "error", "id": frame["id"], "error": {"code": "bad_version", "message": "nope"}})
            """),
    )
    client = HarnessClient(agentpm_path=sys.executable, args=[script])
    with pytest.raises(HarnessProtocolError) as err:
        client.initialize()
    assert err.value.code == "bad_version"
    client.stop()


def test_real_agentpm_harness_process_when_fixture_env_is_set() -> None:
    cli = os.environ.get("AGENTPM_HARNESS_CLI")
    workspace = os.environ.get("AGENTPM_HARNESS_WORKSPACE")
    if not cli or not workspace or not Path(cli).exists() or not Path(workspace).exists():
        pytest.skip("AGENTPM_HARNESS_CLI and AGENTPM_HARNESS_WORKSPACE are not set")
    client = HarnessClient(agentpm_path=cli, cwd=workspace)
    calls: list[str] = []

    def model_provider(_: Any) -> dict[str, Any]:
        calls.append("model")
        return {
            "assistant_content": "real CLI SDK host model response",
            "actions": [],
            "usage": {},
            "finish_reason": "stop",
            "provider_metadata": {},
        }

    def before_model_request(_: Any) -> BeforeModelRequestDecision:
        calls.append("before_model_request")
        return {"decision": "continue"}

    def before_tool_call(_: Any) -> BeforeToolCallDecision:
        calls.append("before_tool_call")
        return {"decision": "continue"}

    def approval(_: Any) -> str:
        calls.append("approval")
        return "approve"

    client.register_model_provider("company-model", model_provider).on_before_model_request(
        before_model_request
    ).on_before_tool_call(before_tool_call).on_approval(approval)

    info = client.initialize()
    assert info["session"] == {"protocol": "agentpm-harness-machine", "version": 1}
    result = client.run("Run the SDK real CLI integration fixture.")
    assert result["status"] == "ended"
    assert "output" in result
    assert "report" in result
    assert "model" in calls
    assert "before_model_request" in calls
    client.shutdown()
