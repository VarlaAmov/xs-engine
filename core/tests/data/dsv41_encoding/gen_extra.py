"""Generate reference prompts only; never bless Rust output.

Run: python3 -I gen_extra.py /absolute/path/to/hf/encoding
Only encoding.py is imported from that read-only directory (stdlib dependencies).
"""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
reference_dir = Path(sys.argv[1]).resolve()
source = reference_dir / "encoding.py"
# Check the reference's imports before executing it. No third-party stubs needed.
for node in ast.walk(ast.parse(source.read_text())):
    if isinstance(node, ast.Import):
        assert all(n.name in {"copy", "json", "re"} for n in node.names)
    elif isinstance(node, ast.ImportFrom):
        assert node.module == "typing" and node.level == 0
sys.path.insert(0, str(reference_dir))
enc = importlib.import_module("encoding")
assert Path(enc.__file__).resolve() == source


def write_json(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def request(messages, **fields):
    return {"messages": messages, **fields}


def function(name, properties):
    return {"type": "function", "function": {
        "name": name, "description": "Execute " + name,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties)}}}


def call(ident, name, args):
    return {"id": ident, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


cases = []


def add(name, body, *, mode=None, effort=None, notes="", reference_messages=None,
        drop_thinking=True, context=None):
    mode = mode or ("chat" if body.get("thinking") in (False, {"type": "disabled"})
                    or body.get("reasoning_effort") == "none" else "thinking")
    messages = copy.deepcopy(reference_messages if reference_messages is not None else body["messages"])
    if body.get("tools"):
        # The HTTP toolkit inserts an empty system message when necessary.
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = copy.deepcopy(body["tools"])
    prompt = enc.encode_messages(messages, thinking_mode=mode, reasoning_effort=effort,
                                 drop_thinking=drop_thinking, context=context)
    write_json(name + ".request.json", body)
    (ROOT / (name + ".reference.txt")).write_bytes(prompt.encode("utf-8"))
    cases.append({"name": name, "request": name + ".request.json",
                  "reference": name + ".reference.txt", "mode": mode,
                  "effort": effort, "drop_thinking": drop_thinking,
                  "context": context, "notes": notes})


# Keep all five upstream files unchanged. Request variants add only transport
# fields that do not affect the reference prompt; re-encode to prove that claim.
for index in range(1, 6):
    original = json.loads((ROOT / f"test_input_{index}.json").read_text())
    body = copy.deepcopy(original if isinstance(original, dict) else {"messages": original})
    mode = body.pop("thinking_mode", "chat")
    body["thinking"] = {"type": "enabled" if mode == "thinking" else "disabled"}
    serial = 0
    for message in body["messages"]:
        calls = message.get("tool_calls")
        if calls == []:
            del message["tool_calls"]
        for tc in calls or []:
            tc.setdefault("id", f"call_{serial}")
            serial += 1
        if message["role"] == "tool":
            message.setdefault("tool_call_id", f"call_{serial - 1}")
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if block["type"] == "image_url":
                    block["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    name = f"official_{index}"
    add(name, body, mode=mode, effort=body.get("reasoning_effort"),
        notes="OpenAI transport: explicit thinking; missing call/result IDs supplied; empty tool_calls omitted; local image paths replaced by data URLs.")
    upstream = (ROOT / f"test_output_{index}.txt").read_bytes()
    assert (ROOT / (name + ".reference.txt")).read_bytes() == upstream
    # Avoid duplicating the official reference output.
    (ROOT / (name + ".reference.txt")).unlink()
    cases[-1]["reference"] = f"test_output_{index}.txt"

# Tokenizer revision is a production input. Keep the legacy repository fixture
# observable separately from the official 1017 spelling used by the main case.
legacy = copy.deepcopy(cases[-1])
legacy["name"] = "official_5_legacy_tokenizer"
legacy["image_token"] = "repository_fixture"
legacy["notes"] = "Same vision request; actual image token at IMAGE_TOKEN_ID in tokenizer-v41.json.gz."
cases.append(legacy)

simple = [{"role": "user", "content": "Question: café, 中文, <think>literal</think>?"}]
for effort in [None, "low", "high", "xhigh", "max", "none", 1, 37, 88, 100, 50, 75]:
    fields = {} if effort is None else {"reasoning_effort": effort}
    add("effort_" + ("absent" if effort is None else str(effort)), request(simple, **fields),
        effort="high" if effort == "xhigh" else None if effort == "none" else effort,
        notes="xhigh is an API alias of high (75), not a reference spelling." if effort == "xhigh" else "")
for name, fields, mode, effort in [
    ("thinking_on", {"thinking": {"type": "enabled"}}, "thinking", None),
    ("thinking_off", {"thinking": {"type": "disabled"}, "reasoning_effort": "max"}, "chat", "max"),
    ("boolean_on", {"thinking": True}, "thinking", None),
    ("boolean_off", {"thinking": False}, "chat", None),
    ("kwargs_off", {"chat_template_kwargs": {"enable_thinking": False}}, "chat", None),
    ("kwargs_high", {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}, "thinking", "high"),
    ("kwargs_100", {"chat_template_kwargs": {"reasoning_effort": 100}}, "thinking", 100),
    ("kwargs_37", {"chat_template_kwargs": {"reasoning_effort": 37}}, "thinking", 37),
    ("effort_37_off", {"thinking": False, "reasoning_effort": 37}, "chat", 37),
    ("kwargs_37_off", {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": 37}}, "chat", 37),
    ("explicit_wins", {"thinking": {"type": "enabled"}, "reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "max"}}, "thinking", "low"),
    ("none_explicit_on", {"thinking": {"type": "enabled"}, "reasoning_effort": "none"}, "thinking", None),
]:
    add(name, request(simple, **fields), mode=mode, effort=effort)

history = [
    {"role": "system", "content": "Be precise."},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "reasoning_content": "old private reasoning", "content": "a1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "reasoning_content": "second private reasoning", "content": "a2"},
    {"role": "user", "content": "q3"},
]
add("history_on", request(history, thinking=True))
add("history_off", request(history, thinking=False))
add("history_keep", request(history, thinking=True), drop_thinking=False,
    notes="Diagnostic: HTTP has no drop_thinking option; reference explicitly retains history.")

tools = [function("lookup", {"query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"}}),
         function("store", {"record": {"type": "object", "properties": {"label": {"type": "string"}}}, "tags": {"type": "array", "items": {"type": "string"}}}),
         function("finish", {"ok": {"type": "boolean"}})]
tool_history = [
    {"role": "system", "content": "Use tools carefully."},
    {"role": "user", "content": "Find and store café."},
    {"role": "assistant", "content": "Checking.", "reasoning_content": "Need both tools.",
     "tool_calls": [call("a", "lookup", {"query": "café", "limit": 2, "exact": True}),
                    call("b", "store", {"record": {"label": "中文"}, "tags": ["a", "b"]})]},
    {"role": "tool", "tool_call_id": "b", "content": "stored second"},
    {"role": "tool", "tool_call_id": "a", "content": "found first"},
    {"role": "assistant", "content": "Done.", "reasoning_content": "Both completed."},
    {"role": "user", "content": "What next?"},
]
add("tools_parallel", request(tool_history, tools=tools, thinking=True, reasoning_effort="high"), effort="high")
add("tools_parallel_off", request(tool_history, tools=tools, thinking=False), mode="chat")
# No schemas: completed-turn reasoning drops, current tool-turn reasoning stays.
add("tool_turn_current", request(tool_history[:5], thinking=True))
add("tool_turn_completed", request(tool_history, thinking=True))
add("mid_system", request(history[:3] + [{"role": "system", "content": "New instruction: answer briefly."}], thinking=False))
add("mid_system_on", request(history[:3] + [{"role": "system", "content": "New instruction: answer briefly."}], thinking=True))

pi_tools = tools + [function("read_file", {"path": {"type": "string"}}),
                    function("run", {"command": {"type": "string"}}),
                    function("edit_file", {"path": {"type": "string"}, "text": {"type": "string"}})]
pi_messages = [{"role": "system", "content": (
    "You are a coding assistant working in a local Rust repository. Read before editing. "
    "Preserve user changes, explain concrete findings, and verify your final changes. "
    "Code comments are English; documentation is Russian. Never invent tool results.\n" * 24)}]
for turn in range(4):
    ident = f"pi_{turn}"
    pi_messages.extend([
        {"role": "user", "content": f"Turn {turn + 1}: inspect src/module_{turn}.rs and explain the next change."},
        {"role": "assistant", "content": "", "reasoning_content": f"Inspect module {turn} before deciding.",
         "tool_calls": [call(ident, "read_file", {"path": f"src/module_{turn}.rs"})]},
        {"role": "tool", "tool_call_id": ident, "content": f"pub fn value_{turn}() -> u32 {{ {turn} }}"},
        {"role": "assistant", "content": f"Module {turn} returns {turn}.", "reasoning_content": "Verified the file."},
    ])
pi_messages.append({"role": "user", "content": "Now implement and verify the combined change."})
add("pi_like", request(pi_messages, tools=pi_tools, thinking={"type": "enabled"}, reasoning_effort="high"), effort="high")

write_json("cases.json", {"reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": cases})
print(f"Generated {len(cases)} cases; reference encoding.py SHA-256: {hashlib.sha256(source.read_bytes()).hexdigest()}")
