# AgentPM Python SDK

A lean, typed Python SDK for **AgentPM** tools. It discovers tools installed by `agentpm install`, executes their entrypoints in a subprocess, and returns JSON results you can pass to your agents.

- 🔎 **Discovers** tools in `.agentpm/tools` (project) and `~/.agentpm/tools` (user), with `AGENTPM_TOOL_DIR` override.
- 🚀 **Runs entrypoints** via `node` or `python` (whitelisted) and exchanges JSON over stdin/stdout.
- 🧩 **Metadata-aware**: `with_meta=True` returns `func + meta` (name, version, description, inputs, outputs).
- 🧪 **Framework adapters (optional)**: e.g., a LangChain adapter you can use if installed.

> Requires Python **3.10+**.

---

## Installation

### From PyPI (recommended)

Using **uv**:
```bash
uv pip install agentpm
```

Or with standard pip:
```bash
python -m pip install agentpm
```

If you'll use the optional LangChain adapter:
```bash
uv pip install 'agentpm[langchain]'
# or
python -m pip install 'agentpm[langchain]'
```
---

## Quick Start (with `uv`)

```bash
# create and activate a venv
uv venv
source .venv/bin/activate

# install SDK in editable dev mode (ruff/black/mypy/pytest, etc.)
uv pip install -e ".[dev]"

# sanity checks
uv run ruff check .
uv run black --check .
uv run mypy
uv run pytest -q
```

> If you're not using `uv`, standard `python -m venv` + `pip install -e ".[dev]"` works too.

---

## Using the SDK

```python
from agentpm import load

# Spec format: "@scope/name@version"
summarize = load("@zack/summarize@0.1.0")

result = summarize({"text": "Long document content..."})
print(result["summary"])
```

### With metadata (build richer tool descriptions)
```python
from agentpm import load

tool = load("@zack/summarize@0.1.0", with_meta=True)
summarize, meta = tool["func"], tool["meta"]

rich_description = (
    f"{meta.get('description','')} "
    f"Inputs: {meta.get('inputs')}. "
    f"Outputs: {meta.get('outputs')}."
)

print(rich_description)
print(summarize({"text": "hello"})["summary"])
```

### Optional: LangChain adapter
The adapter is lazy-imported and only needed if you call it.

```python
from agentpm import load, to_langchain_tool  # to_langchain_tool is loaded on first access

loaded = load("@zack/summarize@0.1.0", with_meta=True)
tool = to_langchain_tool(loaded)  # requires `langchain-core` installed
```

If you use the adapter, install LangChain core:

```bash
uv pip install langchain-core
```

---

## Where tools are discovered

Resolution order:

1. `AGENTPM_TOOL_DIR` (environment variable)
2. `./.agentpm/tools` (project-local)
3. `~/.agentpm/tools` (user-local)

Each tool lives in a directory like:

```
.agentpm/
  tools/
    @zack/summarize@0.1.0/
      agent.json
      (tool files…)
```

---

## Manifest & Runtime Contract

**`agent.json` (minimal fields used by the SDK):**
```json
{
  "name": "@zack/summarize",
  "version": "0.1.0",
  "description": "Summarize long text.",
  "inputs": {
    "type": "object",
    "properties": { "text": { "type": "string", "description": "Text to summarize" } },
    "required": ["text"]
  },
  "outputs": {
    "type": "object",
    "properties": { "summary": { "type": "string", "description": "Summarized text" } },
    "required": ["summary"]
  },
  "entrypoint": {
    "command": "python",
    "args": ["main.py"],
    "cwd": ".",
    "timeout_ms": 60000,
    "env": {}
  }
}
```

**Execution contract:**
- SDK writes **inputs JSON** to the process **stdin**.
- Tool writes a single **outputs JSON** object to **stdout**.
- Non-JSON logs should go to **stderr**.
- Process must exit with **code 0** on success.

**Interpreter whitelist:** `node`, `nodejs`, `python`, `python3`.
The SDK validates the interpreter and checks it’s present on `PATH`.

---

## Development

### Project layout
```
src/
  agentpm/
    __init__.py           # re-exports: load, to_langchain_tool (lazy)
    core.py               # resolver/spawn/JSON plumbing
    types.py              # JsonValue, TypedDicts
    adapters/
      __init__.py
      langchain.py        # optional adapter
    py.typed              # marks package as typed
tests/
  test_basic.py
```

### Common tasks (via `uv`)
```bash
uv run ruff check .
uv run black --check .
uv run mypy
uv run pytest -q

# run hooks locally on all files
uv run pre-commit run --all-files
```

---

## Building & Publishing

```bash
# build wheel & sdist
uv run python -m build

# verify metadata
uv run twine check dist/*

# upload (PyPI)
uv run twine upload dist/*

# or TestPyPI first
uv run twine upload -r testpypi dist/*
```

---

## Troubleshooting

- **`No JSON object found on stdout.`**
  Ensure your tool prints a single JSON object as the last thing on stdout, and writes logs to stderr.

- **`Unsupported agent.json.entrypoint.command`**
  Only `node` / `python` are allowed (including `nodejs` / `python3`). Update `entrypoint.command`.

- **`Interpreter "... " not found on PATH`**
  Install the interpreter or adjust `entrypoint.command`. The SDK runs `<command> --version` to verify availability.

- **PEP 668 / “externally managed”**
  Use a venv (we recommend `uv venv`) and install with `uv pip install -e ".[dev]"`.

- **IDE can’t import `agentpm`**
  Ensure your interpreter is the project’s `.venv/bin/python`, and that you ran the editable install.

---

## License

MIT — see `LICENSE`.
