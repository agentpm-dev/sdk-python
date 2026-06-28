# AgentPM™ Python SDK

A lean, typed Python SDK for **AgentPM** tools and installed agent packages. It discovers tools installed by `agentpm install`, executes their entrypoints in a subprocess, and can also inspect installed agent manifests plus their resolved tool refs.

- 🔎 **Discovers** tools in `.agentpm/tools` (project) and `~/.agentpm/tools` (user), with `AGENTPM_TOOL_DIR` override.
- 📦 **Loads installed agents** from `.agentpm/agents` and exposes their resolved tool and skill refs from `agent.lock`.
- 📚 **Loads installed skills** from `.agentpm/skills` and exposes their manual content plus resolved tool refs.
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

### Load an installed agent package

```python
from agentpm import load, load_agent, load_skill

agent = load_agent("@zack/support-agent@0.1.0")
first_skill = agent["resolvedSkills"][0]
skill = load_skill(f'{first_skill["name"]}@{first_skill["version"]}')
first_tool = skill["resolvedTools"][0]
tool = load(f'{first_tool["name"]}@{first_tool["version"]}')
```

`load_agent()` returns:

- the installed agent manifest
- the installed agent root path
- reserved refs (`knowledge`, `memory`, `profiles`) as metadata
- `resolvedTools` from `agent.lock`
- `resolvedSkills` from `agent.lock`

It does **not** execute the agent package or orchestrate the tools for you.

This is the Python mirror of the Node SDK’s `loadAgent()` flow:

1. load the installed agent package
2. read its resolved skill and tool refs
3. optionally load a resolved skill package
4. choose which tool packages to `load()`

### Load an installed skill package

```python
from agentpm import load_skill

skill = load_skill("@zack/triage-playbook@0.1.0")

print(skill["entrypointPath"])
print(skill["entrypointContent"])
print(skill["references"])
print(skill["scripts"])
print(skill["resolvedTools"])
```

`load_skill()` returns an inspectable Skill object. Skills are **not** runnable SDK objects.

### `load()` stays tool-only

```python
from agentpm import load

load("@zack/triage-playbook@0.1.0")
# raises: use load_skill("@zack/triage-playbook@0.1.0") instead
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
    @zack/summarize/
      0.1.0/
        agent.json
        (tool files…)
```

Installed registry agent packages live separately:

```
.agentpm/
  agents/
    @zack/support-agent/
      0.1.0/
        agent.json
        README.md
```

Installed registry skill packages live separately:

```
.agentpm/
  skills/
    @zack/triage-playbook/
      0.1.0/
        agent.json
        SKILL.md
```

## Where installed agents are discovered

Resolution order for `load_agent()`:

1. `AGENTPM_AGENT_DIR` (environment variable)
2. `./.agentpm/agents` (project-local)
3. `~/.agentpm/agents` (user-local)

You can also override per call:

```python
load_agent("@zack/support-agent@0.1.0", agent_dir_override="/path/to/agents")
```

## Where installed skills are discovered

Resolution order for `load_skill()`:

1. `AGENTPM_SKILL_DIR` (environment variable)
2. `./.agentpm/skills` (project-local)
3. `~/.agentpm/skills` (user-local)

You can also override per call:

```python
load_skill("@zack/triage-playbook@0.1.0", skill_dir_override="/path/to/skills")
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
    __init__.py           # re-exports: load, load_agent, load_skill, to_langchain_tool (lazy)
    core.py               # resolver/spawn/JSON plumbing
    types.py              # JsonValue, TypedDicts
    adapters/
      __init__.py
      langchain.py        # optional adapter
    py.typed              # marks package as typed
tests/
  test_basic.py
  test_load_agent.py
  test_load_skill.py
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

## Running mixed-runtime Agent apps with Docker

Some AgentPM tools run on Node, some on Python—and your agent may need to spawn both. Using Docker gives you a single, reproducible environment where both interpreters are installed and on PATH, which avoids the common “interpreter not found” issues that pop up on PaaS/CI or IDEs.

Why Docker?

✅ Hermetic: Python + Node versions are pinned inside the image.

✅ No PATH drama: node/python are present and discoverable.

✅ Prod/CI parity: the same image runs on your laptop, CI, and servers.

✅ Easy secrets: pass API keys via env at docker run/Compose time.

✅ Fewer surprises: consistent OS libs for LLM clients, SSL, etc.

### When to use it

- You deploy to platforms that don’t let you apt-get both runtimes.
- Your agent uses tools with different interpreters (Node + Python).
- Your local dev/IDE PATH differs from production and causes failures.
- You want reproducible builds and easy rollback.

### How to use it

1. Copy the provided [Dockerfile](https://github.com/agentpm-dev/sdk-python/tree/main/examples/python-agent) into your repo.
2. (Optional) Pre-install tools locally with agentpm install ... and commit or copy .agentpm/tools/ into the image, or run agentpm install at build time if your CLI is available in the image.
3. Build & run:

```bash
docker build -t agent-app .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY agent-app
```

4. For development, use the docker-compose.yml snippet to mount your source and pass env vars conveniently.

### Troubleshooting

- Set `AGENTPM_DEBUG=1` to print the SDK’s project root, search paths, merged PATH, and resolved interpreters.
- You can force interpreters via:
```ini
AGENTPM_NODE=/usr/bin/node
AGENTPM_PYTHON=/usr/local/bin/python3.11
```

- Prefer absolute interpreters in agent.json.entrypoint.command for production (e.g., /usr/bin/node). The SDKs still enforce the Node/Python family.

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
