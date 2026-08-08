# AgentPM™ Python SDK

A lean, typed Python SDK for **AgentPM** tools and installed agent, Knowledge, Memory, and Profile packages. It discovers tools installed by `agentpm install`, executes their entrypoints in a subprocess, and can also inspect installed agent manifests plus their resolved dependency refs.

- 🔎 **Discovers** tools in `.agentpm/tools` (project) and `~/.agentpm/tools` (user), with `AGENTPM_TOOL_DIR` override.
- 📦 **Loads installed agents** from `.agentpm/agents` and exposes their resolved tool and skill refs from `agent.lock`.
- 📚 **Loads installed skills** from `.agentpm/skills` and exposes their manual content plus resolved tool refs.
- 🧠 **Loads installed Knowledge packages** from `.agentpm/knowledge` and exposes mode-specific metadata and canonical paths.
- ♾️ **Loads installed Memory packages** from `.agentpm/memory` and exposes authored blueprint metadata, build metadata, contract indexes, and resolved contract paths.
- 🎭 **Loads installed Profile packages** from `.agentpm/profiles` and exposes authored role, objective, and communication metadata.
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
from agentpm import load, load_agent, load_knowledge, load_memory, load_profile, load_skill

agent = load_agent("@zack/support-agent@0.1.0")
docs = load_knowledge("@zack/python-docs@0.1.0")
memory = load_memory("@zack/profile-memory@0.1.0")
profile = load_profile("@zack/support-style@0.1.0")
first_skill = agent["resolvedSkills"][0]
skill = load_skill(f'{first_skill["name"]}@{first_skill["version"]}')
first_tool = skill["resolvedTools"][0]
tool = load(f'{first_tool["name"]}@{first_tool["version"]}')

print(agent["resolvedKnowledge"])
print(agent["resolvedMemory"])
print(agent["resolvedProfiles"])
print(docs["knowledge"]["mode"])
print(memory["contracts"])
print(profile["profile"]["communication"])
```

`load_agent()` returns:

- the installed agent manifest
- the installed agent root path
- `resolvedKnowledge` from `agent.lock`
- `resolvedMemory` from `agent.lock`
- `resolvedProfiles` from `agent.lock`
- reserved refs (`knowledge`, `memory`, `profiles`) as metadata
- `resolvedTools` from `agent.lock`
- `resolvedSkills` from `agent.lock`

It does **not** execute the agent package or orchestrate the tools for you.

Compatibility note:

- `resolvedKnowledge` is populated from the modern first-class `root.knowledge` entries in `agent.lock`.
- `reserved.knowledge` is legacy pass-through metadata from older lockfile shapes. For current installs, treat `resolvedKnowledge` as the authoritative Knowledge dependency list and expect `reserved.knowledge` to usually be empty.
- If your workspace still has an older pre-Knowledge lockfile shape where Knowledge refs only exist under `reserved.knowledge`, rerun `agentpm install` to rewrite the lockfile before expecting `resolvedKnowledge` to be populated.

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

### Load an installed Knowledge package

```python
from agentpm import load_knowledge

knowledge = load_knowledge("@zack/python-docs@0.1.0")

print(knowledge["knowledge"]["mode"])
print(knowledge["documentPaths"])
print(knowledge["chunksPath"])
print(knowledge["sourcesPath"])
print(knowledge["vectorsPath"])
print(knowledge["indexPaths"])
```

`load_knowledge()` returns an inspectable Knowledge object with:

- the installed knowledge manifest
- the installed package root path
- parsed `knowledge` metadata
- absolute paths for declared context documents, chunks, sources, vectors, indexes, and provenance when present

### Load an installed Memory package

```python
from agentpm import load_memory, load_memory_contract

memory = load_memory("@zack/profile-memory@0.1.0")
profile_contract = load_memory_contract(
    memory,
    space="profile",
    record_type="user_preference",
)

print(memory["memory"]["spaces"])
print(memory["build"])
print(memory["contractIndex"])
print(memory["contracts"])
print(profile_contract)
```

`load_memory()` returns an inspectable Memory Blueprint object with:

- the installed memory manifest
- the installed package root path
- parsed `memory` metadata
- parsed `memory/build.json`
- parsed `memory/contracts/index.json`
- absolute paths for declared source schemas and indexed resolved contracts

It is a metadata and contract loader only. It does not provide live record CRUD, retention enforcement, trigger execution, or a hosted memory runtime.

`load_memory_contract()` loads one indexed resolved contract on demand by `space` + `record_type`.

### Load an installed Profile package

```python
from agentpm import load_profile

profile = load_profile("@zack/support-style@0.1.0")

print(profile["profile"]["identity"]["role"])
print(profile["profile"]["objectives"])
print(profile["profile"]["communication"])
```

`load_profile()` returns an inspectable Instruction Profile object with:

- the installed profile manifest
- the installed package root path
- parsed authored `profile` metadata

### `load()` stays tool-only

```python
from agentpm import load

load("@zack/triage-playbook@0.1.0")
# raises: use load_skill("@zack/triage-playbook@0.1.0") instead

load("@zack/python-docs@0.1.0")
# raises: use load_knowledge("@zack/python-docs@0.1.0") instead

load("@zack/profile-memory@0.1.0")
# raises: use load_memory("@zack/profile-memory@0.1.0") instead

load("@zack/support-style@0.1.0")
# raises: use load_profile("@zack/support-style@0.1.0") instead
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

Installed registry Knowledge packages live separately:

```
.agentpm/
  knowledge/
    @zack/python-docs/
      0.1.0/
        agent.json
        knowledge/
```

Installed registry Memory packages live separately:

```
.agentpm/
  memory/
    @zack/profile-memory/
      0.1.0/
        agent.json
        schemas/
        memory/
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

## Where installed Knowledge packages are discovered

Resolution order for `load_knowledge()`:

1. `AGENTPM_KNOWLEDGE_DIR` (environment variable)
2. `./.agentpm/knowledge` (project-local)
3. `~/.agentpm/knowledge` (user-local)

You can also override per call:

```python
load_knowledge("@zack/python-docs@0.1.0", knowledge_dir_override="/path/to/knowledge")
```

## Where installed Memory packages are discovered

Resolution order for `load_memory()`:

1. `AGENTPM_MEMORY_DIR` (environment variable)
2. `./.agentpm/memory` (project-local)
3. `~/.agentpm/memory` (user-local)

You can also override per call:

```python
load_memory("@zack/profile-memory@0.1.0", memory_dir_override="/path/to/memory")
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
    __init__.py           # re-exports: load, load_agent, load_knowledge, load_memory, load_skill, to_langchain_tool (lazy)
    core.py               # resolver/spawn/JSON plumbing
    types.py              # JsonValue, TypedDicts
    adapters/
      __init__.py
      langchain.py        # optional adapter
    py.typed              # marks package as typed
tests/
  test_basic.py
  test_load_agent.py
  test_load_memory.py
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
