# agentY-core

The repository is `agentY-core`; the Python package in it is `agenty_core`.

Shared, framework-agnostic tool/util layer for the two agentY apps:

- **agentY** — the Strands multi-agent app (`../agentY`)
- **agentY-mcp** — the MCP server (`../agentY-mcp`)

It is the single source of truth for the ComfyUI workflow tools, HuggingFace
model management, web search, file I/O, and the headless batch worker — so that
tool logic is maintained once instead of in two copies.

## What lives here

```
agenty_core/
  paths.py            app-root indirection (set_project_root / project_root)
  _compat.py          no-op @tool decorator (tools stay plain callables)
  comfyui_exec.py     ComfyUI submit/poll/resolve primitives (batch worker)
  batch_runner.py     detached batch worker (python -m agenty_core.batch_runner)
  tools/              comfyui, huggingface, file_tools, shell, web_search,
                      batch, workflow_registry
  utils/              comfyui_client, comfyui_progress, model_node_mapping,
                      progress_signal, secrets, video_frames, workflow_parser, …
```

The tool functions are **plain callables** (the `@tool` here is a no-op). Each
consumer adapts them:

- **agentY-mcp** registers them with FastMCP (`mcp.add_tool`).
- **agentY** wraps them with `strands.tool` in its `src/tools/__init__.py`.

## What does NOT live here (per-app carve-outs)

These diverge by host and stay in each repo:

- `image_handling` — MCP returns `MCPImage`; agentY returns Strands content
  blocks and routes a vision sub-agent.
- `memory` / `memory_tools` — same API, different backend (file store vs
  FAISS+mem0+Ollama).
- `executor` — agentY weaves in an Ollama Vision-QA pass.
- `signal_workflow_ready` — agentY pipeline handoff (agentY only).
- `execution.py` — MCP-only blocking executor.

## App-root resolution

Because this package is a *sibling* of each app, it can't resolve `config/` or
output dirs from its own `__file__`. Each app pins its root at startup via
`agenty_core.set_project_root(<repo root>)` (or the `AGENTY_PROJECT_ROOT` env
var; falls back to CWD). See `paths.py`.

## Install (editable)

From either app's repo root:

```
uv pip install -e ../agentY-core
```

Both apps' `requirements.txt` already include `-e ../agentY-core`.
