# Hotfix notes: model refresh

OpenAI shut down `gpt-4-turbo` and `gpt-3.5-turbo`, and `gpt-4o-mini` is next.
Every live OpenAI chat model in this repo now uses **`gpt-6-luna`**.
Embedding models (`text-embedding-3-small`) are unchanged.

## The one rule you need

`gpt-6-luna` is a reasoning model. On Chat Completions it rejects `temperature`,
`top_p`, `max_tokens` and function tools **unless** you pass `reasoning_effort="none"`.
So wherever the code sets a temperature or uses tools, you'll see `reasoning_effort="none"` next to the model.

## What the video shows vs. what the code does now

| File | Video shows | Code now |
|---|---|---|
| `langmem-agents/coach.py` | `ChatOpenAI(model="gpt-4-turbo", temperature=0.2)` | `ChatOpenAI(model="gpt-6-luna", temperature=0.2, reasoning_effort="none")` |
| `langmem-agents/simple_mem_agent.py` | `create_react_agent("openai:gpt-4o-mini", ...)` | `llm = init_chat_model("openai:gpt-6-luna", reasoning_effort="none")`, then `create_react_agent(llm, ...)`. The string form can't carry `reasoning_effort`, and tools need it. |
| `langmem-agents/recipe_assis.agent.py` | `init_chat_model("openai:gpt-4o-mini", temperature=0.2)` and `create_memory_store_manager("openai:gpt-4o-mini", ...)` | Same `llm` with `gpt-6-luna` + `reasoning_effort="none"`. The memory manager now receives that `llm` object instead of a string. |
| `agents_with_mem/customer_supp.py` | `init_chat_model("openai:gpt-4o-mini")` and `create_react_agent("openai:gpt-4o-mini", ...)` | One `llm` (`gpt-6-luna`, `reasoning_effort="none"`) used by the router and passed to `create_react_agent(llm, ...)` |
| `mem0/coach_mem0.py` | `ChatOpenAI(model="gpt-4o-mini", temperature=0.2)`; `search(query, user_id=..., output_format="v1.1", version="v2")` | `gpt-6-luna` + `reasoning_effort="none"`; `search(query, filters={"user_id": ...})`, which reads `results` from the mem0 2.x response |
| `mem0/simple_agent_local.py` | `Memory()` with mem0's default LLM; `model="gpt-4o-mini"` | `Memory.from_config(config)` pins mem0's internal LLM to `gpt-6-luna`; chat call uses `gpt-6-luna` |
| `mem0/mem0_supabase_simple.py` | `MODEL_CHOICE` default `gpt-4o-mini`; chat `gpt-4o-mini` | default `gpt-6-luna`; chat `gpt-6-luna` |
| `mem0/mem0_supabase.py` | fallback `gpt-3.5-turbo` | fallback `gpt-6-luna` |
| `mem0/final_coach_mem0_supa.py` | `MODEL_CHOICE` default `""` | default `gpt-6-luna` |
| `mem0/coach_autogen_mem0.py` | mem0 + AutoGen on `gpt-4o-mini` | both on `gpt-6-luna` |
| `raw_agent_with_memory/agent_with_memory.py` | `gpt-4o-mini`, `temperature=0.7, max_tokens=1000` | `gpt-6-luna`, `temperature=0.7, max_completion_tokens=1000, reasoning_effort="none"` |
| `.env.example` | `MODEL_CHOICE=gpt-4o-mini` | `MODEL_CHOICE=gpt-6-luna`. **Update your own `.env` too.** |
| `bedrock/cli_commands.sh` | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` with `temperature`/`top_p` | `us.anthropic.claude-sonnet-5-5` (US cross-Region profile, works from us-west-2). `temperature`/`top_p` removed, `max_tokens` 1024 |

## mem0 changes (all mem0 lessons)

- **Every mem0 LLM config** now has `"is_reasoning_model": True, "reasoning_effort": "none"`.
  mem0 doesn't recognize `gpt-6-luna` as a reasoning model, so without these lines it sends
  `temperature`, `top_p` and `max_tokens`, and OpenAI returns a 400.
  If you switch `MODEL_CHOICE` back to a non-reasoning model (e.g. `gpt-4.1`), remove both lines.
- **mem0 2.x search API:** `memory.search(query, user_id=...)` now raises
  `Top-level entity parameters ... not supported`. Use `filters={"user_id": ...}`.
  `memory.add(..., user_id=...)` is unchanged. Updated in every mem0 file, including the
  cloud `AsyncMemoryClient` lessons (`coach_mem0.py`, `simple_agent_api.py`).
- mem0's own default LLM (when you don't pass one) is now `gpt-5-mini` (mem0 2.2.1).
  Older mem0 used `gpt-4o-mini`. The lessons now set the model explicitly, so this no longer matters.
- mem0 prints "spaCy is not installed" / "fastembed not installed" notices. They're harmless.

## requirements.txt

- `langgraph==0.3.21` → `0.3.23`. `langmem==0.0.19` requires `>=0.3.23`, so the old file could not install.
- `pyautogen` → `ag2[openai]==0.13.0`. `pyautogen` is now a proxy for the new `autogen-agentchat`
  and no longer provides `import autogen` / `ConversableAgent`. ag2 0.13 keeps the classic API.
  (ag2 1.x is a rewrite with a different package and pulls in `openai` 3.x. Avoid it here.)
- `mem0ai` pinned to `2.2.1`, the version tested here (it supports `is_reasoning_model`).

## Not changed

- `bedrock/converse.py`, `bedrock/invoke.py` still use `amazon.titan-text-express-v1`.
  Status unverified. Note: it no longer appears on Bedrock's "Models at a glance" page, which lists only Titan *embeddings* models.
- `adk-test/` uses `gemini-2.0-flash-exp` (not OpenAI, out of scope here).
- Commented-out alternative models were left as they were.
