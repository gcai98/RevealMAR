# RevealMAR project rules

## Commands
- `python -m py_compile main_revealmar.py models/revealmar.py`: quick syntax check
- Prefer small smoke checks over long training runs
- Reuse existing baseline commands and argument names where possible

## Workflow
- Always inspect relevant files and propose a file-by-file plan before nontrivial edits
- Keep baseline MAR behavior unchanged by default
- Prefer additive changes over destructive edits
- Keep diffs minimal
- Avoid unrelated refactors
- After a series of edits, run at least a syntax or smoke check
- Use a new conversation for a new implementation phase

## Architecture constraints
- Separate RevealMAR entrypoint: `main_revealmar.py`
- Separate RevealMAR model file: `models/revealmar.py`
- Reuse `engine_mar.py` in early phases unless a separate engine is truly necessary
- Do not silently change `main_mar.py`, `engine_mar.py`, or `models/mar.py`
- Do not reuse stale experimental leftovers:
  - `models/difficulty.py`
  - `models/verifier.py`
  - `models/rerank.py`
  - `models/aura_mar.py`
  - `util/aura_utils.py`
  - `main_aura_mar.py`
  - `engine_aura_mar.py`

## Code style
- Preserve existing project style
- Add concise comments only when tensor flow or design intent would otherwise be unclear
- Do not invent missing APIs or files
- If uncertain, explain assumptions before editing