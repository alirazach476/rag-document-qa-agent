# Contributing

Thanks for improving this project. Keep changes focused and easy to review.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your OPENAI_API_KEY for local runs
```

## Checks before opening a PR

```bash
ruff check .
pytest -q
```

## Guidelines

- Prefer small PRs that do one thing well.
- Do not commit `.env`, API keys, or regenerated `vectorstore/` indexes.
- Add or update tests when changing ingest, config, or answer formatting.
- Match existing naming and module boundaries under `rag_qa/`.
