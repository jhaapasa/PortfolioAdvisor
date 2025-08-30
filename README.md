# PortfolioAdvisor

LangGraph-based, function-defined agents to analyze a portfolio and emit a minimal report.

## Quickstart

1. Ensure Python 3.13.7 is available (e.g., via pyenv) and clone the repo.
2. Create and activate a virtual environment and install deps:

   ```bash
   ./scripts/bootstrap
   source .venv/bin/activate
   ```

3. Set up environment variables for Gemini (optional for stub run):

   - Copy `.env.example` to `.env` and fill in values.

4. Prepare an input directory with one or more files (e.g., `positions.csv`).

5. Run the CLI:

   ```bash
   portfolio-advisor --input-dir ./inputs --output-dir ./outputs --log-level INFO
   ```

The command prints the path to `analysis.md` in the output directory.

## Configuration

Configuration is loaded from environment (and `.env` in development) and can be overridden on the CLI.

- Required at runtime: `--input-dir`, `--output-dir`
- Gemini (optional for stub): `GEMINI_API_KEY`, `GEMINI_API_BASE`, `GEMINI_MODEL`
- Tuning: `REQUEST_TIMEOUT_S`, `MAX_TOKENS`, `TEMPERATURE`
- Logging: `LOG_LEVEL`, `LOG_FORMAT` (plain|json)

CLI overrides example:

```bash
portfolio-advisor \
  --input-dir ./inputs \
  --output-dir ./outputs \
  --gemini-model gemini-1.5-pro \
  --temperature 0.2 \
  --log-format json
```

## Development

- Lint: `./scripts/lint`
- Format: `./scripts/format`
- Test: `./scripts/test`

## Logging and Error Handling

- Logging defaults to INFO, plain format; switch to JSON via `LOG_FORMAT=json` or `--log-format json`.
- Sensitive values (like API keys) are never logged.
- Clear exceptions are raised for invalid inputs (missing directories, write failures) and configuration errors.

