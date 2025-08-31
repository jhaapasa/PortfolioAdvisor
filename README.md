# PortfolioAdvisor

LangGraph-based, function-defined agents to analyze a portfolio and emit a minimal report.

## Quickstart

1. Ensure Python 3.13.7 is available (e.g., via pyenv) and clone the repo.
2. Create and activate a virtual environment and install deps:

   ```bash
   ./scripts/bootstrap
   source .venv/bin/activate
   ```

3. Set up environment variables for Gemini and parser (optional for stub run):

   - Copy `env.example` to `.env` and fill in values.

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
- Parser (LLM Parsing Agent):
  - `PARSER_MAX_CONCURRENCY` (default 4)
  - `PARSER_MAX_RPM` (default 60)
  - `PARSER_MAX_RETRIES` (default 2)
  - `PARSER_MAX_DOC_CHARS` (default 20000)

CLI overrides example:

```bash
portfolio-advisor \
  --input-dir ./inputs \
  --output-dir ./outputs \
  --gemini-model gemini-1.5-pro \
  --temperature 0.2 \
  --log-format json
```

### Parser behavior

- The ingestion node extracts plain text from inputs.
- The parser node fans out with per-document LLM calls, bounded by `PARSER_MAX_CONCURRENCY` and `PARSER_MAX_RPM`.
- Each call asks the model to emit JSON matching the schema for candidate holdings and retries up to `PARSER_MAX_RETRIES` on validation errors.
- All candidates are concatenated and summarized by the analyst node.

## CLI parameters

The `portfolio-advisor` CLI accepts the following parameters. All options can also be provided via environment variables by passing them through to the app entrypoint (see `env.example`).

- `--input-dir` (required)
  - Directory containing input files to analyze.
  - No environment variable; must be provided via CLI.

- `--output-dir` (required)
  - Directory where outputs (e.g., `analysis.md`) will be written. Created if it does not exist.
  - No environment variable; must be provided via CLI.

- `--gemini-api-key`
  - API key for Google Gemini.
  - Env: `GEMINI_API_KEY`
  - Default: empty (a stub LLM is used and returns placeholder text).

- `--gemini-api-base`
  - Optional custom base URL for Gemini API.
  - Env: `GEMINI_API_BASE`
  - Default: empty (use library default).

- `--gemini-model`
  - Gemini model name.
  - Env: `GEMINI_MODEL`
  - Default: `gemini-1.5-pro`.

- `--request-timeout-s`
  - Request timeout in seconds used by LLM calls.
  - Env: `REQUEST_TIMEOUT_S`
  - Default: `60`.

- `--max-tokens`
  - Maximum output tokens requested from the LLM. If unset, the model default is used.
  - Env: `MAX_TOKENS`
  - Default: empty/unset.

- `--temperature`
  - Sampling temperature for the LLM.
  - Env: `TEMPERATURE`
  - Default: `0.2`.

- `--log-level`
  - Logging level (e.g., `DEBUG`, `INFO`, `WARNING`).
  - Env: `LOG_LEVEL`
  - Default: `INFO`.

- `--log-format`
  - Logging format: `plain` or `json`.
  - Env: `LOG_FORMAT`
  - Default: `plain`.

### Parser-related CLI flags

The following flags override the corresponding environment variables:

- `--parser-max-concurrency` → `PARSER_MAX_CONCURRENCY` (default 4)
- `--parser-max-rpm` → `PARSER_MAX_RPM` (default 60)
- `--parser-max-retries` → `PARSER_MAX_RETRIES` (default 2)
- `--parser-max-doc-chars` → `PARSER_MAX_DOC_CHARS` (default 20000)

You can combine them with other CLI options, for example:

```bash
portfolio-advisor \
  --input-dir ./inputs \
  --output-dir ./outputs \
  --parser-max-concurrency 8 \
  --parser-max-rpm 120 \
  --parser-max-retries 3
```

## Development

- Lint: `./scripts/lint`
- Format: `./scripts/format`
- Test: `./scripts/test`

## Logging and Error Handling

- Logging defaults to INFO, plain format; switch to JSON via `LOG_FORMAT=json` or `--log-format json`.
- Sensitive values (like API keys) are never logged.
- Clear exceptions are raised for invalid inputs (missing directories, write failures) and configuration errors.

