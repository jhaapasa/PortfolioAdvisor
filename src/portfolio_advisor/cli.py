from __future__ import annotations

import argparse
import sys

from .analyze import analyze_portfolio


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PortfolioAdvisor CLI")
    p.add_argument("--input-dir", required=True, help="Path to input directory")
    p.add_argument("--output-dir", required=True, help="Path to output directory")

    # Env overrides
    p.add_argument("--gemini-api-key")
    p.add_argument("--gemini-api-base")
    p.add_argument("--gemini-model")
    p.add_argument("--request-timeout-s", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)

    # Logging
    p.add_argument("--log-level", default=None, help="Logging level (e.g., INFO, DEBUG)")
    p.add_argument("--log-format", default=None, choices=["plain", "json"], help="Log format")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    input_dir = overrides.pop("input_dir")
    output_dir = overrides.pop("output_dir")

    try:
        output_path = analyze_portfolio(input_dir=input_dir, output_dir=output_dir, **overrides)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
