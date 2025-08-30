#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/protect_main.sh ORG REPO [check_name]
# check_name defaults to "CI / build-test" (from .github/workflows/ci.yml)

ORG="${1:-jhaapasa}"
REPO="${2:-PortfolioAdvisor}"
CHECK_NAME="${3:-CI / build-test}"

echo "Applying branch protection to ${ORG}/${REPO}@main with required check: ${CHECK_NAME}" >&2

read -r -d '' BODY <<JSON || true
{
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "required_status_checks": {
    "strict": true,
    "contexts": ["${CHECK_NAME}"]
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

echo "$BODY" | gh api -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  --input - \
  "repos/${ORG}/${REPO}/branches/main/protection"

# Enable signature protection (optional)
gh api -X POST \
  -H "Accept: application/vnd.github+json" \
  "repos/${ORG}/${REPO}/branches/main/protection/required_signatures" \
  || true

echo "Branch protection applied." >&2


