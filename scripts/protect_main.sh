#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/protect_main.sh ORG REPO [check_name]
# check_name defaults to "CI / build-test" (from .github/workflows/ci.yml)

ORG="${1:-jhaapasa}"
REPO="${2:-PortfolioAdvisor}"
CHECK_NAME="${3:-CI / build-test}"

echo "Applying branch protection to ${ORG}/${REPO}@main with required check: ${CHECK_NAME}" >&2

# Require signatures (optional step can fail if already enabled)
gh api -X POST \
  -H "Accept: application/vnd.github+json" \
  "repos/${ORG}/${REPO}/branches/main/protection/required_signatures" \
  || true

# Apply branch protection
gh api \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${ORG}/${REPO}/branches/main/protection" \
  -f required_linear_history=true \
  -f allow_force_pushes=false \
  -f allow_deletions=false \
  -f enforce_admins=true \
  -f required_pull_request_reviews='{"dismiss_stale_reviews":true,"require_code_owner_reviews":true,"required_approving_review_count":1,"require_last_push_approval":false}' \
  -f required_status_checks='{"strict":true,"contexts":["'"${CHECK_NAME}"'"]}' \
  -f restrictions='null'

echo "Branch protection applied." >&2


