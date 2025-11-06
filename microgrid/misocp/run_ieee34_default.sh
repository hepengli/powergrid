#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="$ROOT" \
python -m run \
  --scenario "$ROOT/misocp/scenarios/ieee34_default.yaml" \
  --save "$ROOT/misocp/ieee34_misocp.pkl"
