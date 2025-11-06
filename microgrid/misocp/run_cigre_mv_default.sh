#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="$ROOT" \
python -m run \
  --scenario "$ROOT/misocp/scenarios/cigre_mv_default.yaml" \
  --save "$ROOT/misocp/cigre_mv_misocp.pkl"
