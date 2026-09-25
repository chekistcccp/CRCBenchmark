#!/usr/bin/env bash
set -euo pipefail

echo "[DEPRECATED] run_benchmark.sh is kept only for compatibility."
echo "[DEPRECATED] The canonical experiment entry point is now: bash run.sh"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.sh" "$@"
