#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

output_path="${1:-}"
if [[ -z "$output_path" ]]; then
  echo "usage: $0 /path/to/pytorch_goldens.json" >&2
  exit 1
fi

PYTHONPATH=. python \
  examples/generate_pytorch_goldens.py \
  --output "$output_path"
