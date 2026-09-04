#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

command -v rsvg-convert >/dev/null 2>&1 || {
  echo "rsvg-convert is required" >&2
  exit 2
}

rsvg-convert --width=2400 --output=sota_venn.png sota_venn.svg
rsvg-convert --background-color=white --width=2621 --output=conops.png conops_redraw.svg
