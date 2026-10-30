#!/bin/bash
# Reads both API keys from 1Password, then runs the arena. Keys are never printed.
cd "$(dirname "$0")" || exit 1

XAI_API_KEY="$(op read "op://Personal/xAI API Key/credential")" || {
  echo "OP_READ_FAILED: could not read the xAI API Key item from 1Password. Run: eval \$(op signin)" >&2
  exit 1
}
GOOGLE_API_KEY="$(op read "op://Personal/Google API Key/credential")" || {
  echo "OP_READ_FAILED: could not read the Google API Key item from 1Password. Run: eval \$(op signin)" >&2
  exit 1
}
export XAI_API_KEY GOOGLE_API_KEY

exec .venv/bin/python arena.py "$@"
