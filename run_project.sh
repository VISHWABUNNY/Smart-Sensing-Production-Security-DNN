#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=python3
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: Python is not installed. Please install Python 3 and retry." >&2
  exit 1
fi

BACKEND_VENV_DIR="$ROOT/backend/.venv"
FRONTEND_VENV_DIR="$ROOT/frontend/.venv"
BACKEND_VENV_PYTHON="$BACKEND_VENV_DIR/bin/python"
FRONTEND_VENV_PYTHON="$FRONTEND_VENV_DIR/bin/python"
BACKEND_VENV_ACTIVATE="$BACKEND_VENV_DIR/bin/activate"
FRONTEND_VENV_ACTIVATE="$FRONTEND_VENV_DIR/bin/activate"

create_and_check_venv() {
  local venv_dir="$1"
  local venv_python="$2"

  if [[ ! -d "$venv_dir" ]]; then
    echo "Creating virtual environment in $venv_dir..."
    "$PYTHON" -m venv "$venv_dir"
  fi

  if [[ ! -x "$venv_python" ]]; then
    echo "Error: virtual environment python not found: $venv_python" >&2
    exit 1
  fi
}

create_and_check_venv "$BACKEND_VENV_DIR" "$BACKEND_VENV_PYTHON"
create_and_check_venv "$FRONTEND_VENV_DIR" "$FRONTEND_VENV_PYTHON"

install_reqs() {
  local req_file="$1"
  local venv_activate="$2"

  if [[ ! -f "$req_file" ]]; then
    echo "Error: requirements file not found: $req_file" >&2
    exit 1
  fi

  echo "Activating virtual environment: $venv_activate"
  # shellcheck source=/dev/null
  source "$venv_activate"
  echo "Installing dependencies from $req_file..."
  python -m pip install --upgrade pip
  python -m pip install -r "$req_file"
  deactivate
}

install_reqs "$ROOT/backend/requirements.txt" "$BACKEND_VENV_ACTIVATE"
install_reqs "$ROOT/frontend/requirements.txt" "$FRONTEND_VENV_ACTIVATE"

echo "Launching the application from backend venv..."
# shellcheck source=/dev/null
source "$BACKEND_VENV_ACTIVATE"
cd "$ROOT"
python main.py
