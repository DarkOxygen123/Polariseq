#!/usr/bin/env bash
# Set up Polariseq from a git clone, in a virtual environment inside the clone.
# No root, nothing installed into the system Python.
#
#   git clone <repository> polariseq && cd polariseq
#   bash examples/setup_node.sh
#
# Run it again after `git pull` to rebuild. Options:
#
#   --python CMD       the interpreter to use (default python3; 3.10 to 3.14)
#   --install-rust     install Rust with rustup into ~/.cargo if it is missing
#                      (downloads and runs https://sh.rustup.rs; without this
#                      flag the script stops and tells you what to do)
#   --wheel FILE       install a wheel built elsewhere instead of compiling
#   --wheelhouse DIR   install every Python package from DIR, with no network
#
# Offline machines: see examples/NODE_SETUP.md ("Without access to PyPI").
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}; INSTALL_RUST=0; WHEEL=""; WHEELHOUSE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY=$2; shift 2 ;;
    --install-rust) INSTALL_RUST=1; shift ;;
    --wheel) WHEEL=$2; shift 2 ;;
    --wheelhouse) WHEELHOUSE=$2; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
say() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

say "Python"
command -v "$PY" >/dev/null || die "$PY not found. Load a module (module load python) or pass --python."
"$PY" - <<'EOF' || die "Polariseq needs Python 3.10 to 3.14."
import sys
v = sys.version_info[:2]
print(sys.version.split()[0], sys.executable)
sys.exit(0 if (3, 10) <= v <= (3, 14) else 1)
EOF

if [ -z "$WHEEL" ]; then
  say "Rust and a C compiler"
  export PATH="$HOME/.cargo/bin:$PATH"
  if ! command -v cargo >/dev/null; then
    if [ "$INSTALL_RUST" = 1 ]; then
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
      export PATH="$HOME/.cargo/bin:$PATH"
    else
      die "Rust is not installed. Re-run with --install-rust (no root needed; it goes to ~/.cargo), or install it from https://rustup.rs, or build a wheel elsewhere and use --wheel."
    fi
  fi
  cargo --version
  command -v cc >/dev/null || command -v gcc >/dev/null || die "no C compiler. Load one (module load gcc) or install build-essential / Xcode command line tools."
fi

say "Virtual environment in $(pwd)/.venv"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
PIP=(python -m pip install --quiet)
[ -n "$WHEELHOUSE" ] && PIP+=(--no-index --find-links "$WHEELHOUSE")
"${PIP[@]}" numpy scipy matplotlib pandas anndata psutil

if [ -n "$WHEEL" ]; then
  say "Installing the wheel $WHEEL"
  "${PIP[@]}" --force-reinstall --no-deps "$WHEEL"
else
  say "Building (release mode; a first build takes a few minutes, HDF5 is compiled from source)"
  "${PIP[@]}" maturin
  maturin develop --release
fi

say "Check"
python -c "import polariseq as ps; print(ps.get_build_info()); print(ps.resources())"
python examples/06_large_run.py --out examples/output/setup_check --ram-gb 1 --no-export >/dev/null
echo "the example ran: examples/output/setup_check/report.json"

cat <<EOF

Done. Next time:   source $(pwd)/.venv/bin/activate
Then:              python examples/06_large_run.py YOUR.h5ad --out runs/yours --project /scratch/\$USER/polariseq --ram-gb N --threads N
Guide:             examples/NODE_SETUP.md
EOF
