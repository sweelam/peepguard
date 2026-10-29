#!/usr/bin/env bash
# Build a standalone PeepGuard for the OS you run this on (Linux or macOS).
#   Linux -> dist/peepguard            (single file; run ./dist/peepguard run)
#   macOS -> dist/peepguard + dist/PeepGuard.app
# Needs Python 3.11+. Everything is built in a private venv under build/.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
VENV=build/venv-$(uname -s | tr '[:upper:]' '[:lower:]')
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -e ".[shield]" pyinstaller

# Build aside, then rename into dist/: a running peepguard keeps reading its old file
# (a one-file binary loads modules from itself lazily, so overwriting it in place crashes it).
rm -rf build/dist
"$VENV/bin/pyinstaller" --noconfirm --clean \
    --distpath build/dist --workpath build/pyinstaller packaging/peepguard.spec
mkdir -p dist
mv -f build/dist/peepguard dist/peepguard
if [ -d build/dist/PeepGuard.app ]; then rm -rf dist/PeepGuard.app; mv build/dist/PeepGuard.app dist/; fi

echo
echo "Built:"
ls -lh dist/peepguard
[ -d dist/PeepGuard.app ] && echo "      dist/PeepGuard.app"
"dist/peepguard" --help >/dev/null && echo "Smoke test: ok"
