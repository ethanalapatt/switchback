#!/usr/bin/env bash
# Reproduce Switchback's gates and, once implemented, its benchmark.
#
# Usage: bash scripts/reproduce.sh --preset {cpu|smoke|full}
#
#   cpu    offline correctness gate: no GPU, no weights, no network
#   smoke  cpu gate plus the GPU model gate and the small smoke benchmark
#   full   cpu gate plus the GPU model gate and the complete primary matrix
#
# Environment: set BENCH_MAX_SECONDS to cap one unattended stretch of the
# benchmark. The run is resumable, so rerunning the same command continues it.
#
# Exit codes: 0 every requested stage ran and passed; 1 a stage failed;
# 3 the preset is not fully implemented yet and reported INCOMPLETE.
# A stage that is not implemented is never reported as a pass.
set -uo pipefail

PRESET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset) PRESET="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$PRESET" ]]; then
    echo "error: --preset {cpu|smoke|full} is required" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Prefer an activated interpreter, then this repository's .venv, then python3.
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        PYTHON="${VIRTUAL_ENV}/bin/python"
    elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
        PYTHON="${ROOT}/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    else
        PYTHON="$(command -v python)"
    fi
fi
if [[ -z "${PYTHON}" ]]; then
    echo "error: no python interpreter found; set PYTHON=/path/to/python" >&2
    exit 2
fi
FAILED=()
INCOMPLETE=()

run() {
    local label="$1"; shift
    echo
    echo "=== ${label}"
    echo "--- \$ $*"
    if "$@"; then
        echo "--- PASS ${label}"
    else
        echo "--- FAIL ${label}" >&2
        FAILED+=("${label}")
    fi
}

not_implemented() {
    local label="$1" milestone="$2"
    echo
    echo "=== ${label}"
    echo "--- INCOMPLETE ${label}: not implemented until ${milestone}"
    INCOMPLETE+=("${label} (${milestone})")
}

echo "Switchback reproduce, preset=${PRESET}, python=$("$PYTHON" -c 'import sys; print(sys.executable)')"

# ---- Offline gate. Runs everywhere, downloads nothing. --------------------
run "environment doctor"        "$PYTHON" -m switchback doctor --out artifacts/environment.json
run "lint"                      "$PYTHON" -m ruff check .
run "format"                    "$PYTHON" -m ruff format --check .
run "types"                     "$PYTHON" -m mypy src/switchback
run "unit and integration (cpu)" "$PYTHON" -m pytest -m 'not gpu and not download' -q
run "property tests"            "$PYTHON" -m pytest tests/property -q
run "report integrity tests"    "$PYTHON" -m pytest tests/report -q
run "offline tiny-model demo"   "$PYTHON" -m switchback demo

if [[ "$PRESET" != "cpu" ]]; then
    # ---- GPU gate. Needs a CUDA device and the pinned public weights. -----
    run "gpu doctor"       "$PYTHON" -m switchback doctor --require-gpu
    run "model revisions"  "$PYTHON" -m switchback resolve-models --out data/manifest.json
    run "model pair check" "$PYTHON" -m switchback verify-models --out artifacts/model_check.json
    run "gpu tests"        "$PYTHON" -m pytest -m gpu -q

    case "$PRESET" in
        smoke)  CONFIG=configs/smoke.toml;   RUN_DIR=artifacts/runs/smoke ;;
        full)   CONFIG=configs/primary.toml; RUN_DIR=artifacts/runs/primary ;;
        *)      echo "unknown preset: ${PRESET}" >&2; exit 2 ;;
    esac

    run "workload preparation" "$PYTHON" -m bench.prepare --config "$CONFIG"
    run "controller calibration" "$PYTHON" -m switchback calibrate \
        --local-files-only --out artifacts/calibration.json
    run "validation evidence" "$PYTHON" -m switchback evidence --gpu \
        --artifact artifacts/environment.json \
        --artifact artifacts/calibration.json \
        --out artifacts/evidence.json

    # The benchmark is resumable: rerun the same command to continue an
    # interrupted run. --max-seconds caps one unattended stretch.
    run "benchmark" "$PYTHON" -m bench.run --config "$CONFIG" --out "$RUN_DIR" \
        ${BENCH_MAX_SECONDS:+--max-seconds "$BENCH_MAX_SECONDS"}
    run "evidence validation" "$PYTHON" -m bench.validate "$RUN_DIR" --config "$CONFIG"
    run "report rendering" "$PYTHON" scripts/render_results.py "$RUN_DIR" \
        --out "$RUN_DIR/report.md"

    if [[ "$PRESET" == "full" ]]; then
        run "publish RESULTS.md" "$PYTHON" scripts/render_results.py "$RUN_DIR" \
            --out RESULTS.md
    else
        echo
        echo "=== RESULTS.md"
        echo "--- SKIP: the smoke preset is preliminary and never becomes RESULTS.md"
    fi
fi

echo
echo "=== summary"
if ((${#FAILED[@]})); then
    printf 'FAILED: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
if ((${#INCOMPLETE[@]})); then
    printf 'INCOMPLETE: %s\n' "${INCOMPLETE[@]}"
    echo "All implemented stages passed. The preset is not complete; see PROGRESS.md."
    exit 3
fi
echo "All stages passed."
