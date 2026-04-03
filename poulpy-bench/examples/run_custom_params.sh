#!/usr/bin/env bash
# run_custom_params.sh
#
# Runs poulpy-bench benchmarks using the parameter set defined in
# custom_params.json.  Backends and operations to run are controlled via the
# JSON "backends" and "run" fields respectively.
#
# Usage (from the workspace root):
#   bash poulpy-bench/examples/run_custom_params.sh [--avx] [--baseline <name>] [--compare <name>]
#
# Options:
#   --avx              enable AVX2/FMA backends (also set automatically when
#                      the JSON "backends" field contains an AVX label)
#   --baseline <name>  save results under this baseline name for later comparison
#   --compare  <name>  compare against a previously saved baseline
#
# When AVX is active (via --avx or JSON), RUSTFLAGS="-C target-feature=+avx2,+fma"
# is set automatically so the crate compiles without a manual RUSTFLAGS export.
#
# Backend selection is done via the JSON "backends" field:
#   "backends": ["fft64-ref"]               # ref only
#   "backends": ["fft64-avx", "ntt120-avx"] # AVX only (auto-enables the feature)
#   (omit)                                  # all compiled-in backends
#
# Examples:
#   # plain run
#   bash poulpy-bench/examples/run_custom_params.sh
#
#   # save a baseline tagged "before"
#   bash poulpy-bench/examples/run_custom_params.sh --baseline before
#
#   # compare against it after a code change
#   bash poulpy-bench/examples/run_custom_params.sh --compare before

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_FILE="$SCRIPT_DIR/custom_params.json"

# ── argument parsing ──────────────────────────────────────────────────────────

FEATURES=""
# Arguments to pass after `--` to Criterion (no leading `--` here).
CRITERION_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --avx)
      FEATURES="--features enable-avx"
      shift
      ;;
    --baseline)
      CRITERION_ARGS+=(--save-baseline "$2")
      shift 2
      ;;
    --compare)
      CRITERION_ARGS+=(--baseline "$2")
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ── run ───────────────────────────────────────────────────────────────────────

export POULPY_BENCH_PARAMS="$PARAMS_FILE"

echo "Using params: $PARAMS_FILE"
echo "Contents:"
cat "$PARAMS_FILE"
echo ""

# ── known binaries ────────────────────────────────────────────────────────────

ALL_BINARIES=(
  vec_znx vec_znx_big vec_znx_dft convolution svp vmp fft ntt
  operations encryption decryption automorphism external_product keyswitch
  blind_rotate circuit_bootstrapping bdd_prepare bdd_arithmetic
  standard
)

DEFAULT_BINARIES=(
  vec_znx_big vec_znx_dft vmp svp
  operations encryption external_product
  standard
)

run_bench() {
  local binary="$1" filter="${2:-}"
  echo "────────────────────────────────────────"
  echo "  bench: $binary${filter:+ ($filter)}"
  echo "────────────────────────────────────────"
  # Collect all post-`--` args into one array so there is exactly one `--`.
  local post_sep=("${CRITERION_ARGS[@]+"${CRITERION_ARGS[@]}"}")
  [[ -n "$filter" ]] && post_sep+=("$filter")
  local sep_args=()
  [[ ${#post_sep[@]} -gt 0 ]] && sep_args=(-- "${post_sep[@]}")
  # shellcheck disable=SC2086
  cargo bench -p poulpy-bench --bench "$binary" $FEATURES \
    "${sep_args[@]+"${sep_args[@]}"}"
}

# ── parse JSON ───────────────────────────────────────────────────────────────

BINARY_LIST=()   # entries that are exact binary names  → run whole binary
FILTER_TERMS=()  # entries that are not binary names    → Criterion filter
BACKEND_TERMS=() # backend labels from "backends"

if command -v jq &>/dev/null && [[ -f "$PARAMS_FILE" ]]; then
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    if printf '%s\n' "${ALL_BINARIES[@]}" | grep -qx "$entry"; then
      BINARY_LIST+=("$entry")
    else
      FILTER_TERMS+=("$entry")
    fi
  done < <(jq -r '.run[]?' "$PARAMS_FILE" 2>/dev/null)

  while IFS= read -r backend; do
    [[ -z "$backend" ]] && continue
    BACKEND_TERMS+=("$backend")
    # Auto-enable AVX feature flag if any AVX backend is requested.
    if [[ "$backend" == *avx* ]]; then
      FEATURES="--features enable-avx"
    fi
  done < <(jq -r '.backends[]?' "$PARAMS_FILE" 2>/dev/null)
fi

# AVX2+FMA are required at compile time whenever the AVX feature is active.
if [[ "$FEATURES" == *enable-avx* ]]; then
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+avx2,+fma"
fi

# Build the Criterion filter: (fn1|fn2).*(b1|b2), or just one part if only
# one dimension is constrained.
build_filter() {
  local fn_part="" be_part=""
  [[ ${#FILTER_TERMS[@]} -gt 0 ]]  && fn_part="($(IFS='|'; echo "${FILTER_TERMS[*]}"))"
  [[ ${#BACKEND_TERMS[@]} -gt 0 ]] && be_part="($(IFS='|'; echo "${BACKEND_TERMS[*]}"))"
  if [[ -n "$fn_part" && -n "$be_part" ]]; then
    echo "${fn_part}.*${be_part}"
  else
    echo "${fn_part}${be_part}"
  fi
}
FILTER="$(build_filter)"

# ── run ───────────────────────────────────────────────────────────────────────

# 1. Run each explicitly named binary, with optional backend filter.
for binary in "${BINARY_LIST[@]}"; do
  run_bench "$binary" "${BACKEND_TERMS:+$(IFS='|'; echo "(${BACKEND_TERMS[*]})")}"
done

# 2. Run filter terms (functions and/or backends) against the default binary set.
#    If nothing was specified at all, run the full default set.
if [[ -n "$FILTER" ]]; then
  for binary in "${DEFAULT_BINARIES[@]}"; do
    run_bench "$binary" "$FILTER"
  done
elif [[ ${#BINARY_LIST[@]} -eq 0 ]]; then
  for binary in "${DEFAULT_BINARIES[@]}"; do
    run_bench "$binary"
  done
fi

echo ""
echo "Done. HTML reports: target/criterion/"
