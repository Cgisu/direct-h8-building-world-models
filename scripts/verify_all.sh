#!/usr/bin/env bash
# Portable verification for the public repository.
#   PYTHON=.venv/bin/python bash scripts/verify_all.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
python_cmd="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/direct-h8-mpl-${UID}"
mkdir -p "${MPLCONFIGDIR}"

targets=()
while IFS= read -r line; do
  [[ -z "${line}" || "${line}" == \#* ]] && continue
  targets+=("${line}")
done < scripts/portable_tests.txt
if [[ ${#targets[@]} -eq 0 ]]; then
  echo "no portable unit test targets found" >&2
  exit 1
fi

"${python_cmd}" scripts/check_dependencies.py
"${python_cmd}" scripts/verify_repository.py
"${python_cmd}" scripts/verify_downstream_subset.py
"${python_cmd}" -m unittest "${targets[@]}"
"${python_cmd}" scripts/verify_downstream_contract.py
echo "PORTABLE RELEASE CONTRACT: PASS"
