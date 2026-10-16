#!/usr/bin/env bash
# Download and verify the Archive Genocide event-graph tables (no media).
#
# Usage:
#   archive_graph/ingest/download.sh
#   RAW_DIR=/path/to/raw archive_graph/ingest/download.sh
#
# Verifies GPG signature of SHA256SUMS.txt, then sha256 of every listed file,
# then that README.txt names the pinned BUILD_ID.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${RAW_DIR:-${MODULE_DIR}/raw}"
DATA_URL="${DATA_URL:-https://archivegenocide.com/lab/data}"
KEY_URL="${KEY_URL:-https://archivegenocide.com/key.asc}"
PINNED_BUILD="$(tr -d '[:space:]' < "${MODULE_DIR}/BUILD_ID")"
EXPECTED_FPR="C24EC92B12D6670A2516065F9B4D575499AFA53C"
KEY_ID="9B4D575499AFA53C"
USER_AGENT="casualty-source-aggregation archive_graph ingest (event graph only; no media)"

FILES=(
  README.txt
  actors.csv.gz
  event_clips.csv.gz
  event_entities.csv.gz
  event_relations.csv.gz
  events.csv.gz
  events.geojson.gz
  facilities.csv.gz
  graph.graphml.gz
  people.csv.gz
  places.csv.gz
)

FORBIDDEN_PATTERNS='torrent|archivegenocide-media|\.mp4|\.webm|\.mkv'

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required command not found: $1" >&2
    exit 1
  }
}

need curl
need gpg
if command -v sha256sum >/dev/null 2>&1; then
  SHA256SUM=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  SHA256SUM=(shasum -a 256)
else
  echo "error: need sha256sum or shasum" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

fetch() {
  local name="$1"
  if echo "${name}" | grep -Eiq "${FORBIDDEN_PATTERNS}"; then
    echo "error: refusing to download media/torrent path: ${name}" >&2
    exit 1
  fi
  echo "fetch ${name}"
  curl -fL --retry 5 --retry-delay 2 -C - \
    -A "${USER_AGENT}" \
    -o "${name}.partial" \
    "${DATA_URL}/${name}"
  mv -f "${name}.partial" "${name}"
}

echo "==> pinned build ${PINNED_BUILD}"
echo "==> writing to ${RAW_DIR}"
echo "==> source ${DATA_URL}/  (event graph only; no media torrents)"

fetch SHA256SUMS.txt
fetch SHA256SUMS.txt.asc

echo "==> importing release key from ${KEY_URL}"
KEY_TMP="$(mktemp)"
cleanup() { rm -f "${KEY_TMP}"; }
trap cleanup EXIT
curl -fL --retry 5 -A "${USER_AGENT}" -o "${KEY_TMP}" "${KEY_URL}"
gpg --import "${KEY_TMP}" >/dev/null 2>&1 || gpg --import "${KEY_TMP}"

FPR="$(gpg --with-colons --fingerprint "${KEY_ID}" | awk -F: '/^fpr:/ {print $10; exit}')"
if [[ "${FPR}" != "${EXPECTED_FPR}" ]]; then
  echo "error: GPG fingerprint mismatch" >&2
  echo "  expected ${EXPECTED_FPR}" >&2
  echo "  got      ${FPR}" >&2
  exit 1
fi
echo "==> GPG fingerprint OK (${FPR})"

echo "==> verifying SHA256SUMS.txt signature"
VERIFY_OUT="$(gpg --verify SHA256SUMS.txt.asc SHA256SUMS.txt 2>&1)" || {
  echo "${VERIFY_OUT}" >&2
  echo "error: GPG signature verification failed" >&2
  exit 1
}
echo "${VERIFY_OUT}"
if ! echo "${VERIFY_OUT}" | grep -qi "Good signature"; then
  echo "error: gpg did not report a good signature" >&2
  exit 1
fi

missing_from_manifest=0
for f in "${FILES[@]}"; do
  if ! awk '{print $2}' SHA256SUMS.txt | grep -qx "${f}"; then
    echo "error: ${f} not listed in SHA256SUMS.txt" >&2
    missing_from_manifest=1
  fi
done
if [[ "${missing_from_manifest}" -ne 0 ]]; then
  exit 1
fi

while read -r _ listed; do
  [[ -z "${listed:-}" ]] && continue
  if echo "${listed}" | grep -Eiq "${FORBIDDEN_PATTERNS}"; then
    echo "error: manifest lists a media/torrent file: ${listed}" >&2
    exit 1
  fi
done < SHA256SUMS.txt

for f in "${FILES[@]}"; do
  fetch "${f}"
done

echo "==> checking SHA256SUMS.txt"
"${SHA256SUM[@]}" -c SHA256SUMS.txt

PINNED_JSON="${SCRIPT_DIR}/PINNED_SOURCE.json"
if [[ -f "${PINNED_JSON}" ]]; then
  pin_sums="$(python3 -c "import json; print(json.load(open('${PINNED_JSON}'))['sha256_sha256sums'])")"
  got_sums="$("${SHA256SUM[@]}" SHA256SUMS.txt | awk '{print $1}')"
  if [[ "${got_sums}" != "${pin_sums}" ]]; then
    echo "error: SHA256SUMS.txt hash ${got_sums} != pinned ${pin_sums}" >&2
    echo "       this is not the f76bd2fb manifest this repo pins" >&2
    exit 1
  fi
  echo "==> pinned SHA256SUMS.txt hash OK"
fi

if ! grep -q "Build ${PINNED_BUILD}" README.txt; then
  echo "error: README.txt does not name pinned build ${PINNED_BUILD}" >&2
  echo "       (refusing to proceed with a different dataset build)" >&2
  grep -n -i "build" README.txt | head >&2 || true
  exit 1
fi

echo "==> verified event-graph build ${PINNED_BUILD}"
echo "    raw files are local-only; do not commit archive_graph/raw/"
