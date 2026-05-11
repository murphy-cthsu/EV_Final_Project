#!/usr/bin/env bash
# Download the D-NeRF synthetic dataset (8 scenes, ~700 MB unpacked).
#
# Source: D-NeRF official release.
#   - Dropbox:    https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1
#   - Google Drive: https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt/view
#
# Use Dropbox by default (direct ?dl=1 link works with curl/wget; Google Drive
# requires gdown). Override via DATA_URL if you have a private mirror.
#
# Scenes: bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex.
# Articulated subset (used for headline ablation): jumpingjacks, hellwarrior, bouncingballs, standup.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/dnerf"
DATA_URL="${DATA_URL:-https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=1}"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

ARCHIVE="${DATA_DIR}/dnerf.zip"

if [ ! -f "${ARCHIVE}" ]; then
    echo "Downloading D-NeRF archive from ${DATA_URL}..."
    if command -v wget > /dev/null; then
        wget -q --show-progress -O "${ARCHIVE}" "${DATA_URL}"
    elif command -v curl > /dev/null; then
        curl -L -o "${ARCHIVE}" "${DATA_URL}"
    else
        echo "Need wget or curl." >&2
        exit 1
    fi
else
    echo "Archive already present at ${ARCHIVE}; skipping download."
fi

# The official archive unpacks to a top-level "data/" directory containing the scenes.
# We want them at ${DATA_DIR}/<scene>/ directly.
echo "Unpacking..."
TMP_EXTRACT="$(mktemp -d)"
unzip -q -o "${ARCHIVE}" -d "${TMP_EXTRACT}"

if [ -d "${TMP_EXTRACT}/data" ]; then
    cp -rn "${TMP_EXTRACT}/data/"* "${DATA_DIR}/"
else
    cp -rn "${TMP_EXTRACT}/"* "${DATA_DIR}/"
fi
rm -rf "${TMP_EXTRACT}"

EXPECTED_SCENES=(bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex)
MISSING=0
echo ""
echo "Scene presence check:"
for s in "${EXPECTED_SCENES[@]}"; do
    if [ -d "${DATA_DIR}/${s}" ]; then
        n_train=$(find "${DATA_DIR}/${s}/train" -name '*.png' 2>/dev/null | wc -l)
        echo "  ${s}: OK (${n_train} train frames)"
    else
        echo "  ${s}: MISSING"
        MISSING=$((MISSING + 1))
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "WARN: ${MISSING} scenes missing. The Dropbox link may have been rate-limited or moved."
    echo "Alternatives:"
    echo "  - Manual: download from https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip"
    echo "  - Manual: download from https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt"
    echo "  - Mirror: pip install gdown && gdown --id 19Na95wk0uikquivC7uKWVqllmTx-mBHt -O ${ARCHIVE}"
    exit 1
fi

echo ""
echo "D-NeRF ready at ${DATA_DIR}/"
echo "Articulated subset used in headline ablation: jumpingjacks, hellwarrior, bouncingballs, standup."
