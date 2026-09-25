#!/usr/bin/env bash
set -euo pipefail

# Read-only CARE audit.
# Examples:
#   CARE_SOURCE=data/raw/CARE/CARE.zip bash run_data_audit.sh
#   CARE_SOURCE=data/extracted/CARE bash run_data_audit.sh

CARE_SOURCE=${CARE_SOURCE:-data/raw/CARE/CARE.zip}
CARE_AUDIT_OUTPUT=${CARE_AUDIT_OUTPUT:-manifests/care_audit.json}
CARE_SAMPLE_N=${CARE_SAMPLE_N:-20}

mkdir -p manifests
python scripts/inspect_care.py "$CARE_SOURCE" --sample-n "$CARE_SAMPLE_N" --output "$CARE_AUDIT_OUTPUT"
