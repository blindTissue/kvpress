#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs both experiments (color-apparel + sentiment) across all models,
# data types, and locations.
#
# Usage:
#   cd /path/to/kvpress
#   bash experiments/run_all.sh            # full run (all 1000 samples)
#   bash experiments/run_all.sh 50         # quick run (50 samples)

set -euo pipefail

NUM_SAMPLES="${1:--1}"   # first arg = num samples, default -1 (all)

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-4B"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
)

DATA_TYPES=(1 2)
LOCATIONS=("first" "last")
CONDITIONS="1 2 3 4 5"
QUESTION_TYPES="1 2"
MAX_NEW_TOKENS=30

RESULTS_DIR="experiments/results"
LOG_FILE="${RESULTS_DIR}/run_all.log"

mkdir -p "${RESULTS_DIR}"

model_short() {
    echo "$1" | sed 's|.*/||'
}

# ── Count total runs ─────────────────────────────────────────────────
# Color-apparel: models × data_types × locations
color_runs=$(( ${#MODELS[@]} * ${#DATA_TYPES[@]} * ${#LOCATIONS[@]} ))
# Sentiment:    models × data_types × locations
sentiment_runs=$(( ${#MODELS[@]} * ${#DATA_TYPES[@]} * ${#LOCATIONS[@]} ))
total=$(( color_runs + sentiment_runs ))
current=0
skipped=0
failed=0

echo "==========================================" | tee "${LOG_FILE}"
echo "KV Cache Locality - Full Batch Run"        | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "Models:       ${#MODELS[@]}"               | tee -a "${LOG_FILE}"
echo "Data types:   ${DATA_TYPES[*]}"             | tee -a "${LOG_FILE}"
echo "Locations:    ${LOCATIONS[*]}"               | tee -a "${LOG_FILE}"
echo "Num samples:  ${NUM_SAMPLES}"                | tee -a "${LOG_FILE}"
echo "Total runs:   ${total} (${color_runs} color-apparel + ${sentiment_runs} sentiment)" | tee -a "${LOG_FILE}"
echo "Results dir:  ${RESULTS_DIR}"                | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# ── Phase 1: Color-Apparel Experiment ────────────────────────────────
echo "========== PHASE 1: COLOR-APPAREL ==========" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for model in "${MODELS[@]}"; do
    short="$(model_short "${model}")"
    model_dir="${RESULTS_DIR}/color_apparel/${short}"
    mkdir -p "${model_dir}"

    for dt in "${DATA_TYPES[@]}"; do
        for loc in "${LOCATIONS[@]}"; do
            current=$((current + 1))
            out_file="${model_dir}/type${dt}_loc${loc}.json"

            if [[ -f "${out_file}" ]]; then
                skipped=$((skipped + 1))
                echo "[${current}/${total}] SKIP (exists) | color-apparel | ${short} | type=${dt} loc=${loc}" | tee -a "${LOG_FILE}"
            else
                echo "[${current}/${total}] $(date '+%Y-%m-%d %H:%M:%S') | color-apparel | ${short} | type=${dt} loc=${loc}" | tee -a "${LOG_FILE}"

                start_time=$(date +%s)

                if python -m experiments.run_experiment \
                    --model-name "${model}" \
                    --data-type "${dt}" \
                    --location "${loc}" \
                    --conditions ${CONDITIONS} \
                    --question-types ${QUESTION_TYPES} \
                    --num-samples "${NUM_SAMPLES}" \
                    --max-new-tokens "${MAX_NEW_TOKENS}" \
                    --output-path "${out_file}" \
                    --rerotate \
                    2>&1 | tee -a "${LOG_FILE}"; then

                    elapsed=$(( $(date +%s) - start_time ))
                    echo "  -> Done in ${elapsed}s, saved to ${out_file}" | tee -a "${LOG_FILE}"
                else
                    elapsed=$(( $(date +%s) - start_time ))
                    failed=$((failed + 1))
                    echo "  -> FAILED after ${elapsed}s (continuing)" | tee -a "${LOG_FILE}"
                fi
            fi

            echo "" | tee -a "${LOG_FILE}"
        done
    done
done

# ── Phase 2: Sentiment Experiment ────────────────────────────────────
echo "========== PHASE 2: SENTIMENT ==========" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for model in "${MODELS[@]}"; do
    short="$(model_short "${model}")"
    model_dir="${RESULTS_DIR}/sentiment/${short}"
    mkdir -p "${model_dir}"

    for dt in "${DATA_TYPES[@]}"; do
        for loc in "${LOCATIONS[@]}"; do
            current=$((current + 1))
            out_file="${model_dir}/type${dt}_loc${loc}.json"

            if [[ -f "${out_file}" ]]; then
                skipped=$((skipped + 1))
                echo "[${current}/${total}] SKIP (exists) | sentiment | ${short} | type=${dt} loc=${loc}" | tee -a "${LOG_FILE}"
            else
                echo "[${current}/${total}] $(date '+%Y-%m-%d %H:%M:%S') | sentiment | ${short} | type=${dt} loc=${loc}" | tee -a "${LOG_FILE}"

                start_time=$(date +%s)

                if python -m experiments.run_sentiment_experiment \
                    --model-name "${model}" \
                    --data-type "${dt}" \
                    --location "${loc}" \
                    --conditions ${CONDITIONS} \
                    --num-samples "${NUM_SAMPLES}" \
                    --max-new-tokens "${MAX_NEW_TOKENS}" \
                    --output-path "${out_file}" \
                    --rerotate \
                    2>&1 | tee -a "${LOG_FILE}"; then

                    elapsed=$(( $(date +%s) - start_time ))
                    echo "  -> Done in ${elapsed}s, saved to ${out_file}" | tee -a "${LOG_FILE}"
                else
                    elapsed=$(( $(date +%s) - start_time ))
                    failed=$((failed + 1))
                    echo "  -> FAILED after ${elapsed}s (continuing)" | tee -a "${LOG_FILE}"
                fi
            fi

            echo "" | tee -a "${LOG_FILE}"
        done
    done
done

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Batch complete: ${current} runs, ${skipped} skipped, ${failed} failed" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
