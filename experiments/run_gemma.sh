#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs Gemma models WITHOUT --rerotate (Gemma3's dual RoPE
# for full/sliding attention layers is incompatible with our rerotation).

set -euo pipefail

NUM_SAMPLES="${1:--1}"

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)

DATA_TYPES=(1 2)
LOCATIONS=("first" "last")
CONDITIONS="1 2 3 4 5"
QUESTION_TYPES="1 2"
MAX_NEW_TOKENS=30

RESULTS_DIR="experiments/results"
LOG_FILE="${RESULTS_DIR}/run_gemma.log"

mkdir -p "${RESULTS_DIR}"

model_short() { echo "$1" | sed 's|.*/||'; }

total=$(( ${#MODELS[@]} * ${#DATA_TYPES[@]} * ${#LOCATIONS[@]} * 2 ))
current=0
failed=0

echo "==========================================" | tee "${LOG_FILE}"
echo "Gemma Batch Run (no rerotate)"             | tee -a "${LOG_FILE}"
echo "Total runs: ${total}"                       | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# ── Color-Apparel ────────────────────────────────────────────────────
echo "========== COLOR-APPAREL ==========" | tee -a "${LOG_FILE}"

for model in "${MODELS[@]}"; do
    short="$(model_short "${model}")"
    model_dir="${RESULTS_DIR}/color_apparel/${short}"
    mkdir -p "${model_dir}"

    for dt in "${DATA_TYPES[@]}"; do
        for loc in "${LOCATIONS[@]}"; do
            current=$((current + 1))
            out_file="${model_dir}/type${dt}_loc${loc}.json"
            echo "[${current}/${total}] $(date '+%H:%M:%S') | color-apparel | ${short} | type=${dt} loc=${loc}" | tee -a "${LOG_FILE}"
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
                2>&1 | tee -a "${LOG_FILE}"; then
                echo "  -> Done in $(( $(date +%s) - start_time ))s" | tee -a "${LOG_FILE}"
            else
                failed=$((failed + 1))
                echo "  -> FAILED after $(( $(date +%s) - start_time ))s" | tee -a "${LOG_FILE}"
            fi
            echo "" | tee -a "${LOG_FILE}"
        done
    done
done

# ── Sentiment ────────────────────────────────────────────────────────
echo "========== SENTIMENT ==========" | tee -a "${LOG_FILE}"

for model in "${MODELS[@]}"; do
    short="$(model_short "${model}")"
    model_dir="${RESULTS_DIR}/sentiment/${short}"
    mkdir -p "${model_dir}"

    for dt in "${DATA_TYPES[@]}"; do
        for loc in "${LOCATIONS[@]}"; do
            current=$((current + 1))
            out_file="${model_dir}/type${dt}_loc${loc}.json"
            echo "[${current}/${total}] $(date '+%H:%M:%S') | sentiment | ${short} | type=${dt} loc=${loc}" | tee -a "${LOG_FILE}"
            start_time=$(date +%s)

            if python -m experiments.run_sentiment_experiment \
                --model-name "${model}" \
                --data-type "${dt}" \
                --location "${loc}" \
                --conditions ${CONDITIONS} \
                --num-samples "${NUM_SAMPLES}" \
                --max-new-tokens "${MAX_NEW_TOKENS}" \
                --output-path "${out_file}" \
                2>&1 | tee -a "${LOG_FILE}"; then
                echo "  -> Done in $(( $(date +%s) - start_time ))s" | tee -a "${LOG_FILE}"
            else
                failed=$((failed + 1))
                echo "  -> FAILED after $(( $(date +%s) - start_time ))s" | tee -a "${LOG_FILE}"
            fi
            echo "" | tee -a "${LOG_FILE}"
        done
    done
done

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Gemma batch done: ${current} runs, ${failed} failed" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
