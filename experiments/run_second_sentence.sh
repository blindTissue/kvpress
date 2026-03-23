#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Second Sentence Recall Experiment
#
# Tests whether a distant surviving apparel token (in a second sentence)
# enables color recall after evicting the color-apparel pair from the
# first sentence.
#
# Fixed parameters: data_type=1, location=first, question_type=1 (ask_color),
#                   conditions=1 2 3
#
# Usage:
#   cd /path/to/kvpress
#   bash experiments/run_second_sentence.sh            # full run (all 1000 samples)
#   bash experiments/run_second_sentence.sh 50         # quick run (50 samples)

set -euo pipefail

NUM_SAMPLES="${1:--1}"

MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-4B"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
)

DATASET_PATH="custom_dataset/color_apparel_second_sentence_dataset.json"
CONDITIONS="1 2 3"
QUESTION_TYPES="1"
DATA_TYPE=1
LOCATION="first"
MAX_NEW_TOKENS=30

RESULTS_DIR="experiments/results"
LOG_FILE="${RESULTS_DIR}/run_second_sentence.log"

mkdir -p "${RESULTS_DIR}"

model_short() {
    echo "$1" | sed 's|.*/||'
}

total=${#MODELS[@]}
current=0
skipped=0
failed=0

echo "==========================================" | tee "${LOG_FILE}"
echo "Second Sentence Recall Experiment"          | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "Models:       ${#MODELS[@]}"                | tee -a "${LOG_FILE}"
echo "Data type:    ${DATA_TYPE}"                  | tee -a "${LOG_FILE}"
echo "Location:     ${LOCATION}"                   | tee -a "${LOG_FILE}"
echo "Conditions:   ${CONDITIONS}"                 | tee -a "${LOG_FILE}"
echo "Question:     ask_color only"                | tee -a "${LOG_FILE}"
echo "Num samples:  ${NUM_SAMPLES}"                | tee -a "${LOG_FILE}"
echo "Dataset:      ${DATASET_PATH}"               | tee -a "${LOG_FILE}"
echo "Total runs:   ${total}"                      | tee -a "${LOG_FILE}"
echo "Results dir:  ${RESULTS_DIR}"                | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

for model in "${MODELS[@]}"; do
    short="$(model_short "${model}")"
    model_dir="${RESULTS_DIR}/color_apparel_second_sentence/${short}"
    mkdir -p "${model_dir}"

    current=$((current + 1))
    out_file="${model_dir}/type${DATA_TYPE}_loc${LOCATION}.json"

    # Gemma3's dual RoPE is incompatible with rerotation
    REROTATE_FLAG=""
    if [[ "${model}" != *"gemma"* ]]; then
        REROTATE_FLAG="--rerotate"
    fi

    if [[ -f "${out_file}" ]]; then
        skipped=$((skipped + 1))
        echo "[${current}/${total}] SKIP (exists) | ${short}" | tee -a "${LOG_FILE}"
    else
        echo "[${current}/${total}] $(date '+%Y-%m-%d %H:%M:%S') | ${short}" | tee -a "${LOG_FILE}"

        start_time=$(date +%s)

        if python -m experiments.run_experiment \
            --model-name "${model}" \
            --dataset-path "${DATASET_PATH}" \
            --data-type "${DATA_TYPE}" \
            --location "${LOCATION}" \
            --conditions ${CONDITIONS} \
            --question-types ${QUESTION_TYPES} \
            --num-samples "${NUM_SAMPLES}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --output-path "${out_file}" \
            ${REROTATE_FLAG} \
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

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Batch complete: ${current} runs, ${skipped} skipped, ${failed} failed" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
