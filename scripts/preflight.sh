#!/bin/bash
# preflight.sh — Pre-flight validation for eval SLURM jobs
#
# Source this file and call preflight_eval_check before torchrun.
# Catches common mistakes BEFORE burning GPU time:
#   - Missing checkpoint files
#   - Wrong model config (bagel_mot vs bagel_mot_vcot)
#   - Stale prediction files that would be reused
#   - Output paths on home directory (quota risk)
#
# Usage in SLURM scripts:
#   source scripts/preflight.sh
#   preflight_eval_check "${CKPT}" "${SUBSET}" "bagel_mot"
#
# Set PREFLIGHT_STRICT=1 to treat warnings as failures.

_preflight_pass=0
_preflight_warn=0
_preflight_fail=0

_pf_pass() {
    echo "[PASS] $1"
    _preflight_pass=$((_preflight_pass + 1))
}

_pf_warn() {
    echo "[WARN] $1"
    _preflight_warn=$((_preflight_warn + 1))
}

_pf_fail() {
    echo "[FAIL] $1"
    _preflight_fail=$((_preflight_fail + 1))
}

# Check 1: Checkpoint exists and contains expected files
_check_checkpoint() {
    local ckpt="$1"

    if [[ ! -d "$ckpt" ]]; then
        _pf_fail "Checkpoint directory not found: ${ckpt}"
        return
    fi

    # Look for safetensors files (converted checkpoint)
    local safetensors
    safetensors=$(ls "${ckpt}"/*.safetensors 2>/dev/null | head -1)
    if [[ -n "$safetensors" ]]; then
        _pf_pass "Checkpoint exists: ${ckpt} ($(basename "$safetensors") found)"
        return
    fi

    # Look for FSDP sharded format (model/ dir with .distcp files)
    if [[ -d "${ckpt}/model" ]]; then
        local distcp
        distcp=$(ls "${ckpt}"/model/*.distcp 2>/dev/null | head -1)
        if [[ -n "$distcp" ]]; then
            _pf_warn "Checkpoint is FSDP sharded format (not converted): ${ckpt}"
            return
        fi
    fi

    _pf_fail "No .safetensors or FSDP shards found in: ${ckpt}"
}

# Check 2: Model config matches checkpoint type
_check_model_config() {
    local ckpt="$1"
    local model_config="$2"
    local ckpt_lower
    ckpt_lower=$(echo "$ckpt" | tr '[:upper:]' '[:lower:]')

    if [[ "$model_config" == "bagel_mot_vcot" || "$model_config" == "bagel_mot_vcot_prefill" ]]; then
        # vcot/prefill config should be used for vcot or mmcot checkpoints
        if echo "$ckpt_lower" | grep -qE '(vcot|mmcot|visual_cot|mixed)'; then
            _pf_pass "Model config: ${model_config} matches VCoT/MMCoT/Mixed checkpoint"
        elif echo "$ckpt_lower" | grep -qE '(answer_only|_ao_|/ao/|textcot|text_cot)'; then
            _pf_fail "Model config mismatch: ${model_config} (image-gen) used with AO/TextCoT checkpoint — should be bagel_mot"
        else
            _pf_warn "Model config: ${model_config} — cannot determine checkpoint type from path, verify manually"
        fi
    elif [[ "$model_config" == "bagel_mot_answeronly" || "$model_config" == "bagel_mot_nothink" ]]; then
        # answeronly/nothink configs are valid for any checkpoint type (text-only eval with optional VAE input)
        _pf_pass "Model config: ${model_config} (text-only eval, valid for any checkpoint)"
    elif [[ "$model_config" == "bagel_mot" ]]; then
        # bagel_mot (text-only) should be used for AO or textcot checkpoints
        if echo "$ckpt_lower" | grep -qE '(answer_only|_ao_|/ao/|textcot|text_cot)'; then
            _pf_pass "Model config: ${model_config} matches AO/TextCoT checkpoint"
        elif echo "$ckpt_lower" | grep -qE '(vcot|mmcot|visual_cot)'; then
            _pf_fail "Model config mismatch: ${model_config} (text-only) used with VCoT/MMCoT checkpoint — should be bagel_mot_vcot"
        else
            _pf_warn "Model config: ${model_config} — cannot determine checkpoint type from path, verify manually"
        fi
    else
        _pf_warn "Unknown model config: ${model_config}"
    fi
}

# Check 3: Stale result files that would be reused
_check_stale_results() {
    local ckpt="$1"
    local subset="$2"
    local work_dir="${ckpt}/eval"

    if [[ ! -d "$work_dir" ]]; then
        _pf_pass "No stale results: ${work_dir} does not exist yet"
        return
    fi

    local stale_files=()

    # Results are stored in work_dir/<model_config>/ subdirectories
    # Scan all subdirectories and work_dir itself
    local search_dirs=("${work_dir}" "${work_dir}"/*)
    for dir in "${search_dirs[@]}"; do
        [[ -d "$dir" ]] || continue
        for f in "${dir}"/*"${subset}"*_result.xlsx "${dir}"/*"${subset}"*.pkl "${dir}"/*"${subset}"*_acc.csv; do
            [[ -e "$f" ]] && stale_files+=("${f#${work_dir}/}")
        done
    done

    if [[ ${#stale_files[@]} -gt 0 ]]; then
        _pf_warn "Stale results found in ${work_dir}/ — --reuse will pick these up: ${stale_files[*]}"
    else
        _pf_pass "No stale results for ${subset} in ${work_dir}/"
    fi
}

# Check 4: Output path is not on home directory
_check_storage_tier() {
    local ckpt="$1"

    if [[ "$ckpt" == /gpfs/home/* ]]; then
        _pf_fail "Checkpoint/output on home directory (10GB quota!): ${ckpt} — use scrubbed or projects storage"
    elif [[ "$ckpt" == /gpfs/scrubbed/* ]]; then
        _pf_pass "Storage tier: scrubbed (OK)"
    elif [[ "$ckpt" == /gpfs/projects/* ]]; then
        _pf_pass "Storage tier: projects (OK)"
    else
        _pf_warn "Storage tier: unrecognized path prefix — verify sufficient space: ${ckpt}"
    fi
}

# Main entry point
preflight_eval_check() {
    local ckpt="$1"
    local subset="$2"
    local model_config="$3"

    if [[ -z "$ckpt" || -z "$subset" || -z "$model_config" ]]; then
        echo "PREFLIGHT ERROR: Usage: preflight_eval_check <ckpt_path> <subset> <model_config>"
        exit 1
    fi

    _preflight_pass=0
    _preflight_warn=0
    _preflight_fail=0

    echo "=== PREFLIGHT CHECK ==="
    echo "  Checkpoint: ${ckpt}"
    echo "  Subset:     ${subset}"
    echo "  Config:     ${model_config}"
    echo ""

    _check_checkpoint "$ckpt"
    _check_model_config "$ckpt" "$model_config"
    _check_stale_results "$ckpt" "$subset"
    _check_storage_tier "$ckpt"

    echo ""
    echo "=== PREFLIGHT: ${_preflight_pass} PASS, ${_preflight_warn} WARN, ${_preflight_fail} FAIL ==="

    if [[ $_preflight_fail -gt 0 ]]; then
        echo "PREFLIGHT FAILED — aborting job."
        exit 1
    fi

    if [[ "${PREFLIGHT_STRICT:-0}" == "1" && $_preflight_warn -gt 0 ]]; then
        echo "PREFLIGHT STRICT MODE — warnings treated as failures. Aborting."
        exit 1
    fi

    echo "Preflight passed. Proceeding with eval."
    echo ""
}
