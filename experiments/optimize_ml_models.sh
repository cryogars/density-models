#!/bin/bash
# ======================================================
# Hyperparameter Optimization for Snow Density Models
# ======================================================

# Exit on error
set -e

# Configuration
SCRIPT_PATH="../snowmodels/utils/_hyperopt_utils.py"
DATA_PATH="../data/data_splits.pkl"
CONFIG_PATH="hyperparameters.yaml"
DB_PATH="sqlite:///optuna_studies.db"
LOG_DIR="logs"

# Create directories if they don't exist
mkdir -p $LOG_DIR

# Experiment settings
N_TRIALS=100  
MODELS=("lightgbm" "xgboost" "rf" "extratrees")
VARIANTS=("main" "climate_7d" "climate_14d")
ENCODERS=("onehot" "target" "catboost")

# Test settings (smaller subset)
TEST_MODELS=("lightgbm")
TEST_VARIANTS=("main")
TEST_ENCODERS=("catboost")
TEST_TRIALS=10

# ===================
# Helper Functions
# ===================

run_experiment() {
    local models="$1"
    local n_trials=$2
    local variants="$3"
    local encoders="$4"
    local tuning_mode="$5"
    
    local tuning_mode_=$(echo "$tuning_mode" | tr ' ' '_')
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    local log_filename="${tuning_mode_}_${timestamp}.log"
    
    echo "==============================================================================" | tee -a "$LOG_DIR/$log_filename"
    echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') - INFO - Log file: $log_filename" | tee -a "$LOG_DIR/$log_filename"
    echo "==============================================================================" | tee -a "$LOG_DIR/$log_filename"
        
    echo "============================================================"
    echo "Starting experiment with models: $models"
    echo "Variants: $variants"
    echo "Encoders: $encoders"
    echo "Tuning Mode: $tuning_mode"
    echo "Trials: $n_trials"
    echo "Time: $(date)"
    echo "============================================================"
        
    python $SCRIPT_PATH \
        --models $models \
        --variants $variants \
        --encoders $encoders \
        --tuning-mode $tuning_mode \
        --n-trials $n_trials \
        --data-path $DATA_PATH \
        --config-path $CONFIG_PATH \
        --storage-url $DB_PATH \
        2>&1 | tee -a "$LOG_DIR/$log_filename"
        
    echo "Completed: $tuning_mode experiment at $(date)"
    echo ""
}

# =======================
# Main Execution Options
# =======================

# Check command line argument for run mode
RUN_MODE=${1:-"all"}

case $RUN_MODE in
    "test")
        echo "Running TEST mode - Quick validation (default only)"
        echo "=================================================="
        echo "Using: ${TEST_MODELS[*]} | ${TEST_VARIANTS[*]} | ${TEST_ENCODERS[*]} | $TEST_TRIALS trials"
        echo ""
        run_experiment "${TEST_MODELS[*]}" $TEST_TRIALS "${TEST_VARIANTS[*]}" "${TEST_ENCODERS[*]}" "default"
        ;;

    "test-all")
        echo "Running TEST mode - Both default and tune"
        echo "========================================="
        echo "Using: ${TEST_MODELS[*]} | ${TEST_VARIANTS[*]} | ${TEST_ENCODERS[*]} | $TEST_TRIALS trials each"
        echo ""
        echo "--- Running test DEFAULT mode ---"
        run_experiment "${TEST_MODELS[*]}" $TEST_TRIALS "${TEST_VARIANTS[*]}" "${TEST_ENCODERS[*]}" "default"
        echo ""
        echo "--- Running test TUNE mode ---"
        run_experiment "${TEST_MODELS[*]}" $TEST_TRIALS "${TEST_VARIANTS[*]}" "${TEST_ENCODERS[*]}" "tune"
        ;;

    "default")
        echo "Running the default setting"
        echo "==========================="
        run_experiment "${MODELS[*]}" $N_TRIALS "${VARIANTS[*]}" "${ENCODERS[*]}" "default"
        ;;
        
    "tune")
        echo "Tuning ...."
        echo "==========="
        run_experiment "${MODELS[*]}" $N_TRIALS "${VARIANTS[*]}" "${ENCODERS[*]}" "tune"
        ;;
        
    "all")
        echo "Running both default and tune modes"
        echo "==================================="
        run_experiment "${MODELS[*]}" $N_TRIALS "${VARIANTS[*]}" "${ENCODERS[*]}" "default"
        run_experiment "${MODELS[*]}" $N_TRIALS "${VARIANTS[*]}" "${ENCODERS[*]}" "tune"
        ;;
        
    *)
        echo "Unknown run mode: $RUN_MODE"
        echo "Available modes:"
        echo "  test     - Quick test with default mode only ($TEST_TRIALS trials)"
        echo "  test-all - Quick test with both default and tune modes ($TEST_TRIALS trials each)"
        echo "  default  - Run default hyperparameters (full scale)"
        echo "  tune     - Run hyperparameter tuning (full scale)"
        echo "  all      - Run both default and tune modes (full scale)"
        echo ""
        echo "Usage: $0 [test|test-all|default|tune|all]"
        exit 1
        ;;
esac

# =================
# Post-processing
# =================

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Time: $(date)"
echo ""
echo "Results (tuning only) saved to:"
echo "  - Database: $DB_PATH"
echo "  - Logs: $LOG_DIR/"
echo ""
echo "To view results in Optuna dashboard:"
echo "  optuna-dashboard $DB_PATH"
echo ""