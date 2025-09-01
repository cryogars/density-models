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

# Quick test settings
QUICK_MODELS=("rf")
QUICK_VARIANTS=("main")
QUICK_ENCODERS=("onehot")
QUICK_TRIALS=10

# Boosting models
BOOSTING_MODELS=("lightgbm" "xgboost")
BOOSTING_VARIANTS=("main" "climate_7d" "climate_14d")
BOOSTING_ENCODERS=("onehot" "target" "catboost")

# Sklearn models
SKLEARN_MODELS=("rf" "extratrees")
SKLEARN_VARIANTS=("main" "climate_7d" "climate_14d")
SKLEARN_ENCODERS=("onehot" "target" "catboost")

# ============================================================================
# Helper Functions
# ============================================================================

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

# ============================================================================
# Main Execution Options
# ============================================================================

# Check command line argument for run mode
RUN_MODE=${1:-"all"}

case $RUN_MODE in
    "quick")
        echo "Running QUICK test mode"
        echo "======================"
        echo "Using: ${QUICK_MODELS[*]} | ${QUICK_VARIANTS[*]} | ${QUICK_ENCODERS[*]} | $QUICK_TRIALS trials"
        echo ""
        run_experiment "${QUICK_MODELS[*]}" $QUICK_TRIALS "${QUICK_VARIANTS[*]}" "${QUICK_ENCODERS[*]}" "tune"
        ;;

    "default")
        echo "--- Running DEFAULT hyperparameters ---"
        echo "======================================="
        run_experiment "${MODELS[*]}" $N_TRIALS "${VARIANTS[*]}" "${ENCODERS[*]}" "default"
        ;;

    "boosting")
        echo "Running BOOSTING models (LightGBM & XGBoost)"
        echo "==========================================="
        echo "Using: ${BOOSTING_MODELS[*]} | ${BOOSTING_VARIANTS[*]} | ${BOOSTING_ENCODERS[*]} | $N_TRIALS trials"
        echo ""
        run_experiment "${BOOSTING_MODELS[*]}" $N_TRIALS "${BOOSTING_VARIANTS[*]}" "${BOOSTING_ENCODERS[*]}" "tune"
        ;;

    "sklearn")
        echo "Running SKLEARN models (Random Forest & Extra Trees)"
        echo "===================================================="
        echo "Using: ${SKLEARN_MODELS[*]} | ${SKLEARN_VARIANTS[*]} | ${SKLEARN_ENCODERS[*]} | $N_TRIALS trials"
        echo ""
        run_experiment "${SKLEARN_MODELS[*]}" $N_TRIALS "${SKLEARN_VARIANTS[*]}" "${SKLEARN_ENCODERS[*]}" "tune"
        ;;
        
    "all")
        echo "Running ALL models - both default and tuned hyperparameters"
        echo "=========================================================="
        echo ""
        echo "--- Running DEFAULT hyperparameters ---"
        run_experiment "${MODELS[*]}" $N_TRIALS "${VARIANTS[*]}" "${ENCODERS[*]}" "default"
        echo ""
        echo "--- Running TUNED hyperparameters ---"
        run_experiment "${MODELS[*]}" $N_TRIALS "${VARIANTS[*]}" "${ENCODERS[*]}" "tune"
        ;;
        
    *)
        echo "Unknown run mode: $RUN_MODE"
        echo "Available modes:"
        echo "  quick    - Quick test: RF + main + onehot ($QUICK_TRIALS trials, tune only)"
        echo "  boosting - LightGBM & XGBoost with all variants ($N_TRIALS trials, tune only)"
        echo "  sklearn  - Random Forest & Extra Trees with all variants ($N_TRIALS trials, tune only)"
        echo "  all      - All models with all variants ($N_TRIALS trials, both default AND tuned)"
        echo ""
        echo "Usage: $0 [quick|boosting|sklearn|all]"
        echo ""
        echo "Note: 'all' mode runs both default hyperparameters and hyperparameter tuning"
        echo "      for complete baseline vs optimized comparison."
        exit 1
        ;;
esac

# ============================================================================
# Post-processing
# ============================================================================

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "============================================================"
echo "Time: $(date)"
echo ""
echo "Results saved to:"
echo "  - Database: $DB_PATH"
echo "  - Logs: $LOG_DIR/"
echo ""
echo "To view results in Optuna dashboard:"
echo "  optuna-dashboard $DB_PATH"
echo ""