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
RESULTS_DIR="results"

# Create directories if they don't exist
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR
mkdir -p "best_config"

# Experiment settings
N_TRIALS=100  
MODELS=("lightgbm" "xgboost" "rf" "extratrees")
VARIANTS=("main" "climate_7d" "climate_14d" "main_geo" "climate_7d_geo" "climate_14d_geo")
ENCODERS=("onehot" "target" "catboost")
EVAL_METHODS=("cv" "validation")

# ============================================================================
# Helper Functions
# ============================================================================

run_experiment() {
    local model=$1
    local n_trials=$2
    local variants=$3
    local encoders=$4
    local eval_methods=$5
    
    echo "============================================================"
    echo "Starting experiment: $model"
    echo "Variants: $variants"
    echo "Encoders: $encoders"
    echo "Eval methods: $eval_methods"
    echo "Trials: $n_trials"
    echo "Time: $(date)"
    echo "============================================================"
    
    python $SCRIPT_PATH \
        --model $model \
        --model-variants $variants \
        --encoders $encoders \
        --eval-methods $eval_methods \
        --n-trials $n_trials \
        --data-path $DATA_PATH \
        --config $CONFIG_PATH \
        --storage-url $DB_PATH \
        2>&1 | tee -a "$LOG_DIR/${model}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Completed: $model at $(date)"
    echo ""
}

# ============================================================================
# Main Execution Options
# ============================================================================

# Check command line argument for run mode
RUN_MODE=${1:-"all"}

case $RUN_MODE in
    "quick")
        echo "Running QUICK TEST (10 trials, main variant only)"
        echo "================================================"
        for model in "${MODELS[@]}"; do
            run_experiment $model 10 "main" "catboost" "validation"
        done
        ;;
    
    "main_only")
        echo "Running MAIN VARIANT ONLY (all models, 100 trials)"
        echo "=================================================="
        for model in "${MODELS[@]}"; do
            run_experiment $model $N_TRIALS "main main_geo" "onehot target catboost" "cv validation"
        done
        ;;
    
    "climate_only")
        echo "Running CLIMATE VARIANTS ONLY (all models, 100 trials)"
        echo "======================================================="
        for model in "${MODELS[@]}"; do
            run_experiment $model $N_TRIALS "climate_7d climate_14d climate_7d_geo climate_14d_geo" "onehot target catboost" "cv validation"
        done
        ;;
    
    "by_model")
        echo "Running ONE MODEL AT A TIME (all variants, 100 trials)"
        echo "======================================================"
        for model in "${MODELS[@]}"; do
            echo ">>> Processing $model with all variants <<<"
            run_experiment $model $N_TRIALS "main climate_7d climate_14d main_geo climate_7d_geo climate_14d_geo" "onehot target catboost" "cv validation"
            echo ">>> Completed $model, moving to next model <<<"
            sleep 5  # Brief pause between models
        done
        ;;
    
    "boosting_only")
        echo "Running BOOSTING MODELS ONLY (LightGBM & XGBoost)"
        echo "================================================="
        for model in "lightgbm" "xgboost"; do
            run_experiment $model $N_TRIALS "main climate_7d climate_14d main_geo climate_7d_geo climate_14d_geo" "onehot target catboost" "cv validation"
        done
        ;;
    
    "trees_only")
        echo "Running TREE MODELS ONLY (RF & ExtraTrees)"
        echo "=========================================="
        for model in "rf" "extratrees"; do
            run_experiment $model $N_TRIALS "main climate_7d climate_14d main_geo climate_7d_geo climate_14d_geo" "onehot target catboost" "cv validation"
        done
        ;;
    
    "cv_only")
        echo "Running CV EVALUATION ONLY (all models)"
        echo "======================================="
        for model in "${MODELS[@]}"; do
            run_experiment $model $N_TRIALS "main climate_7d climate_14d main_geo climate_7d_geo climate_14d_geo" "onehot target catboost" "cv"
        done
        ;;
    
    "validation_only")
        echo "Running VALIDATION EVALUATION ONLY (all models)"
        echo "==============================================="
        for model in "${MODELS[@]}"; do
            run_experiment $model $N_TRIALS "main climate_7d climate_14d main_geo climate_7d_geo climate_14d_geo" "onehot target catboost" "validation"
        done
        ;;
    
    "custom")
        # For custom single experiment - edit as needed
        echo "Running CUSTOM EXPERIMENT"
        echo "========================"
        run_experiment "lightgbm" 10 "main" "catboost" "validation"
        ;;
    
    "all"|*)
        echo "Running ALL EXPERIMENTS (all models, all variants, 100 trials each)"
        echo "=================================================================="
        echo "WARNING: This will take a long time!"
        echo "Starting in 10 seconds... (Ctrl+C to cancel)"
        sleep 10
        
        for model in "${MODELS[@]}"; do
            echo ""
            echo "##################################################"
            echo "# Processing Model: $model"
            echo "##################################################"
            run_experiment $model $N_TRIALS "main climate_7d climate_14d" "onehot target catboost" "cv validation"
            
            # Optional: pause between models to check results
            if [ "$model" != "extratrees" ]; then
                echo "Pausing for 30 seconds before next model..."
                sleep 30
            fi
        done
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
echo "  - Database: optuna_studies.db"
echo "  - Logs: $LOG_DIR/"
echo "  - Results: $RESULTS_DIR/"
echo ""
echo "To view results in Optuna dashboard:"
echo "  optuna-dashboard $DB_PATH"
echo ""

