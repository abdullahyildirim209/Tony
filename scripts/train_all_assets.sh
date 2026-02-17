#!/usr/bin/env bash
# Train walk-forward models and run statistical analysis for all 8 assets.
# Run from tony/: bash scripts/train_all_assets.sh
set -e

SYMBOLS=(
    BTCUSDT ETHUSDT BNBUSDT ADAUSDT
    DOGEUSDT DOTUSDT AVAXUSDT ATOMUSDT
)

for symbol in "${SYMBOLS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Training: $symbol"
    echo "============================================================"
    .venv/bin/python experiments/walk_forward.py --symbol "$symbol"

    echo ""
    echo "  Statistical analysis: $symbol"
    echo "------------------------------------------------------------"
    .venv/bin/python experiments/statistical_analysis.py --symbol "$symbol"
done

echo ""
echo "============================================================"
echo "  All ${#SYMBOLS[@]} assets trained and analyzed."
echo "============================================================"
