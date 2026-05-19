#!/bin/bash
# =============================================================================
# TimeMosaic Fusion - MoE Grid Search Script
# Only tunes: num_moe_experts, num_moe_prefix_experts
#
# Usage:
#   bash tune_moe.sh                          # Run all datasets
#   bash tune_moe.sh ETTm1 ETTm2              # Run specific datasets
#   bash tune_moe.sh --dry-run ETTm1           # Dry run (print commands only)
#
# Datasets: ETTm1, ETTm2, ETTh1, ETTh2, Exchange, Weather, ECL, Traffic, Solar
# =============================================================================

set -e

# ---- Grid search space ----
MOE_EXPERTS_LIST=(2 3 4 5 6 7 8 9)
PREFIX_EXPERTS_LIST=(1 2 3 4 5)

# ---- Common fixed parameters ----
COMMON="--task_name Exp_Fusion --is_training 1 --model TimeMosaic_Fusion \
  --features M --d_layers 1 --factor 3 --label_len 48 \
  --lam_moe 0.001 --prefix_len 4 --lam_prefix_moe 0.001 --itr 1"

# ---- Dry run mode ----
DRY_RUN=0
DATASETS=()

for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    else
        DATASETS+=("$arg")
    fi
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=( "Weather" "ECL" "Traffic" "Solar")
fi

# ---- Grid search function ----
run_grid() {
    local root_path=$1 data_path=$2 data_type=$3
    local model_id=$4 seq_len=$5 pred_len=$6
    local channel=$7 e_layers=$8 n_heads=$9 enc_in=${10}
    shift 10
    local extra="$@"

    for E in "${MOE_EXPERTS_LIST[@]}"; do
        for pfxE in "${PREFIX_EXPERTS_LIST[@]}"; do
            echo "=========================================="
            echo ">>> ${model_id} | E=${E} pfxE=${pfxE}"
            echo "=========================================="
            if [ $DRY_RUN -eq 1 ]; then
                echo python -u run.py $COMMON \
                    --root_path "$root_path" --data_path "$data_path" --data "$data_type" \
                    --model_id "$model_id" \
                    --seq_len "$seq_len" --pred_len "$pred_len" \
                    --channel "$channel" \
                    --e_layers "$e_layers" --n_heads "$n_heads" \
                    --enc_in "$enc_in" --dec_in "$enc_in" --c_out "$enc_in" \
                    --num_moe_experts "$E" --num_moe_prefix_experts "$pfxE" \
                    $extra
                echo
            else
                python -u run.py $COMMON \
                    --root_path "$root_path" --data_path "$data_path" --data "$data_type" \
                    --model_id "$model_id" \
                    --seq_len "$seq_len" --pred_len "$pred_len" \
                    --channel "$channel" \
                    --e_layers "$e_layers" --n_heads "$n_heads" \
                    --enc_in "$enc_in" --dec_in "$enc_in" --c_out "$enc_in" \
                    --num_moe_experts "$E" --num_moe_prefix_experts "$pfxE" \
                    $extra
            fi
        done
    done
}

# =============================================================================
# Dataset configurations
# =============================================================================

run_ETTm1() {
    local root="./dataset/ETT-small/" file="ETTm1.csv" data="ETTm1" enc=7
    run_grid $root $file $data ETTm1_96_96   96  96  CI  1 8 $enc
    run_grid $root $file $data ETTm1_96_192  96  192 CI  2 1 $enc
    run_grid $root $file $data ETTm1_96_336  96  336 CI  1 3 $enc
    run_grid $root $file $data ETTm1_96_720  96  720 CD  1 4 $enc
    run_grid $root $file $data ETTm1_320_96  320 96  CI  1 8 $enc
    run_grid $root $file $data ETTm1_320_192 320 192 CI  2 1 $enc
    run_grid $root $file $data ETTm1_320_336 320 336 CI  1 3 $enc
    run_grid $root $file $data ETTm1_320_720 320 720 CD  1 4 $enc
}

run_ETTm2() {
    local root="./dataset/ETT-small/" file="ETTm2.csv" data="ETTm2" enc=7
    run_grid $root $file $data ETTm2_96_96   96  96  CDA 2 2  $enc
    run_grid $root $file $data ETTm2_96_192  96  192 CDA 2 1  $enc
    run_grid $root $file $data ETTm2_96_336  96  336 CI  2 16 $enc
    run_grid $root $file $data ETTm2_96_720  96  720 CDA 1 2  $enc
    run_grid $root $file $data ETTm2_320_96  320 96  CDA 2 2  $enc
    run_grid $root $file $data ETTm2_320_192 320 192 CDA 2 1  $enc
    run_grid $root $file $data ETTm2_320_336 320 336 CI  2 16 $enc
    run_grid $root $file $data ETTm2_320_720 320 720 CDA 1 2  $enc
}

run_ETTh1() {
    local root="./dataset/ETT-small/" file="ETTh1.csv" data="ETTh1" enc=7
    run_grid $root $file $data ETTh1_96_96   96  96  CI  2 8  $enc
    run_grid $root $file $data ETTh1_96_192  96  192 CI  3 1  $enc
    run_grid $root $file $data ETTh1_96_336  96  336 CDA 2 16 $enc
    run_grid $root $file $data ETTh1_96_720  96  720 CD  3 2  $enc
    run_grid $root $file $data ETTh1_320_96  320 96  CI  2 8  $enc
    run_grid $root $file $data ETTh1_320_192 320 192 CI  3 1  $enc
    run_grid $root $file $data ETTh1_320_336 320 336 CDA 2 16 $enc
    run_grid $root $file $data ETTh1_320_720 320 720 CD  3 2  $enc
}

run_ETTh2() {
    local root="./dataset/ETT-small/" file="ETTh2.csv" data="ETTh2" enc=7
    run_grid $root $file $data ETTh2_96_96   96  96  CDA 2 1  $enc
    run_grid $root $file $data ETTh2_96_192  96  192 CDA 1 1  $enc
    run_grid $root $file $data ETTh2_96_336  96  336 CDA 1 2  $enc
    run_grid $root $file $data ETTh2_96_720  96  720 CI  3 16 $enc
    run_grid $root $file $data ETTh2_320_96  320 96  CDA 2 1  $enc
    run_grid $root $file $data ETTh2_320_192 320 192 CDA 1 1  $enc
    run_grid $root $file $data ETTh2_320_336 320 336 CDA 1 2  $enc
    run_grid $root $file $data ETTh2_320_720 320 720 CI  3 16 $enc
}

run_Exchange() {
    local root="./dataset/exchange_rate/" file="exchange_rate.csv" data="custom" enc=8
    run_grid $root $file $data Exchange_96_96   96  96  CI  2 4  $enc
    run_grid $root $file $data Exchange_96_192  96  192 CDA 2 4  $enc
    run_grid $root $file $data Exchange_96_336  96  336 CDA 2 4  $enc
    run_grid $root $file $data Exchange_96_720  96  720 CDA 2 4  $enc
    run_grid $root $file $data Exchange_320_96  320 96  CI  3 16 $enc
    run_grid $root $file $data Exchange_320_192 320 192 CDA 3 1  $enc
    run_grid $root $file $data Exchange_320_336 320 336 CDA 1 4  $enc
    run_grid $root $file $data Exchange_320_720 320 720 CDA 2 2  $enc
}

run_Weather() {
    local root="./dataset/weather/" file="weather.csv" data="custom" enc=21
    run_grid $root $file $data weather_96_96   96  96  CDA 3 1 $enc "--train_epochs 10"
    run_grid $root $file $data weather_96_192  96  192 CDA 3 4 $enc "--train_epochs 10"
    run_grid $root $file $data weather_96_336  96  336 CDA 2 8 $enc "--train_epochs 10"
    run_grid $root $file $data weather_96_720  96  720 CDA 2 8 $enc "--train_epochs 10"
    run_grid $root $file $data weather_320_96  320 96  CDA 3 1 $enc "--train_epochs 10"
    run_grid $root $file $data weather_320_192 320 192 CDA 3 4 $enc "--train_epochs 10"
    run_grid $root $file $data weather_320_336 320 336 CDA 2 8 $enc "--train_epochs 10"
    run_grid $root $file $data weather_320_720 320 720 CDA 2 8 $enc "--train_epochs 10"
}

run_ECL() {
    local root="./dataset/electricity/" file="electricity.csv" data="custom" enc=321
    local extra="--batch_size 16 --use_multi_gpu --devices 0,1,2,3"
    run_grid $root $file $data ECL_96_96  96 96  CDP 2 8 $enc "$extra"
    run_grid $root $file $data ECL_96_192 96 192 CDP 2 8 $enc "$extra"
    run_grid $root $file $data ECL_96_336 96 336 CI+ 2 8 $enc "$extra"
    run_grid $root $file $data ECL_96_720 96 720 CDP 2 8 $enc "$extra"
}

run_Traffic() {
    local root="./dataset/traffic/" file="traffic.csv" data="custom" enc=862
    local extra="--d_model 512 --d_ff 512 --batch_size 4 --use_multi_gpu --devices 0,1"
    run_grid $root $file $data traffic_96_96  96 96  CDP 2 8 $enc "$extra"
    run_grid $root $file $data traffic_96_192 96 192 CDP 2 8 $enc "$extra"
    run_grid $root $file $data traffic_96_336 96 336 CDP 2 8 $enc "$extra"
    run_grid $root $file $data traffic_96_720 96 720 CDP 2 8 $enc "$extra"
}

run_Solar() {
    local root="./dataset/Solar/" file="solar_AL.txt" data="Solar" enc=137
    local extra="--d_model 512 --d_ff 2048 --label_len 0 --learning_rate 0.001 --train_epochs 10 --patience 3 --use_multi_gpu --devices 0,1"
    run_grid $root $file $data solar_96_96  96 96  CD 3 8 $enc "$extra"
    run_grid $root $file $data solar_96_192 96 192 CD 3 8 $enc "$extra"
    run_grid $root $file $data solar_96_336 96 336 CD 3 8 $enc "$extra"
    run_grid $root $file $data solar_96_720 96 720 CI 3 8 $enc "$extra"
}

# =============================================================================
# Main
# =============================================================================

echo "============================================================"
echo " TimeMosaic Fusion - MoE Grid Search"
echo " num_moe_experts:      ${MOE_EXPERTS_LIST[*]}"
echo " num_moe_prefix_experts: ${PREFIX_EXPERTS_LIST[*]}"
echo " Datasets:             ${DATASETS[*]}"
echo " Dry run:              $DRY_RUN"
echo "============================================================"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo ">>>>>>>>>> Dataset: $ds <<<<<<<<<<"
    case $ds in
        ETTm1)    run_ETTm1    ;;
        ETTm2)    run_ETTm2    ;;
        ETTh1)    run_ETTh1    ;;
        ETTh2)    run_ETTh2    ;;
        Exchange) run_Exchange ;;
        Weather)  run_Weather  ;;
        ECL)      run_ECL      ;;
        Traffic)  run_Traffic  ;;
        Solar)    run_Solar    ;;
        *)        echo "WARNING: Unknown dataset '$ds', skipping" ;;
    esac
done

echo ""
echo "============================================================"
echo " Grid search complete!"
echo "============================================================"
