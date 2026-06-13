#!/bin/bash

# ============================================================
# ETTm1 MoE 网格调参脚本
# 只调 num_moe_experts 和 num_moe_prefix_experts
# 其他参数和 ETTm1.sh 保持一致
# ============================================================

RESULT_FILE="./result_ETTm1_grid.txt"

MOE_EXPERTS_LIST=(2 3 4 5 6 7 8 9)
PREFIX_EXPERTS_LIST=(1 2 3 4 5 6)

GPU=0
TRAIN_EPOCHS=10
BATCH_SIZE=32
ITR=1

TASKS=(
    "ETTm1_96_96|96|96|1|8|CI"
    "ETTm1_96_192|96|192|2|1|CI"
    "ETTm1_96_336|96|336|1|3|CI"
    "ETTm1_96_720|96|720|1|4|CD"
    "ETTm1_320_96|320|96|1|8|CI"
    "ETTm1_320_192|320|192|2|1|CI"
    "ETTm1_320_336|320|336|1|3|CI"
    "ETTm1_320_720|320|720|1|4|CD"
)

n_e=${#MOE_EXPERTS_LIST[@]}
n_p=${#PREFIX_EXPERTS_LIST[@]}
n_cfg=${#TASKS[@]}
total=$(( n_cfg * n_e * n_p ))
count=0

echo "=== ETTm1 MoE Grid Search: $(date) ==="
echo "num_moe_experts       = ${MOE_EXPERTS_LIST[*]}"
echo "num_moe_prefix_experts= ${PREFIX_EXPERTS_LIST[*]}"
echo "Configs: ${n_cfg}  Total: ${total}"
echo "Output: ${RESULT_FILE}"
echo ""

for task_cfg in "${TASKS[@]}"; do
    IFS='|' read -r model_id seq_len pred_len e_layers n_heads channel <<< "$task_cfg"

    for E in "${MOE_EXPERTS_LIST[@]}"; do
        for pfxE in "${PREFIX_EXPERTS_LIST[@]}"; do
            count=$((count + 1))
            des="E${E}_pfxE${pfxE}"

            echo "[${count}/${total}] ${model_id}  E=${E}, pfxE=${pfxE}"

            python -u run.py \
                --task_name Exp_Fusion \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTm1.csv \
                --model_id "${model_id}_${des}" \
                --model TimeMosaic_Fusion \
                --data ETTm1 \
                --features M \
                --seq_len "$seq_len" \
                --label_len 48 \
                --pred_len "$pred_len" \
                --e_layers "$e_layers" \
                --d_layers 1 \
                --factor 3 \
                --enc_in 7 \
                --dec_in 7 \
                --c_out 7 \
                --des "$des" \
                --n_heads "$n_heads" \
                --channel "$channel" \
                --train_epochs "$TRAIN_EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --gpu "$GPU" \
                --itr "$ITR" \
                --lam_moe 0.001 \
                --prefix_len 4 \
                --lam_prefix_moe 0.001 \
  --freq_num 4 \
  --result_file "result_Fusion.txt" \
                --num_moe_experts "$E" \
                --num_moe_prefix_experts "$pfxE" \

            result_line=$(grep -A1 "${model_id}_${des}" "$RESULT_FILE" | tail -1 | grep 'mse:')
            mse=$(echo "$result_line" | grep -oP '\bmse:\K[0-9.]+' || echo "N/A")
            echo "  => mse=${mse}"
            echo ""
        done
    done
done

echo "=== Done: $(date) ==="
echo "Results: $RESULT_FILE"
