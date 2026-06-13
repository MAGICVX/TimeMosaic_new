#!/bin/bash

# ============================================================
# 退化任务 MoE 网格调参脚本
# 针对 Fusion 表现不如 Baseline 的任务进行调参
# 只调 num_moe_experts 和 num_moe_prefix_experts
# ============================================================

RESULT_FILE="./result/result_degraded_grid.txt"

MOE_EXPERTS_LIST=(2 3 4 5 6 7 8 9)
PREFIX_EXPERTS_LIST=(1 2 3 4 5 6)

GPU=0
TRAIN_EPOCHS=10
BATCH_SIZE=32
ITR=1

# 退化任务列表
# 格式: "model_id|seq_len|pred_len|e_layers|n_heads|channel|data_name|data_path|enc_in"
TASKS=(
    # ETTh2 数据集
    # "ETTh2_96_192|96|192|1|1|CDA|ETTh2|./dataset/ETT-small/|7"
    # "ETTh2_320_336|320|336|1|2|CDA|ETTh2|./dataset/ETT-small/|7"
    
    # ETTm1 数据集
    "ETTm1_320_96|320|96|1|8|CI|ETTm1|./dataset/ETT-small/|7"
    
    # ETTm2 数据集
    "ETTm2_320_96|320|96|2|2|CDA|ETTm2|./dataset/ETT-small/|7"
    
    # ETTh1 数据集
    "ETTh1_96_192|96|192|3|1|CI|ETTh1|./dataset/ETT-small/|7"
    "ETTh1_320_96|320|96|2|8|CI|ETTh1|./dataset/ETT-small/|7"
    
)

n_e=${#MOE_EXPERTS_LIST[@]}
n_p=${#PREFIX_EXPERTS_LIST[@]}
n_cfg=${#TASKS[@]}
total=$(( n_cfg * n_e * n_p ))
count=0

echo "=== Degraded Tasks MoE Grid Search: $(date) ==="
echo "num_moe_experts       = ${MOE_EXPERTS_LIST[*]}"
echo "num_moe_prefix_experts= ${PREFIX_EXPERTS_LIST[*]}"
echo "Configs: ${n_cfg}  Total: ${total}"
echo "Output: ${RESULT_FILE}"
echo ""

for task_cfg in "${TASKS[@]}"; do
    IFS='|' read -r model_id seq_len pred_len e_layers n_heads channel data_name data_path enc_in <<< "$task_cfg"

    for E in "${MOE_EXPERTS_LIST[@]}"; do
        for pfxE in "${PREFIX_EXPERTS_LIST[@]}"; do
            count=$((count + 1))
            des="E${E}_pfxE${pfxE}"

            echo "[${count}/${total}] ${model_id}  E=${E}, pfxE=${pfxE}"

            python -u run.py \
                --task_name Exp_Fusion \
                --is_training 1 \
                --root_path "$data_path" \
                --data_path "${data_name}.csv" \
                --model_id "${model_id}_${des}" \
                --model TimeMosaic_Fusion \
                --data "$data_name" \
                --features M \
                --seq_len "$seq_len" \
                --label_len 48 \
                --pred_len "$pred_len" \
                --e_layers "$e_layers" \
                --d_layers 1 \
                --factor 3 \
                --enc_in "$enc_in" \
                --dec_in "$enc_in" \
                --c_out "$enc_in" \
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