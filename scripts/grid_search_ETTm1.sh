#!/bin/bash

# 3D Grid search for ETTm1 (seq_len=96)
# Usage: bash scripts/grid_search_ETTm1.sh

GPU=0
MODEL="TimeMosaic_MIM"
TASK="TimeMosaic_MIM"
RESULT_FILE="./result_grid_3d.txt"

LAM_ENTROPY_LIST=("0.01" "0.03" "0.05")
LAM_DIVERSITY_LIST=("0.004" "0.008" "0.012")
LAM_ORTHOGONAL_LIST=("0.0" "0.005" "0.01" "0.02" "0.05")

DATASETS=(
  "ETTm1_96_96|./dataset/ETT-small/|ETTm1.csv|ETTm1|7|96|96|8|1|CI"
  "ETTm1_96_192|./dataset/ETT-small/|ETTm1.csv|ETTm1|7|96|192|1|2|CI"
  "ETTm1_96_336|./dataset/ETT-small/|ETTm1.csv|ETTm1|7|96|336|3|1|CI"
  "ETTm1_96_720|./dataset/ETT-small/|ETTm1.csv|ETTm1|7|96|720|4|1|CD"
)

LABEL_LEN=48
FEATURES="M"
FACTOR=3
D_LAYERS=1
TRAIN_EPOCHS=10
BATCH_SIZE=32
ITR=1

n_e=${#LAM_ENTROPY_LIST[@]}
n_d=${#LAM_DIVERSITY_LIST[@]}
n_o=${#LAM_ORTHOGONAL_LIST[@]}
n_cfg=${#DATASETS[@]}
total=$(( n_cfg * n_e * n_d * n_o ))
count=0

echo "=== 3D Grid Search ETTm1: $(date) ==="
echo "lam_entropy   = ${LAM_ENTROPY_LIST[*]}"
echo "lam_diversity = ${LAM_DIVERSITY_LIST[*]}"
echo "lam_orthogonal= ${LAM_ORTHOGONAL_LIST[*]}"
echo "Configs: ${n_cfg}  Total: ${total}"
echo "Output: ${RESULT_FILE}"
echo ""

for dataset_cfg in "${DATASETS[@]}"; do
  IFS='|' read -r dname root data_path data_type enc_in seq_len pred_len n_heads e_layers channel <<< "$dataset_cfg"

  for lam_e in "${LAM_ENTROPY_LIST[@]}"; do
    for lam_d in "${LAM_DIVERSITY_LIST[@]}"; do
      for lam_o in "${LAM_ORTHOGONAL_LIST[@]}"; do
        count=$((count + 1))

        model_id="GS3D_${dname}_e${lam_e}_d${lam_d}_o${lam_o}"
        des="e${lam_e}_d${lam_d}_o${lam_o}"

        echo "[${count}/${total}] ${dname}  e=${lam_e} d=${lam_d} o=${lam_o}"

        python -u run.py \
          --task_name "$TASK" \
          --is_training 1 \
          --root_path "$root" \
          --data_path "$data_path" \
          --model_id "$model_id" \
          --model "$MODEL" \
          --data "$data_type" \
          --features "$FEATURES" \
          --seq_len "$seq_len" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --e_layers "$e_layers" \
          --d_layers "$D_LAYERS" \
          --factor "$FACTOR" \
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
          --lam_entropy "$lam_e" \
          --lam_diversity "$lam_d" \
          --lam_orthogonal "$lam_o" \
          --result_file "$RESULT_FILE"

        result_line=$(grep -A1 "$model_id" "$RESULT_FILE" | tail -1 | grep 'mse:')
        mse=$(echo "$result_line" | grep -oP '\bmse:\K[\d.]+')
        echo "  => mse=${mse:-N/A}"
        echo ""
      done
    done
  done
done

echo "=== Done: $(date) ==="

python3 << PYEOF
import re
with open('${RESULT_FILE}') as f:
    lines = f.readlines()
results = {}
for i, line in enumerate(lines):
    m = re.match(r'.*GS3D_(ETTm1_\d+_\d+)_e([\d.]+)_d([\d.]+)_o([\d.]+)_', line)
    if m:
        cfg, lam_e, lam_d, lam_o = m.group(1), m.group(2), m.group(3), m.group(4)
        nl = lines[i+1] if i+1 < len(lines) else ''
        mse_m = re.search(r'\bmse:([\d.]+)', nl)
        if mse_m:
            results.setdefault(cfg, []).append((lam_e, lam_d, lam_o, float(mse_m.group(1))))
print("\n=== Best per config ===")
for cfg in sorted(results):
    best = sorted(results[cfg], key=lambda x: x[3])[0]
    e, d, o, mse = best
    print(f"{cfg:<20} e={e} d={d} o={o}  mse={mse:.4f}")
PYEOF
