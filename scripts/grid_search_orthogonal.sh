#!/bin/bash

# Grid search: scan lam_orthogonal with fixed best entropy/diversity
# Usage: bash scripts/grid_search_orthogonal.sh

GPU=0
MODEL="TimeMosaic"
TASK="TimeMosaic"
RESULT_FILE="./result_grid_orthogonal.txt"
SUMMARY_FILE="./result_grid_orthogonal_summary.txt"

# Best known values from previous tuning
LAM_ENTROPY=0.03
LAM_DIVERSITY=0.008

# Orthogonal loss values to scan
LAM_ORTHOGONAL_LIST=("0.0" "0.001" "0.005" "0.01" "0.02" "0.05" "0.1")

# Dataset configs: name, root, data_path, data_type, enc_in, seq_len, pred_len, n_heads, e_layers, channel
# Format: "name|root|data_path|data_type|enc_in|seq_len|pred_len|n_heads|e_layers|channel"
DATASETS=(
  # seq_len=96
  "ETTh1_96_96|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|96|96|8|2|CI"
  "ETTh1_96_192|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|96|192|1|3|CI"
  "ETTh1_96_336|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|96|336|16|2|CDA"
  "ETTh1_96_720|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|96|720|2|3|CD"
  # seq_len=320
  "ETTh1_320_96|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|320|96|8|2|CI"
  "ETTh1_320_192|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|320|192|1|3|CI"
  "ETTh1_320_336|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|320|336|16|2|CDA"
  "ETTh1_320_720|./dataset/ETT-small/|ETTh1.csv|ETTh1|7|320|720|2|3|CD"
)

LABEL_LEN=48
FEATURES="M"
FACTOR=3
D_LAYERS=1
TRAIN_EPOCHS=10
BATCH_SIZE=32
ITR=1

echo "=== Grid Search Orthogonal Loss: $(date) ===" | tee "$SUMMARY_FILE"
echo "Fixed: lam_entropy=$LAM_ENTROPY  lam_diversity=$LAM_DIVERSITY" | tee -a "$SUMMARY_FILE"
echo "Scanning: lam_orthogonal = ${LAM_ORTHOGONAL_LIST[*]}" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

total=$(( ${#DATASETS[@]} * ${#LAM_ORTHOGONAL_LIST[@]} ))
count=0

for dataset_cfg in "${DATASETS[@]}"; do
  IFS='|' read -r dname root data_path data_type enc_in seq_len pred_len n_heads e_layers channel <<< "$dataset_cfg"

  for lam_orth in "${LAM_ORTHOGONAL_LIST[@]}"; do
    count=$((count + 1))

    model_id="GS_${dname}_orth${lam_orth}"
    des="orth${lam_orth}"

    echo "[${count}/${total}] ${dname} lam_orthogonal=${lam_orth}"

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
      --lam_entropy "$LAM_ENTROPY" \
      --lam_diversity "$LAM_DIVERSITY" \
      --lam_orthogonal "$lam_orth" \
      --result_file "$RESULT_FILE"

    # Extract mse/mae from result file (metrics line follows the setting line)
    result_line=$(grep -A1 "$model_id" "$RESULT_FILE" | tail -1)
    mse=$(echo "$result_line" | grep -oP 'mse:\K[\d.]+')
    mae=$(echo "$result_line" | grep -oP 'mae:\K[\d.]+')

    printf "  %-16s lam_orth=%-5s  mse=%-10s mae=%-10s\n" "$dname" "$lam_orth" "${mse:-N/A}" "${mae:-N/A}" | tee -a "$SUMMARY_FILE"
    echo ""
  done
done

echo "" | tee -a "$SUMMARY_FILE"
echo "=== Done: $(date) ===" | tee -a "$SUMMARY_FILE"
echo "Grid search results: $RESULT_FILE" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"

# Print best per config (sorted by mse)
echo "" | tee -a "$SUMMARY_FILE"
echo "=== Best per config ===" | tee -a "$SUMMARY_FILE"
for dataset_cfg in "${DATASETS[@]}"; do
  IFS='|' read -r dname root data_path data_type enc_in seq_len pred_len n_heads e_layers channel <<< "$dataset_cfg"
  echo "--- ${dname} ---" | tee -a "$SUMMARY_FILE"
  grep -A1 "GS_${dname}_orth" "$RESULT_FILE" | grep 'mse:' | \
    sort -t':' -k2 -n | while read line; do
      mse=$(echo "$line" | grep -oP 'mse:\K[\d.]+')
      mae=$(echo "$line" | grep -oP 'mae:\K[\d.]+')
      # get lam_o from previous line
      prev=$(grep -B1 "mse:${mse}" "$RESULT_FILE" | head -1)
      lam_o=$(echo "$prev" | grep -oP 'lam_o=\K[\d.]+')
      printf "  lam_o=%-6s mse=%-10s mae=%-10s\n" "${lam_o:-?}" "${mse:-N/A}" "${mae:-N/A}" | tee -a "$SUMMARY_FILE"
    done
done
