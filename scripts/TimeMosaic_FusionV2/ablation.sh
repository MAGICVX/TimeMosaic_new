# TimeMosaic FusionV2 — Ablation Study
# Disable individual V2 components to measure their contribution
model_name=TimeMosaic_FusionV2

# === Baseline: Full V2 (DRoPE + LearableBias + LogDecay) ===
python -u run.py \
  --task_name Exp_FusionV2 \
  --is_training 1 \
  --gpu 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96_V2_full \
  --model TimeMosaic_FusionV2 \
  --data Solar \
  --features M \
  --channel CD \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --n_heads 2 \
  --moe_bias_rate 0.001 \
  --prefix_len 4 \
  --freq_num 4 \
  --use_drope 1 \
  --use_log_decay 1 \
  --result_file "result_FusionV2_ablation.txt" \
  --d_model 128 --d_ff 256 \
  --learning_rate 0.001 \
  --train_epochs 10 --patience 3 --batch_size 16 --itr 1

# === Ablation 1: No DRoPE ===
python -u run.py \
  --task_name Exp_FusionV2 \
  --is_training 1 \
  --gpu 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96_V2_noDRoPE \
  --model TimeMosaic_FusionV2 \
  --data Solar \
  --features M \
  --channel CD \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --n_heads 2 \
  --moe_bias_rate 0.001 \
  --prefix_len 4 \
  --freq_num 4 \
  --use_drope 0 \
  --use_log_decay 1 \
  --result_file "result_FusionV2_ablation.txt" \
  --d_model 128 --d_ff 256 \
  --learning_rate 0.001 \
  --train_epochs 10 --patience 3 --batch_size 16 --itr 1

# === Ablation 2: No LogDecay (uniform time weights) ===
python -u run.py \
  --task_name Exp_FusionV2 \
  --is_training 1 \
  --gpu 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96_V2_noLogDecay \
  --model TimeMosaic_FusionV2 \
  --data Solar \
  --features M \
  --channel CD \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --n_heads 2 \
  --moe_bias_rate 0.001 \
  --prefix_len 4 \
  --freq_num 4 \
  --use_drope 1 \
  --use_log_decay 0 \
  --result_file "result_FusionV2_ablation.txt" \
  --d_model 128 --d_ff 256 \
  --learning_rate 0.001 \
  --train_epochs 10 --patience 3 --batch_size 16 --itr 1

# === Ablation 3: Both off (closest to V1 baseline) ===
python -u run.py \
  --task_name Exp_FusionV2 \
  --is_training 1 \
  --gpu 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96_V2_minimal \
  --model TimeMosaic_FusionV2 \
  --data Solar \
  --features M \
  --channel CD \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --n_heads 2 \
  --moe_bias_rate 0.001 \
  --prefix_len 4 \
  --freq_num 4 \
  --use_drope 0 \
  --use_log_decay 0 \
  --result_file "result_FusionV2_ablation.txt" \
  --d_model 128 --d_ff 256 \
  --learning_rate 0.001 \
  --train_epochs 10 --patience 3 --batch_size 16 --itr 1
