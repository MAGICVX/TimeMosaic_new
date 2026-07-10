# TimeMosaic FusionV3 — ETTh1 quick test
# MoS hierarchical patching + DRoPE + learnable-bias prefix

model_name=TimeMosaic_FusionV3

# pred_len=96, seq_len=96
python -u run.py \
  --task_name Exp_FusionV3 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TimeMosaic_FusionV3 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model 512 \
  --d_ff 2048 \
  --num_latent_token 4 \
  --channel CI \
  --freq_num 4 \
  --mos_levels 3 \
  --mos_max_patch_len 32 \
  --use_drope 1 \
  --use_prefix 1 \
  --prefix_len 4 \
  --num_moe_prefix_experts 4 \
  --pre96 48 \
  --result_file "result_FusionV3.txt" \
  --itr 1

# pred_len=192, seq_len=96
python -u run.py \
  --task_name Exp_FusionV3 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model TimeMosaic_FusionV3 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model 512 \
  --d_ff 2048 \
  --num_latent_token 4 \
  --channel CI \
  --freq_num 4 \
  --mos_levels 3 \
  --mos_max_patch_len 32 \
  --use_drope 1 \
  --use_prefix 1 \
  --prefix_len 4 \
  --num_moe_prefix_experts 4 \
  --pre192 48 \
  --result_file "result_FusionV3.txt" \
  --itr 1

# pred_len=336, seq_len=96
python -u run.py \
  --task_name Exp_FusionV3 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model TimeMosaic_FusionV3 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model 512 \
  --d_ff 2048 \
  --num_latent_token 4 \
  --channel CI \
  --freq_num 4 \
  --mos_levels 3 \
  --mos_max_patch_len 32 \
  --use_drope 1 \
  --use_prefix 1 \
  --prefix_len 4 \
  --num_moe_prefix_experts 4 \
  --pre336 48 \
  --result_file "result_FusionV3.txt" \
  --itr 1

# pred_len=720, seq_len=96
python -u run.py \
  --task_name Exp_FusionV3 \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model TimeMosaic_FusionV3 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model 512 \
  --d_ff 2048 \
  --num_latent_token 4 \
  --channel CI \
  --freq_num 4 \
  --mos_levels 3 \
  --mos_max_patch_len 32 \
  --use_drope 1 \
  --use_prefix 1 \
  --prefix_len 4 \
  --num_moe_prefix_experts 4 \
  --pre720 48 \
  --result_file "result_FusionV3.txt" \
  --itr 1
