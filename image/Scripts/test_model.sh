torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      main.py --bsize 50 --vit-folder $1 \
      --epoch $2 --resume --interpolate --lambda_gt 1.0 --interpolate_rate 0.3 \
      --grad-cum 2 --step $3 --cfg_w $4 --exp_name $5 \
      --train_sch arccos  --dropout 0.03 --test-only --test_by_train_data --use_precomputed_stats \
      --r_temp $6 --sm_temp $7 --test_image_num 10000