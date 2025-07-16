torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      main.py --bsize 64 --vit-folder $1 \
      --epoch $2 --resume --interpolate --lambda_gt 1.0 --interpolate_rate 0.3 \
      --grad-cum 2 --step $3 --data_load --cfg_w $4 --exp_name $5 \
      --train_sch linear --dropout 0.03 --lr_cosine --load_optimizer_states