torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      main.py --bsize 64 --vit-folder ./pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth \
      --epoch 301 --resume --interpolate --lambda_gt 1.0 --interpolate_rate 0.3 \
      --grad-cum 2 --exp_name ReDi0 \
      --train_sch linear