# Image Generation (Base: [Halton-MaskGIT(MaskGIT-Pytorch)](https://github.com/valeoai/Halton-MaskGIT/tree/v1.0))

In this project, we used ImageNet dataset.

We used A6000 4 gpus for training and inference.

## Usage

To get started with this project, follow these steps:

1. Install requirement 
   ```bash
   # We used docker image with torch==2.5.1+cu121
   pip install -r requirements.txt
   ```

2. Setting the pretrained checkpoints and ImageNet dataset
   ```bash
   # 1. Download pretrained VQGAN models 
   # If you want to finetune the model, you need to uncomment the 
   # "hf_hub_download(repo_id="llvictorll/Maskgit-pytorch", filename="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth", local_dir=".")"
   # line to download original MaskGIT model
   python download_models.py

   # 2. Download finetuned models with ReDi from Google Drive
   # Set with file structure like 'pretrained_maskgit/MaskGIT/redi1.ckpt' or 'pretrained_maskgit/MaskGIT/redi2.ckpt'
   # Download from https://huggingface.co/Ugness/ReDi

   # 3. Make link or download ImageNet for train and test
   ln -s ~/ImageNet/train/
   ln -s ~/ImageNet/val/
   
   # 4. Make link or download VIRTUAL_imagenet256_labeled.npz on image folder
   ln -s ~/VIRTUAL_imagenet256_labeled.npz

   # 5. Download fid_stats_imagenet256_guided_diffusion.npz from https://github.com/openai/guided-diffusion/tree/main/evaluations
   ln -s ~/fid_stats_imagenet256_guided_diffusion.npz
   ```

3. Use ReDi method
   ``` bash
   # Finetune MaskGIT with Stochastic Initial States
   bash Scripts/finetune_model.sh

   # Create Rectified Coupling and Train a model
   bash Scripts/create_rectified_dataset.sh ./pretrained_maskgit/MaskGIT/ReDi0.ckpt 401 16 1.0 ReDi1
   bash Scripts/train_model.sh ./pretrained_maskgit/MaskGIT/ReDi0.ckpt 401 16 1.0 ReDi1
   
   # Test a model
   # Should set the right condition depending on each checkpoints
   bash Scripts/test_model.sh ./pretrained_maskgit/MaskGIT/ReDi1.ckpt 401 4 6.0 ReDi1 4.5 1.0
   ```

## Results
| Ckpt | Step | CFG | r_temp | sm_temp | FID | Inception_score |
|------|------|-----|--------|---------|-----|-----------------|
| ReDi1 | 4 | 6.0 | 4.5 | 1.0 | 7.515497 | 228.104477 |
| ReDi2 | 4 | 4.0 | 4.5 | 1.0 | 7.859749 | 240.29361 |
| ReDi3-distilled | 1 | 1.0 | 4.5 | 2.0 | 11.676962 | 181.790146 |
