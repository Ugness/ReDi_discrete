# Text Generation (Base: [DUO](https://github.com/s-sahoo/duo))

In this project, we use OpenWebText dataset.

We used A6000 8 gpus for training and inference.

## Usage

To get started with this project, follow these steps:

1. Install requirement

    ```bash
    # We used docker image with torch==2.3.1+cu121
    pip install -r requirements.txt
    ```

2. Download Pretrained models (of DUO)
    ```bash
    # Finetuned models with ReDi
    # Download from Hugginface(https://huggingface.co/Ugness/ReDi)

    # Or
    # Pretrained models from origin DUO
    # Download origin DUO checkpoint from Google Drive folder(https://drive.google.com/drive/folders/1JpqFM8XRvifwIkjWPfMyuDvu41r1yk0t?usp=share_link).
    ```

3. Use ReDi method
    ```bash
    # Create Rectified Coupling and train a model
    bash scripts/train_owt_duo_reflow_greedy_gen.sh --checkpoint_path "PATH" --ckpt "ReDi1"
    bash scripts/train_owt_duo_reflow_train.sh --checkpoint_path "PATH" --ckpt "ReDi1"

    # Test a model
    bash scripts/gen_ppl_tc_owt_duo.sh --checkpoint_path "PATH" --ckpt "ReDi1" --steps 32
    ```
