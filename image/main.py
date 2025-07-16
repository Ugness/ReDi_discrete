# Main file to launch training or evaluation
import os
import random

import numpy as np
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group

from Trainer.vit import MaskGIT


def main(args):
    """ Main function:Train or eval MaskGIT """
    maskgit = MaskGIT(args)

    if args.test_only:  # Evaluate the networks
        maskgit.eval()

    elif args.nelbo_test:
        if args.data_load:
            loss_hist, nelbo_hist, cnt = maskgit.validate_one_epoch_reflow(ign_gt=True)
        else:
            loss_hist, nelbo_hist, cnt = maskgit.validate_one_epoch(ign_gt=True)
        
        save_path = os.path.join('results', maskgit.run_id, 'nelbo_hist.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save((loss_hist, nelbo_hist, cnt), save_path)
        print(f"nelbo_hist saved in {save_path}")

    elif args.debug:  # custom code for testing inference
        import torchvision.utils as vutils
        from torchvision.utils import save_image
        with torch.no_grad():
            labels, name = [1, 7, 282, 604, 724, 179, 681, 367, 635, random.randint(0, 999)] * 1, "r_row"
            labels = torch.LongTensor(labels).to(args.device)
            sm_temp = 1.3          # Softmax Temperature
            r_temp = 7             # Gumbel Temperature
            w = 9                  # Classifier Free Guidance
            randomize = "linear"   # Noise scheduler
            step = 32              # Number of step
            sched_mode = "arccos"  # Mode of the scheduler
            # Generate sample
            gen_sample, _, _ = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, r_temp=r_temp, w=w,
                                              randomize=randomize, sched_mode=sched_mode, step=step)
            gen_sample = vutils.make_grid(gen_sample, nrow=5, padding=2, normalize=True)
            # Save image
            save_image(gen_sample, f"saved_img/sched_{sched_mode}_step={step}_temp={sm_temp}"
                                   f"_w={w}_randomize={randomize}_{name}.jpg")
    elif args.data_gen:
        pass
    else:  # Begin training
        maskgit.fit()


def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",     type=str,   default=None,       help="EXP NAME")
    parser.add_argument("--data",         type=str,   default="imagenet", help="dataset on which dataset to train")
    parser.add_argument("--data-folder",  type=str,   default="",         help="folder containing the dataset")
    parser.add_argument("--vqgan-folder", type=str,   default="./pretrained_maskgit/VQGAN/",         help="folder of the pretrained VQGAN")
    parser.add_argument("--vit-size",     type=str,   default="base",     help="size of the vit")
    parser.add_argument("--vit-folder",   type=str,   default="",         help="folder where to save the Transformer")
    parser.add_argument("--writer-log",   type=str,   default="./logs/",         help="folder where to store the logs")
    parser.add_argument("--sched_mode",   type=str,   default="arccos",   help="scheduler mode whent sampling")
    parser.add_argument("--train_sch",    type=str,   default="linear",   help="scheduler mode whent sampling")
    parser.add_argument("--grad-cum",     type=int,   default=1,          help="accumulate gradient")
    parser.add_argument('--channel',      type=int,   default=3,          help="rgb or black/white image")
    parser.add_argument("--num_workers",  type=int,   default=4,          help="number of workers")
    parser.add_argument("--step",         type=int,   default=8,          help="number of step for sampling")
    parser.add_argument('--seed',         type=int,   default=42,         help="fix seed")
    parser.add_argument("--epoch",        type=int,   default=300,        help="number of epoch")
    parser.add_argument('--img-size',     type=int,   default=256,        help="image size")
    parser.add_argument("--bsize",        type=int,   default=256,        help="batch size")
    parser.add_argument("--mask-value",   type=int,   default=1024,       help="number of epoch")
    parser.add_argument("--warm_up",      type=int,   default=0,          help="lr warmup")
    parser.add_argument("--lr",           type=float, default=1e-4,       help="learning rate to train the transformer")
    parser.add_argument("--cfg_w",        type=float, default=3,          help="classifier free guidance wight")
    parser.add_argument("--r_temp",       type=float, default=4.5,        help="Gumbel noise temperature when sampling")
    parser.add_argument("--sm_temp",      type=float, default=1.,         help="temperature before softmax when sampling")
    parser.add_argument("--drop-label",   type=float, default=0.1,        help="drop rate for cfg")
    parser.add_argument("--dropout",      type=float, default=0.1,        help="dropout rate")
    parser.add_argument("--interpolate_rate",    type=float, default=0.5,        help="randomize noise")
    parser.add_argument("--interpolate",         action='store_true',            help="mode to interpolate")
    parser.add_argument("--lambda_gt",    type=float, default=1.0,        help="lambda for recon loss")
    parser.add_argument("--data_load",    action='store_true',            help="data load")
    parser.add_argument("--data_gen",     action='store_true',            help="data generation and save")
    parser.add_argument("--nelbo_test",   action='store_true',            help="nelbo_test")
    parser.add_argument("--reflow_num",   type=int,   default=1,          help="reflow num")
    parser.add_argument("--duplicate",    type=int,   default=0,          help="duplicate x0,y for total correlation compute")
    parser.add_argument("--test_by_train_data", action='store_true',      help="test by train data")
    parser.add_argument("--test-only",    action='store_true',            help="only evaluate the model")
    parser.add_argument("--save_only",    action='store_true',            help="only evaluate the model")
    parser.add_argument("--save_results", action='store_true',            help="save all imgs used for fid measurement")
    parser.add_argument("--test_image_num", type=int,   default=50000,    help="the number of images to test")
    parser.add_argument("--num_vis", type=int,   default=500,    help="the number of images to test")
    parser.add_argument("--label_smoothing", type=float, default=0.1,     help="label smoothing rate")
    parser.add_argument("--use_precomputed_stats", action='store_true',  help="use precomputed stats for FID calculation")
    parser.add_argument("--precomputed_stats_path", type=str, default="fid_stats_imagenet256_guided_diffusion.npz", help="path to precomputed stats file")
    parser.add_argument("--resume",       action='store_true',            help="resume training of the model")
    parser.add_argument("--rand_mask",    action='store_true',            help="denote random tokens with [RAND]")
    parser.add_argument("--load_optimizer_states",    action='store_true',            help="denote random tokens with [RAND]")
    parser.add_argument("--adaptive_loss", choices=['v1', 'v2', None],    help="use adaptive loss") 
    parser.add_argument("--analytic_reflow",   action='store_true',       help="force analytic sampling during reflow")
    parser.add_argument("--debug",        action='store_true',            help="debug")
    parser.add_argument("--lr_cosine",    action='store_true',       help="test")
    parser.add_argument("--compute_validation_fid",    action='store_true',       help="test")
    parser.add_argument("--compute_recon_fid",    action='store_true',       help="test")
    parser.add_argument("--compute_train_recon_fid",    action='store_true',       help="test")
    parser.add_argument("--cos_max_iter", type=int, default=970*20, help="max iter for cosine lr #default: 970*200")
    parser.add_argument("--save_freq", type=int, default=10, help="save_network frequency")
    parser.add_argument("--repeat_reflow_iter", type=int, default=-1, help="reflow repeat iter")
    parser.add_argument("--test_with_train_x0", action='store_true',       help="compute FID with train_x0")
    parser.add_argument("--reflow_data_path", type=str, default=None, help="reflow data path")
    parser.add_argument("--optim_mode", choices=['Adam', 'AdamW'], default='AdamW', help="optimizer mode")
    parser.add_argument("--use_DA", action='store_true',       help="use Data Augmentation if needed")
    parser.add_argument("--dropout_reflow", action='store_true',       help="use training mode when generating reflow data")
    parser.add_argument("--fill_mask_first", action='store_true',       help="fill_mask_first")
    parser.add_argument("--const_cfg", action='store_true',       help="constant_cfg")
    parser.add_argument("--get_val_data", action='store_true', help="define to Indipendent Coupling")

    # for revise method
    parser.add_argument("--revise_M", type=int, default=0, help="use revise method")
    parser.add_argument("--revise_T", type=int, default=0, help="use revise method")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0
    args.sch_iter = 0

    world_size = torch.cuda.device_count()

    if args.test_by_train_data:
        print("Test by train data")
        args.use_precomputed_stats = True
        print('Using precomputed stats for FID calculation')

    if args.seed > 0: # Set the seed for reproducibility
        if world_size > 1:
            seed =  args.seed + int(os.environ["LOCAL_RANK"])
        else:
            seed = args.seed
        print(f"Set seed to {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    if world_size > 1:  # launch multi training
        print(f"{world_size} GPU(s) found, launch multi-gpus training")
        args.is_multi_gpus = True
        launch_multi_main(args)
    else:  # launch single Gpu training
        print(f"{world_size} GPU found")
        args.is_master = True
        args.is_multi_gpus = False
        main(args)
