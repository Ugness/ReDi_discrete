# Trainer for MaskGIT
import os
import random
import time
import math
import wandb

import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from Trainer.trainer import Trainer
from Network.transformer import MaskTransformer

from Network.Taming.models.vqgan import VQModel

class ReflowDataset(torch.utils.data.Dataset):
    def __init__(self, x0, x1, y):
        self.x0 = x0
        self.x1 = x1
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx], self.y[idx]

class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        self.args = args                                                        # Main argument see main.py
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        self.ae, _ = self.get_network("autoencoder")
        self.codebook_size = self.ae.n_embed   
        self.patch_size = self.args.img_size // 2**(self.ae.encoder.num_resolutions-1)     # Load VQGAN
        print(f"Acquired codebook size: {self.codebook_size}, f_factor: {2**(self.ae.encoder.num_resolutions-1)}")
        self.vit, ckpt = self.get_network("vit")                                      # Load Masked Bidirectional Transformer
        self.criterion = self.get_loss("cross_entropy", label_smoothing=self.args.label_smoothing, reduction='none')    # Get cross entropy loss
        self.optim = self.get_optim(self.vit, self.args.lr, mode=self.args.optim_mode, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay
        if self.args.load_optimizer_states:
            self.optim.load_state_dict(ckpt['optimizer_state_dict'])
            print("Optimizer state loaded")

        self.run_id = self.args.exp_name
        if self.args.is_master:
            if not self.args.debug:
                os.makedirs(os.path.join(self.args.vit_folder, self.run_id), exist_ok=True)  # Create folder for saving the model

        # Load data if aim to train or test the model
        if not args.debug:
            if self.args.data_gen:
                self.generate_reflow_data(get_val_data = self.args.get_val_data)
            elif self.args.data_load:
                self.train_data, self.test_data = self.get_reflow_dataloader()
            else:
                self.train_data, self.test_data = self.get_data()

        # Initialize evaluation object if testing
        if self.args.test_only:
            from Metrics.sample_and_eval import SampleAndEval
            if self.args.test_by_train_data:
                fname = f"{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}" \
                     f"_rtemp={self.args.r_temp}_w={self.args.cfg_w}_test_by_train"
            else:
                fname = f"{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}" \
                     f"_rtemp={self.args.r_temp}_w={self.args.cfg_w}"
            if self.args.revise_M > 0 and self.args.revise_T > 0:
                fname += f"_revise_M={self.args.revise_M}_T={self.args.revise_T}"
            save_dir = os.path.join('results/vis_all', self.run_id, fname) if self.args.save_results else None
            self.sae = SampleAndEval(device=self.args.device, 
                                     num_images=self.args.test_image_num, 
                                     save_dir=save_dir,
                                     use_precomputed_stats=self.args.use_precomputed_stats,
                                     precomputed_stats_path=self.args.precomputed_stats_path,
                                     test_by_train_data=self.args.test_by_train_data,
                                     save_only=self.args.save_only,
                                     exp_name=f"results/{self.run_id}/metrics_{fname}",)

    @torch.no_grad()
    def generate_reflow_data(self, reflow_data_path=None, get_val_data=False):
        """
            Todo: get the data for training and testing
            For test data, it is equal to the origin get_data method
            For training data, it should be a reflow data loader; return x0, x1, y

            :return
                train_data -> DataLoader: the training data: x0, x1, y
                test_data  -> DataLoader: the testing data: x, y
        """
        assert self.args.reflow_num > 0, "Reflow number should be greater than 0"
        with torch.no_grad():
            if reflow_data_path is None:
                if self.args.interpolate:
                    reflow_data_path = f"./dataset/{self.run_id}"
                else:
                    reflow_data_path = f"./dataset/"

                # Train options / Decoding Options / Reflow num
                reflow_data_path += f"lambda{self.args.lambda_gt}_grad{self.args.grad_cum}/" \
                                    f"{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}_rtemp={self.args.r_temp}" \
                                    f"_w={self.args.cfg_w}_randomize=linear"
                if self.args.revise_M > 0 and self.args.revise_T > 0:
                    reflow_data_path += f"_revise_M={self.args.revise_M}_T={self.args.revise_T}"
                if self.args.analytic_reflow:
                    reflow_data_path += f"_analytic"
                
                reflow_data_path += f"/re{self.args.reflow_num}"
            
            # create psuedo dataset to get each label evenly
            data_train, _ = self.get_pseudo_data(num_classes=1000, num_images=self.args.test_image_num, duplicate=self.args.duplicate)
            _, data_test = self.get_imagenet_data()

            if get_val_data:
                # TODO: create distributed sampler
                val_sampler = DistributedSampler(data_test, shuffle=False) if self.args.is_multi_gpus else None
                val_origin = DataLoader(data_test, batch_size=250,
                                        shuffle=False,
                                        num_workers=self.args.num_workers, pin_memory=False,
                                        drop_last=True, sampler=val_sampler)
                # train_origin = tqdm(train_origin, leave=False, desc="Reflow dataset load") if self.args.is_master else train_origin

                data_val_x0 = []
                data_val_x1 = []
                data_val_y = []

                print("Generating reflow data for validation...")
                from einops import repeat

                rng = torch.Generator()
                rng.manual_seed(self.args.seed)

                # TODO: enable multi-gpu gather
                for x1, y in tqdm(val_origin):
                    y = y.to(self.args.device)
                    x1 = x1.to(self.args.device)

                    emb, _, [_, _, code] = self.ae.encode(x1)
                    data_val_x1.append(code.clone())
                    data_val_y.append(y.clone())

                    M = 5
                    N = 5
                    x0_mask = torch.full(size=(N, M, self.patch_size, self.patch_size), fill_value=self.args.mask_value)
                    x0_rand = torch.randint(0, self.codebook_size, size=(N, M, self.patch_size, self.patch_size), generator=rng)
                    x0 = torch.where(torch.rand(N, M, self.patch_size, self.patch_size, generator=rng) < self.args.interpolate_rate, x0_mask, x0_rand)
                    x0 = repeat(x0, 'n m h w -> (n m d) h w', n=5, d=10)
                    data_val_x0.append(x0)
                
                data_val_x0 = torch.cat(data_val_x0, dim=0).cpu()
                data_val_x1 = torch.cat(data_val_x1, dim=0).cpu()
                data_val_y = torch.cat(data_val_y, dim=0).cpu()
                # torch.distributed.barrier()

                # assert len(reflow_train_data) == len(train_origin), "Reflow data size not equal to original data size"
                # all_gather
                if self.args.is_multi_gpus:
                    # Only the destination process (e.g., rank 0) provides a list to receive the tensors
                    # gather data_train_x0 from multiple GPUs
                    data_val_x0_all = [torch.empty_like(data_val_x0) for _ in range(torch.cuda.device_count())] if self.args.is_master else None
                    torch.distributed.gather(data_val_x0, data_val_x0_all, dst=0)
                    # gather data_train_x1 from multiple GPUs
                    data_val_x1_all = [torch.empty_like(data_val_x1) for _ in range(torch.cuda.device_count())] if self.args.is_master else None
                    torch.distributed.gather(data_val_x1, data_val_x1_all, dst=0)
                    # gather data_train_y from multiple GPUs
                    data_val_y_all = [torch.empty_like(data_val_y) for _ in range(torch.cuda.device_count())] if self.args.is_master else None
                    torch.distributed.gather(data_val_y, data_val_y_all, dst=0)
                    
                if self.args.is_master:
                    # data_train_x0 = torch.cat(data_train_x0_all, dim=0).cpu()[:self.args.test_image_num]
                    # data_val_x0 = torch.cat(data_val_x0_all, dim=0).cpu()
                    assert len(data_val_x0) == self.args.test_image_num
                    # data_val_x1 = torch.cat(data_val_x1_all, dim=0).cpu()[:self.args.test_image_num]
                    # data_val_y = torch.cat(data_val_y_all, dim=0).cpu()[:self.args.test_image_num]
                    postfix = '' if self.args.duplicate == 0 else f"/{self.args.duplicate}"
                    os.makedirs(reflow_data_path+postfix, exist_ok=True)
                    np.save(f"{reflow_data_path}{postfix}/x0.npy", data_val_x0.numpy(), allow_pickle=True)
                    np.save(f"{reflow_data_path}{postfix}/x1.npy", data_val_x1.numpy(), allow_pickle=True)
                    np.save(f"{reflow_data_path}{postfix}/y.npy", data_val_y.numpy(), allow_pickle=True)
                    print(f"Reflow data saved to {reflow_data_path}")
                # barrier
                # torch.distributed.barrier()
                return

            # TODO: create distributed sampler
            train_sampler = DistributedSampler(data_train, shuffle=False) if self.args.is_multi_gpus else None
            train_origin = DataLoader(data_train, batch_size=125,
                                    shuffle=False,
                                    num_workers=self.args.num_workers, pin_memory=False,
                                    drop_last=True, sampler=train_sampler)
            # train_origin = tqdm(train_origin, leave=False, desc="Reflow dataset load") if self.args.is_master else train_origin

            data_train_x0 = []
            data_train_x1 = []
            data_train_y = []

            # TODO: enable multi-gpu gather
            for x0, y in train_origin:
                y = y.to(self.args.device)
                x0 = x0.to(self.args.device)
                data_train_x0.append(x0.clone())
                data_train_y.append(y.clone())

                x1 = self.sample(init_code=x0,
                        nb_sample=y.size(0),
                        labels=y,
                        sm_temp=self.args.sm_temp,
                        w=self.args.cfg_w,
                        randomize="linear",
                        r_temp=self.args.r_temp,
                        sched_mode=self.args.sched_mode,
                        step=self.args.step,
                        skip_decode=True,
                        analytic_sampling=self.args.analytic_reflow,
                        use_train_mode=self.args.dropout_reflow,
                        revise_M=self.args.revise_M,
                        revise_T=self.args.revise_T,
                )[0].view(*x0.shape)

                data_train_x1.append(x1)
            
            data_train_x0 = torch.cat(data_train_x0, dim=0)
            data_train_x1 = torch.cat(data_train_x1, dim=0)
            data_train_y = torch.cat(data_train_y, dim=0)
            torch.distributed.barrier()

            # assert len(reflow_train_data) == len(train_origin), "Reflow data size not equal to original data size"
            # all_gather
            if self.args.is_multi_gpus:
                # Only the destination process (e.g., rank 0) provides a list to receive the tensors
                # gather data_train_x0 from multiple GPUs
                data_train_x0_all = [torch.empty_like(data_train_x0) for _ in range(torch.cuda.device_count())] if self.args.is_master else None
                torch.distributed.gather(data_train_x0, data_train_x0_all, dst=0)
                # gather data_train_x1 from multiple GPUs
                data_train_x1_all = [torch.empty_like(data_train_x1) for _ in range(torch.cuda.device_count())] if self.args.is_master else None
                torch.distributed.gather(data_train_x1, data_train_x1_all, dst=0)
                # gather data_train_y from multiple GPUs
                data_train_y_all = [torch.empty_like(data_train_y) for _ in range(torch.cuda.device_count())] if self.args.is_master else None
                torch.distributed.gather(data_train_y, data_train_y_all, dst=0)
                
            if self.args.is_master:
                # data_train_x0 = torch.cat(data_train_x0_all, dim=0).cpu()[:self.args.test_image_num]
                data_train_x0 = torch.cat(data_train_x0_all, dim=0).cpu()
                assert len(data_train_x0) == self.args.test_image_num, f"Reflow data size not equal to original data size: {len(data_train_x0)} != {self.args.test_image_num}"
                data_train_x1 = torch.cat(data_train_x1_all, dim=0).cpu()[:self.args.test_image_num]
                data_train_y = torch.cat(data_train_y_all, dim=0).cpu()[:self.args.test_image_num]
                postfix = '' if self.args.duplicate == 0 else f"/{self.args.duplicate}"
                os.makedirs(reflow_data_path+postfix, exist_ok=True)
                np.save(f"{reflow_data_path}{postfix}/x0.npy", data_train_x0.numpy(), allow_pickle=True)
                np.save(f"{reflow_data_path}{postfix}/x1.npy", data_train_x1.numpy(), allow_pickle=True)
                np.save(f"{reflow_data_path}{postfix}/y.npy", data_train_y.numpy(), allow_pickle=True)
                print(f"Reflow data saved to {reflow_data_path}")
            # barrier
            torch.distributed.barrier()
        return

    def get_reflow_dataloader(self, reflow_data_path=None):
        """
            For test data, it is equal to the origin get_data method
            For training data, it should be a reflow data loader; return x0, x1, y

            :return
                train_data -> DataLoader: the training data: x0, x1, y
                test_data  -> DataLoader: the testing data: x, y
        """
        assert self.args.reflow_num > 0, "Reflow number should be greater than 0"

        if reflow_data_path is None:
            if self.args.interpolate:
                reflow_data_path = f"./dataset/{self.run_id}"
            else:
                reflow_data_path = f"./dataset/"

            # Train options / Decoding Options / Reflow num
            reflow_data_path += f"lambda{self.args.lambda_gt}_grad{self.args.grad_cum}/" \
                                f"{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}_rtemp={self.args.r_temp}" \
                                f"_w={self.args.cfg_w}_randomize=linear"
            if self.args.revise_M > 0 and self.args.revise_T > 0:
                reflow_data_path += f"_revise_M={self.args.revise_M}_T={self.args.revise_T}"
            if self.args.analytic_reflow:
                reflow_data_path += f"_analytic"
            
            reflow_data_path += f"/re{self.args.reflow_num}"

        _, data_test = self.get_imagenet_data()
        postfix = '' if self.args.duplicate == 0 else f"/{self.args.duplicate}"
        data_train_x0 = np.load(f"{reflow_data_path}{postfix}/x0.npy", allow_pickle=True)
        data_train_x1 = np.load(f"{reflow_data_path}{postfix}/x1.npy", allow_pickle=True)
        data_train_y  = np.load(f"{reflow_data_path}{postfix}/y.npy", allow_pickle=True)
        data_train = ReflowDataset(data_train_x0, data_train_x1, data_train_y)
            
        train_sampler, test_sampler = self.get_distributed_sampler(data_train, data_test)

        train_loader = DataLoader(data_train, batch_size=self.args.bsize,
                                shuffle=False if self.args.is_multi_gpus else True,
                                num_workers=self.args.num_workers, pin_memory=True,
                                drop_last=True, sampler=train_sampler)
        test_loader = DataLoader(data_test, batch_size=self.args.bsize,
                                shuffle=False if self.args.is_multi_gpus else True,
                                num_workers=self.args.num_workers, pin_memory=True,
                                sampler=test_sampler)
        return train_loader, test_loader

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        checkpoint=None
        if archi == "vit":
            if self.args.vit_size == "base":
                model = MaskTransformer(
                    img_size=self.args.img_size, hidden_dim=768, codebook_size=self.codebook_size, f_factor=self.patch_size, depth=24, heads=16, mlp_dim=3072, dropout=self.args.dropout,  # Base
                    rand_mask=self.args.rand_mask, adaptive_loss=self.args.adaptive_loss  # Base
                )
            elif self.args.vit_size == "big":
                model = MaskTransformer(
                    img_size=self.args.img_size, hidden_dim=1024, codebook_size=self.codebook_size, f_factor=self.patch_size, depth=32, heads=16, mlp_dim=3072, dropout=self.args.dropout,  # Base
                    rand_mask=self.args.rand_mask, adaptive_loss=self.args.adaptive_loss  # Base
                )
            elif self.args.vit_size == "huge":
                model = MaskTransformer(
                    img_size=self.args.img_size, hidden_dim=1024, codebook_size=self.codebook_size, f_factor=self.patch_size, depth=48, heads=16, mlp_dim=3072, dropout=self.args.dropout,  # Base
                    rand_mask=self.args.rand_mask, adaptive_loss=self.args.adaptive_loss  # Base
                )
            if self.args.resume:
                ckpt = self.args.vit_folder
                ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
                self.args.vit_folder = os.path.dirname(self.args.vit_folder)
                if self.args.is_master:
                    print("load ckpt from:", ckpt)
                # Read checkpoint file
                checkpoint = torch.load(ckpt, map_location='cpu')
                # Update the current epoch and iteration
                self.args.iter += checkpoint['iter']
                self.args.global_epoch += checkpoint['global_epoch']
                # Load network
                m, u = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Missing keys:", m)
                print("Unexpected keys:", u)

            model = model.to(self.args.device)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "autoencoder":
            # Load config
            config = OmegaConf.load(os.path.join(self.args.vqgan_folder, "model.yaml"))
            model = VQModel(**config.model.params)
            checkpoint = torch.load(os.path.join(self.args.vqgan_folder, "last.ckpt"), map_location="cpu")["state_dict"]
            # Load network
            model.load_state_dict(checkpoint, strict=False)
            model = model.eval()
            model = model.to(self.args.device)

            if self.args.is_multi_gpus: # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model, checkpoint

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None, codebook_size=256, interpolate_rate=None):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":                # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":              # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        # interpolate mode
        if interpolate_rate is not None:
            assert value > 0, "Value must be positive in interpolate mode"
            x0_mask = torch.full_like(code, value)
            x0_rand = torch.randint_like(code, 0, codebook_size)
            maskmask = torch.rand(size=code.size()).to(code.device) < interpolate_rate # 1 if mask, 0 if random
            mask_code[mask] = torch.where(maskmask[mask], x0_mask[mask], x0_rand[mask])
            randommask = mask * (~maskmask.cpu())
            maskmask = mask * maskmask.cpu()
        elif value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
            maskmask = mask
            randommask = torch.zeros_like(mask)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)
            randommask = mask
            maskmask = torch.zeros_like(mask)

        return mask_code, mask, maskmask, randommask

    @staticmethod
    def get_mask_code_reflow(code, mask, mode="arccos", value=None, codebook_size=256, is_interpolate=False):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mask  -> torch.LongTensor(): bsize * 16 * 16, the mask of the code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask_pos    -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":                # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":              # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        elif mode == "distill":
            val_to_mask = torch.ones_like(r) + 0.1
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask_pos = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        mask_code[mask_pos] = mask[mask_pos]

        # interpolate mode
        if is_interpolate:
            x0_mask_pos = (mask == value)
            randommask = mask_pos * (~x0_mask_pos.cpu())
            maskmask = mask_pos * x0_mask_pos.cpu()
        elif value > 0:  # Mask the selected token by the value
            maskmask = mask_pos
            randommask = torch.zeros_like(mask_pos)
        else:  # Replace by a randon token
            randommask = mask_pos
            maskmask = torch.zeros_like(mask_pos)

        return mask_code, mask_pos, maskmask, randommask

    def adap_sche(self, step, mode="arccos", leave=False, v2=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        if 'v2' in mode:
            mode = mode.split('_')[0]
            v2 = True
        else:
            v2 = False
        if mode == "distill" or self.args.train_sch == "distill":
            sche = torch.linspace(1, 0, 1)
            sche = torch.ones_like(sche) * (self.patch_size * self.patch_size)
            sche = sche.long()
            return tqdm(sche.int(), leave=leave)

        if v2:
            r = torch.linspace(0, 1, step+1)
            if mode == "linear":                # linear scheduler
                val_to_mask = 1 - r
            elif mode == "square":              # square scheduler
                val_to_mask = 1 - (r ** 2)
            elif mode == "cosine":              # cosine scheduler
                val_to_mask = torch.cos(r * math.pi * 0.5)
            elif mode == "arccos":              # arc cosine scheduler
                val_to_mask = torch.arccos(r) / (math.pi * 0.5)
            else:
                val_to_mask = None
            
            sche = val_to_mask * (self.patch_size * self.patch_size)
            sche = sche.round()
            sche = sche[:-1] - sche[1:]  # diff between two steps
            sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
            sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up nb of code
            return tqdm(sche.int(), leave=leave)
        
        else: # Original Halton-maskgit impl. (Diff. from MaskGiT)
            r = torch.linspace(1, 0, step)
            if mode == "root":              # root scheduler
                val_to_mask = 1 - (r ** .5)
            elif mode == "linear":          # linear scheduler
                val_to_mask = 1 - r
            elif mode == "square":          # square scheduler
                val_to_mask = 1 - (r ** 2)
            elif mode == "cosine":          # cosine scheduler
                val_to_mask = torch.cos(r * math.pi * 0.5)
            elif mode == "arccos":          # arc cosine scheduler
                val_to_mask = torch.arccos(r) / (math.pi * 0.5)
            else:
                return

            # fill the scheduler by the ratio of tokens to predict at each step
            sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
            sche = sche.round()
            sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
            sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up nb of code
            return tqdm(sche.int(), leave=leave)

    def train_one_epoch(self, log_iter=2500):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss, cum_acc = 0., 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        window_loss_random = deque(maxlen=self.args.grad_cum)
        window_loss_mask = deque(maxlen=self.args.grad_cum)
        window_loss_gt = deque(maxlen=self.args.grad_cum)
        window_acc = deque(maxlen=self.args.grad_cum)
        window_acc_ignore = deque(maxlen=self.args.grad_cum)
        window_acc_random = deque(maxlen=self.args.grad_cum)
        window_acc_mask = deque(maxlen=self.args.grad_cum)
        window_acc_gt = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        self.grad_iter = 0
        for x, y in bar:
            self.adapt_learning_rate()   # adapt the learning rate with a warmup and a cosine decay
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

            # Drop xx% of the condition for cfg
            drop_label = (torch.rand(x.size(0)) < self.args.drop_label).bool().to(self.args.device)

            # VQGAN encoding to img tokens
            with torch.no_grad():
                emb, _, [_, _, code] = self.ae.encode(x)
                code = code.reshape(x.size(0), self.patch_size, self.patch_size)

            # Mask the encoded tokens
            if self.args.interpolate:
                masked_code, mask, maskmask, randommask = self.get_mask_code(code, mode=self.args.train_sch, value=self.args.mask_value, codebook_size=self.codebook_size, interpolate_rate=self.args.interpolate_rate)
            else:
                masked_code, mask, maskmask, randommask = self.get_mask_code(code, mode=self.args.train_sch, value=self.args.mask_value, codebook_size=self.codebook_size)

            randmask = randommask.to(masked_code.device)

            # compute t with mask_ratio for adaptive loss
            mask_t = mask.view(mask.size(0), -1).to(self.args.device)
            mask_ratio = mask_t.sum(-1) / mask_t.size(1)

            with torch.amp.autocast("cuda"):                             # half precision
                pred, u = self.vit(masked_code, y, drop_label=drop_label, randmask=randmask, t=mask_ratio)  # The unmasked tokens prediction
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1)) / self.args.grad_cum
                loss = loss.view(-1)
                loss_gt = loss[~mask.view(-1)].sum() / loss.numel()
                loss_mask = loss[maskmask.view(-1)].sum() / loss.numel()
                loss_random = loss[randommask.view(-1)].sum() / loss.numel()
                assert torch.isclose(loss_gt + loss_mask + loss_random, loss.mean()), "Loss not equal to the sum of the parts"
                loss = loss_gt * self.args.lambda_gt + loss_mask + loss_random

            # update weight if accumulation of gradient is done
            update_grad = self.grad_iter % self.args.grad_cum == self.args.grad_cum - 1
            if update_grad:
                self.optim.zero_grad()
                self.grad_iter = 0
            else:
                self.grad_iter += 1

            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            if update_grad:
                self.scaler.unscale_(self.optim)                      # rescale loss
                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
                self.scaler.step(self.optim)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())
            window_loss_gt.append(loss_gt.cpu().item())
            window_loss_mask.append(loss_mask.cpu().item())
            window_loss_random.append(loss_random.cpu().item())
            acc = torch.max(pred.reshape(-1, self.codebook_size+1).data, 1)[1]
            acc = (acc.view(-1) == code.view(-1)).float()
            cum_acc += acc.mean().item()
            window_acc.append(acc.mean().item())
            window_acc_ignore.append(acc[mask.view(-1)].mean().item())
            window_acc_gt.append(acc[~mask.view(-1)].mean().item())
            window_acc_mask.append(acc[maskmask.view(-1)].mean().item())
            window_acc_random.append(acc[randommask.view(-1)].mean().item())

            # logs
            if update_grad and self.args.is_master:
                # Mini Batch loss
                self.log_add_scalar('Train/MiniLoss', np.array(window_loss).sum(), self.args.iter)
                self.log_add_scalar('Train/MiniLossGT', np.array(window_loss_gt).sum(), self.args.iter)
                self.log_add_scalar('Train/MiniLossMASK', np.array(window_loss_mask).sum(), self.args.iter)
                self.log_add_scalar('Train/MiniLossRANDOM', np.array(window_loss_random).sum(), self.args.iter)
                
                # Overall Accuracy (including un-masked visual token)
                self.log_add_scalar('Train/MiniAcc', np.array(window_acc).mean(), self.args.iter)
                
                # Accuracy excluding un-masked visual token
                self.log_add_scalar('Train/MiniAccIgnore', np.array(window_acc_ignore).mean(), self.args.iter)
                self.log_add_scalar('Train/MiniAccGT', np.array(window_acc_gt).mean(), self.args.iter)
                self.log_add_scalar('Train/MiniAccMASK', np.array(window_acc_mask).mean(), self.args.iter)
                self.log_add_scalar('Train/MiniAccRANDOM', np.array(window_acc_random).mean(), self.args.iter)

            if self.args.iter % log_iter == 0 and self.args.is_master:
                nb_sample = 10
                if self.args.interpolate:
                    init_code_rand = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size))
                    init_code_mask = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value)
                    # Generate init_code by mixing random and masked code
                    init_code = torch.where(torch.rand(nb_sample, self.patch_size, self.patch_size) < self.args.interpolate_rate, init_code_mask, init_code_rand).to(self.args.device)
                else:
                    init_code = None
                # Generate sample for visualization
                gen_sample = self.sample(init_code=init_code,
                                         nb_sample=nb_sample,
                                         labels=None,
                                         sm_temp=self.args.sm_temp,
                                         w=self.args.cfg_w,
                                         randomize="linear",
                                         r_temp=self.args.r_temp,
                                         sched_mode=self.args.sched_mode,
                                         step=self.args.step
                                         )[0]
                gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                # Show reconstruction
                nb_sample = min(10, x.size(0))
                unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                mask = mask.to(unmasked_code.device).view(mask.size(0), -1)
                masked_code = masked_code.view(masked_code.size(0), -1)
                unmasked_code = masked_code * (~mask) + unmasked_code * mask
                if self.args.interpolate:
                    reco_sample = self.reco(x=x[:10], code=code[:10], masked_code=masked_code[:10], unmasked_code=unmasked_code[:10])
                else:
                    reco_sample = self.reco(x=x[:10], code=code[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
                reco_sample = vutils.make_grid(reco_sample.data, nrow=nb_sample, padding=2, normalize=True)
                self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)

                # Save Network
                self.save_network(model=self.vit, path=os.path.join(self.args.vit_folder, self.run_id, "current.pth"),
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            if update_grad:
                self.args.iter += 1
                self.args.sch_iter += 1
                # log lr
                if self.args.is_master:
                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.args.iter)

        return cum_loss / n
    
    def train_one_epoch_reflow(self, log_iter=2500):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss, cum_acc = 0., 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        window_loss_random = deque(maxlen=self.args.grad_cum)
        window_loss_mask = deque(maxlen=self.args.grad_cum)
        window_loss_gt = deque(maxlen=self.args.grad_cum)
        window_acc = deque(maxlen=self.args.grad_cum)
        window_acc_ignore = deque(maxlen=self.args.grad_cum)
        window_acc_random = deque(maxlen=self.args.grad_cum)
        window_acc_mask = deque(maxlen=self.args.grad_cum)
        window_acc_gt = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        self.grad_iter = 0
        for x0, x1, y in bar:
            self.adapt_learning_rate()   # adapt the learning rate with a warmup and a cosine decay
            x0 = x0.to(self.args.device)
            x1 = x1.to(self.args.device)
            y = y.to(self.args.device)

            # Drop xx% of the condition for cfg
            drop_label = (torch.rand(x1.size(0)) < self.args.drop_label).bool().to(self.args.device)

            # Mask the encoded tokens
            masked_code, mask, maskmask, randommask = self.get_mask_code_reflow(x1, mask=x0, mode=self.args.train_sch, value=self.args.mask_value, codebook_size=self.codebook_size, is_interpolate=self.args.interpolate)

            # Compute t from mask
            mask_t = mask.view(mask.size(0), -1)
            mask_ratio = mask_t.sum(-1) / mask_t.size(1)
            t_idx = (1-mask_ratio)
            randmask = randommask.to(masked_code.device)
            with torch.amp.autocast("cuda"):                             # half precision
                pred, u = self.vit(masked_code, y, drop_label=drop_label, randmask=randmask, t=mask_ratio)  # The unmasked tokens prediction
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), x1.view(-1)) / self.args.grad_cum
                loss = loss.view(u.shape[0], -1)
                loss = loss / torch.exp(u) + u
                loss = loss.view(-1)
                loss_gt = loss[~mask.view(-1)].sum() / loss.numel()
                loss_mask = loss[maskmask.view(-1)].sum() / loss.numel()
                loss_random = loss[randommask.view(-1)].sum() / loss.numel()
                assert torch.isclose(loss_gt + loss_mask + loss_random, loss.mean()), "Loss not equal to the sum of the parts"
                loss = loss_gt * self.args.lambda_gt + loss_mask + loss_random

            # update weight if accumulation of gradient is done
            update_grad = self.grad_iter % self.args.grad_cum == self.args.grad_cum - 1
            if update_grad:
                self.optim.zero_grad()
                self.grad_iter = 0
            else:
                self.grad_iter += 1

            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            if update_grad:
                self.scaler.unscale_(self.optim)                      # rescale loss
                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
                self.scaler.step(self.optim)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())
            window_loss_gt.append(loss_gt.cpu().item())
            window_loss_mask.append(loss_mask.cpu().item())
            window_loss_random.append(loss_random.cpu().item())
            acc = torch.max(pred.reshape(-1, self.codebook_size+1).data, 1)[1]
            acc = (acc.view(-1) == x1.view(-1)).float()
            cum_acc += acc.mean().item()
            window_acc.append(acc.mean().item())
            window_acc_ignore.append(acc[mask.view(-1)].mean().item())
            window_acc_gt.append(acc[~mask.view(-1)].mean().item())
            window_acc_mask.append(acc[maskmask.view(-1)].mean().item())
            window_acc_random.append(acc[randommask.view(-1)].mean().item())

            # logs
            if update_grad and self.args.is_master:
                # Mini Batch loss
                self.log_add_scalar('Train/MiniLoss', np.array(window_loss).sum(), self.args.iter)
                self.log_add_scalar('Train/MiniLossGT', np.array(window_loss_gt).sum(), self.args.iter)
                self.log_add_scalar('Train/MiniLossMASK', np.array(window_loss_mask).sum(), self.args.iter)
                self.log_add_scalar('Train/MiniLossRANDOM', np.array(window_loss_random).sum(), self.args.iter)
                # Overall Accuracy (including un-masked visual token)
                self.log_add_scalar('Train/MiniAcc', np.array(window_acc).mean(), self.args.iter)
                # Accuracy excluding un-masked visual token
                self.log_add_scalar('Train/MiniAccIgnore', np.array(window_acc_ignore).mean(), self.args.iter)
                self.log_add_scalar('Train/MiniAccGT', np.array(window_acc_gt).mean(), self.args.iter)
                self.log_add_scalar('Train/MiniAccMASK', np.array(window_acc_mask).mean(), self.args.iter)
                self.log_add_scalar('Train/MiniAccRANDOM', np.array(window_acc_random).mean(), self.args.iter)
            
            # For duplicate dataset training, save the model every 970 iterations before 769825
            # if (self.args.iter - 760125) % 970 == 0 and self.args.is_master:
            #     # 769825
            #     self.save_network(model=self.vit, path=os.path.join(self.args.vit_folder, self.run_id, f"iter_{self.args.iter:08d}.pth"),
            #                       iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)
            #     if self.args.iter > 770000:
            #         import sys
            #         print("Training finished, exiting...")
            #         sys.exit(0)
            

            if self.args.iter % log_iter == 0 and self.args.is_master:
                nb_sample = 10
                if self.args.interpolate:
                    init_code_rand = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size))
                    init_code_mask = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value)
                    # Generate init_code by mixing random and masked code
                    init_code = torch.where(torch.rand(nb_sample, self.patch_size, self.patch_size) < self.args.interpolate_rate, init_code_mask, init_code_rand).to(self.args.device)
                else:
                    init_code = None
                # Generate sample for visualization
                gen_sample = self.sample(init_code=init_code,
                                         nb_sample=nb_sample,
                                         labels=None,
                                         sm_temp=self.args.sm_temp,
                                         w=self.args.cfg_w,
                                         randomize="linear",
                                         r_temp=self.args.r_temp,
                                         sched_mode=self.args.sched_mode,
                                         step=self.args.step
                                         )[0]
                gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                # Show reconstruction
                nb_sample = min(10, x1.size(0))
                unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                mask = mask.to(unmasked_code.device).view(mask.size(0), -1)
                masked_code = masked_code.view(masked_code.size(0), -1)
                unmasked_code = masked_code * (~mask) + unmasked_code * mask
                if self.args.interpolate:
                    reco_sample = self.reco(x=x1[:10], code=x1[:10], masked_code=masked_code[:10], unmasked_code=unmasked_code[:10])
                else:
                    reco_sample = self.reco(x=x1[:10], code=x1[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
                reco_sample = vutils.make_grid(reco_sample.data, nrow=nb_sample, padding=2, normalize=True)
                self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)

                # Save Network
                self.save_network(model=self.vit, path=os.path.join(self.args.vit_folder, self.run_id, "current.pth"),
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # repeat reflow if condition is satisfied
            if self.args.repeat_reflow_iter > 0 and self.args.sch_iter % self.args.repeat_reflow_iter == 0 and self.args.sch_iter > 0:
                if self.args.is_multi_gpus:
                    torch.distributed.barrier()
                self.save_network(model=self.vit, path=os.path.join(self.args.vit_folder, self.run_id, f"reflow_{self.args.reflow_num}.pth"),
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)
                self.args.reflow_num += 1
                self.generate_reflow_data()
                self.train_data, self.test_data = self.get_reflow_dataloader()
                self.args.sch_iter += 1
                self.args.iter += 1
                break # move to next epoch

            if update_grad:
                self.args.iter += 1
                self.args.sch_iter += 1
                # log lr
                if self.args.is_master:
                    self.log_add_scalar('Train/LearningRate', self.optim.param_groups[0]['lr'], self.args.iter)

        return cum_loss / n

    @torch.inference_mode()
    def validate_one_epoch(self, log_iter=2500, ign_gt=False):
        """ Train the model for 1 epoch """
        self.vit.eval()
        bar = tqdm(self.test_data, leave=False) if self.args.is_master else self.test_data
        # Start training for 1 epoch
        
        # create bin from 0 to 1 with 100 intervals
        loss_bins = torch.linspace(0, 1, 257).to(self.args.device)
        loss_hist = torch.zeros(257).to(self.args.device)
        nelbo_hist = torch.zeros(257).to(self.args.device)
        cnt_hist = torch.zeros(257).to(self.args.device)

        for x, y in bar:
            # x0 = x0.to(self.args.device)
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

            # Drop xx% of the condition for cfg
            drop_label = (torch.rand(x.size(0)) < self.args.drop_label).bool().to(self.args.device)

            # VQGAN encoding to img tokens
            with torch.no_grad():
                emb, _, [_, _, x1] = self.ae.encode(x)
                x1 = x1.reshape(x.size(0), self.patch_size, self.patch_size)

            # Mask the encoded tokens
            masked_code, mask, maskmask, randommask = self.get_mask_code(x1, mode='linear', value=self.args.mask_value, codebook_size=self.codebook_size, interpolate_rate=self.args.interpolate_rate)
            
            # Compute t from mask
            mask_t = mask.view(mask.size(0), -1)
            mask_ratio = mask_t.sum(-1) / mask_t.size(1)
            t_idx = (1-mask_ratio)

            randmask = randommask.to(masked_code.device)
            with torch.amp.autocast("cuda"):                             # half precision
                pred = self.vit(masked_code, y, drop_label=drop_label, randmask=randmask)  # The unmasked tokens prediction
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), x1.view(-1)) / self.args.grad_cum
                loss = loss.view(pred.size(0), -1)
                mask = mask.view(pred.size(0), -1).to(self.args.device)
                if ign_gt:
                    loss = loss * mask
                    loss = loss.sum(-1) / mask.sum(-1)
                else:
                    loss = loss.mean(-1)

                # loss_gt = loss[~mask.view(-1)].sum() / loss.numel()
                # loss_mask = loss[maskmask.view(-1)].sum() / loss.numel()
                # loss_random = loss[randommask.view(-1)].sum() / loss.numel()
                # assert torch.isclose(loss_gt + loss_mask + loss_random, loss.mean()), "Loss not equal to the sum of the parts"
                # loss = loss_gt * self.args.lambda_gt + loss_mask + loss_random
            
            # add loss to hist based on loss_bins and mask_ratio
            t_idx = torch.bucketize(t_idx, loss_bins.to(t_idx.device), right=True).to(self.args.device)-1
            for i, idx in enumerate(t_idx):
                assert idx >= 0
                loss_hist[idx] += loss[i]
                nelbo_hist[idx] += loss[i] / (1-mask_ratio[i]+1e-6)
                cnt_hist[idx] += 1

        return loss_hist, nelbo_hist, cnt_hist
    
    @torch.inference_mode()
    def validate_one_epoch_reflow(self, log_iter=2500, ign_gt=False):
        """ Train the model for 1 epoch """
        self.vit.eval()
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        # Start training for 1 epoch
        
        # create bin from 0 to 1 with 100 intervals
        loss_bins = torch.linspace(0, 1, 257).to(self.args.device)
        loss_hist = torch.zeros(257).to(self.args.device)
        nelbo_hist = torch.zeros(257).to(self.args.device)
        cnt_hist = torch.zeros(257).to(self.args.device)

        for x0, x1, y in bar:
            self.adapt_learning_rate()   # adapt the learning rate with a warmup and a cosine decay
            x0 = x0.to(self.args.device)
            x1 = x1.to(self.args.device)
            y = y.to(self.args.device)

            # Drop xx% of the condition for cfg
            drop_label = (torch.rand(x1.size(0)) < self.args.drop_label).bool().to(self.args.device)

            # Mask the encoded tokens
            masked_code, mask, maskmask, randommask = self.get_mask_code_reflow(x1, mask=x0, mode='linear', value=self.args.mask_value, codebook_size=self.codebook_size, is_interpolate=self.args.interpolate)

            # Compute t from mask
            mask_t = mask.view(mask.size(0), -1)
            mask_ratio = mask_t.sum(-1) / mask_t.size(1)
            t_idx = (1-mask_ratio)

            randmask = randommask.to(masked_code.device)
            with torch.amp.autocast("cuda"):                             # half precision
                pred = self.vit(masked_code, y, drop_label=drop_label, randmask=randmask)  # The unmasked tokens prediction
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), x1.view(-1)) / self.args.grad_cum
                loss = loss.view(pred.size(0), -1)
                mask = mask.view(pred.size(0), -1).to(self.args.device)

                if ign_gt:
                    loss = loss * mask
                    loss = loss.sum(-1) / mask.sum(-1)
                else:
                    loss = loss.mean(-1)

                # loss_gt = loss[~mask.view(-1)].sum() / loss.numel()
                # loss_mask = loss[maskmask.view(-1)].sum() / loss.numel()
                # loss_random = loss[randommask.view(-1)].sum() / loss.numel()
                # assert torch.isclose(loss_gt + loss_mask + loss_random, loss.mean()), "Loss not equal to the sum of the parts"
                # loss = loss_gt * self.args.lambda_gt + loss_mask + loss_random
            
            # add loss to hist based on loss_bins and mask_ratio
            t_idx = torch.bucketize(t_idx, loss_bins.to(t_idx.device), right=True).to(self.args.device)-1
            for i, idx in enumerate(t_idx):
                assert idx >= 0
                loss_hist[idx] += loss[i]
                nelbo_hist[idx] += loss[i] / (1-mask_ratio[i]+1e-6)
                cnt_hist[idx] += 1

        return loss_hist, nelbo_hist, cnt_hist

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.epoch):
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            if self.args.data_gen or self.args.data_load:
                train_loss = self.train_one_epoch_reflow()
            else:
                train_loss = self.train_one_epoch()

            # Synch loss
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss, torch.cuda.device_count())

            # Save model
            if e % self.args.save_freq == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=os.path.join(self.args.vit_folder, self.run_id, f"epoch_{self.args.global_epoch:03d}.pth"),
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Train/GlobalLoss', train_loss, self.args.global_epoch)
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter :},"
                      f" Loss {train_loss:.4f},"
                      f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1

    @torch.inference_mode()
    def eval(self):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master and not self.args.save_only:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}"
                  f"_{self.args.const_cfg}_{self.args.fill_mask_first}")
            # TODO Generate N samples and save imgs.
            nb_sample = 10
            gen_samples = []
            for _ in range(10):
                if self.args.interpolate:
                    init_code_rand = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size))
                    init_code_mask = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value)
                    # Generate init_code by mixing random and masked code
                    init_code = torch.where(torch.rand(nb_sample, self.patch_size, self.patch_size) < self.args.interpolate_rate, init_code_mask, init_code_rand).to(self.args.device)
                else:
                    init_code = None
                # Generate sample for visualization
                gen_sample = self.sample(init_code=init_code,
                                            nb_sample=nb_sample,
                                            labels=None,
                                            sm_temp=self.args.sm_temp,
                                            w=self.args.cfg_w,
                                            randomize="linear",
                                            r_temp=self.args.r_temp,
                                            sched_mode=self.args.sched_mode,
                                            step=self.args.step,
                                            revise_M=self.args.revise_M,
                                            revise_T=self.args.revise_T,
                                            )[0]
                gen_samples.append(gen_sample)
            gen_sample = torch.cat(gen_samples, dim=0)
            gen_sample = gen_sample.clamp(-1, 1) * 0.5 + 0.5
            gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
            if self.args.test_by_train_data:
                fname = f"metrics_{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}_{self.args.const_cfg}_{self.args.fill_mask_first}" \
                     f"_rtemp={self.args.r_temp}_w={self.args.cfg_w}_randomize=linear_{self.run_id}_test_by_train"
            else:
                fname = f"metrics_{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}_{self.args.const_cfg}_{self.args.fill_mask_first}" \
                     f"_rtemp={self.args.r_temp}_w={self.args.cfg_w}_randomize=linear_{self.run_id}"
            if self.args.revise_M > 0 and self.args.revise_T > 0:
                fname += f"_revise_M={self.args.revise_M}_T={self.args.revise_T}"
            fname += ".png"
            img_path = os.path.join('results/vis', self.run_id, fname)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            vutils.save_image(gen_sample, img_path)
            print(f"Save generated images to {img_path}")
        # barrier
        if self.args.is_multi_gpus:
            torch.distributed.barrier()

        # Evaluate the model
        if self.args.save_only:
            print("Save only, no evaluation")
            print("Generating Vis Data")
            data_train, _ = self.get_vis_data()
            train_sampler = DistributedSampler(data_train, shuffle=False) if self.args.is_multi_gpus else None
            train_origin = DataLoader(data_train, batch_size=125,
                                    shuffle=False,
                                    num_workers=self.args.num_workers, pin_memory=False,
                                    drop_last=True, sampler=train_sampler)
            self.train_data = train_origin
            self.test_data = train_origin
            
        m = self.sae.compute_and_log_metrics(self)
        if self.args.save_only:
            return
        if self.args.is_master:
            if self.args.test_by_train_data:
                fname = f"metrics_{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}_{self.args.const_cfg}_{self.args.fill_mask_first}" \
                     f"_rtemp={self.args.r_temp}_w={self.args.cfg_w}_randomize=linear_{self.run_id}_test_by_train"
            else:
                fname = f"metrics_{self.args.sched_mode}_step={self.args.step}_temp={self.args.sm_temp}_{self.args.const_cfg}_{self.args.fill_mask_first}" \
                     f"_rtemp={self.args.r_temp}_w={self.args.cfg_w}_randomize=linear_{self.run_id}"
            if self.args.revise_M > 0 and self.args.revise_T > 0:
                fname += f"_revise_M={self.args.revise_M}_T={self.args.revise_T}"
            fname += ".json"
            json_path = os.path.join('results', self.run_id, fname)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            result_dict = m
            result_dict.update({
                'sched_mode': self.args.sched_mode,
                'step': self.args.step,
                'sm_temp': self.args.sm_temp,
                'cfg_w': self.args.cfg_w,
                'r_temp': self.args.r_temp,
                'const_cfg': self.args.const_cfg,
                'fill_mask_first': self.args.fill_mask_first,
            })
            json_str = json.dumps(result_dict, indent=4)
            with open(json_path, 'w') as f:
                f.write(json_str)
            print(f"Save metrics to {json_path}")
        self.vit.train()
        return m

    def reco(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        if self.args.data_gen or self.args.data_load:
            l_visual = []
        else:
            l_visual = [x]
        with torch.no_grad():
            if x is not None and (self.args.data_gen or self.args.data_load):
                x = x.view(x.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(x, 0,  self.codebook_size-1))
                l_visual.append(__x)
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                # Decoding reel code
                _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
                if mask is not None:
                    # Decoding reel code with mask to hide
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask, (self.args.img_size, self.args.img_size)).to(self.args.device))
                    l_visual.append(__x2)
            if masked_code is not None:
                # Decoding masked code
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0,  self.codebook_size-1))
                l_visual.append(__x)

            if unmasked_code is not None:
                # Decoding predicted code
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
                l_visual.append(___x)

        return torch.cat(l_visual, dim=0)

    def sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12, skip_decode=False,
               mask=None, analytic_sampling=False, v2=False, use_train_mode=False, revise_M=0, revise_T=0):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        if use_train_mode:
            self.vit.train()
        else:
            self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        if analytic_sampling:
            sched_mode = 'linear'
            step = 256
            r_temp = 1000
            randomize = 'random'
            v2=True
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            
            if init_code is not None:  # Start with a pre-define code
                code = init_code.detach().clone() # make sure to not modify the original code
                if mask is None:
                    if self.args.interpolate:
                        mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)
                    else:
                        mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
                else:
                    mask = mask.view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.interpolate:
                    assert 0
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)
            x0 = code.detach().clone()
            random_mask = (code != self.args.mask_value).float().view(nb_sample, -1)
            random_mask = mask * random_mask

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode, v2=v2)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                random_mask = mask * random_mask
                with torch.amp.autocast('cuda'):  # half precision
                    if w != 0:
                        # Model Prediction
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0),
                                         randmask=torch.cat([random_mask, random_mask], dim=0)
                                         )
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        if sched_mode == "distill" or self.args.train_sch == 'distill' or self.args.const_cfg:
                            _w = w
                        else:
                            if step == 1:
                                _w = w
                            else:
                                _w = w * (indice / (len(scheduler)-1+1e-6))

                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        logit = self.vit(code.clone(), labels, drop_label=~drop,
                                         randmask=random_mask,
                                         )
                logit[..., -1] = float('-inf')

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1+1e-6))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf.squeeze()) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf.squeeze())
                
                if self.args.fill_mask_first:
                    # TODO: check shape
                    conf[random_mask.bool()] = -(conf.max()-conf.min()+10) # -10 lower than the min value

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            # decode the final prediction
            # _code = torch.clamp(code, 0,  self.codebook_size-1) #0425 patch - force clamping may degrade reflow performance
            _code = code
            if revise_M > 0 and revise_T > 0:
                # Do revise
                assert revise_T == 2
                x1 = _code
                for _ in range(revise_M):
                    mask = torch.stack([torch.randperm(self.patch_size*self.patch_size).to(self.args.device) for _ in range(nb_sample)]) # B, 16*16
                    mask = mask > 127
                    mask = mask.view(nb_sample, self.patch_size, self.patch_size)

                    # iter 1
                    xt = x0 * mask + x1 * (~mask)
                    _x1 = self.sample(xt, nb_sample, labels, w=w, sched_mode='distill', step=1, skip_decode=True,)[0]
                    x1 = _x1 * mask + x1 * (~mask)

                    # flip mask
                    mask = ~mask
                    # iter 2
                    xt = x0 * mask + x1 * (~mask)
                    _x1 = self.sample(xt, nb_sample, labels, w=w, sched_mode='distill', step=1, skip_decode=True,)[0]
                    x1 = _x1 * mask + x1 * (~mask)
                _code = x1

            if not skip_decode:
                x = self.ae.decode_code(_code)
            else:
                x = _code

        self.vit.train()
        return x, l_codes, l_mask
