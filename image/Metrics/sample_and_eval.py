# Borrowed from https://github.com/nicolas-dufour/diffusion/blob/master/metrics/sample_and_eval.py
import random
import clip
import torch
from tqdm import tqdm
import os
import numpy as np

from Metrics.inception_metrics import MultiInceptionMetrics
import cv2

from torch.utils.data import DataLoader

def remap_image_torch(image):
    # min_norm = image.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
    # max_norm = image.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
    image_torch = image * 255
    image_torch = torch.clip(image_torch, 0, 255).to(torch.uint8)
    return image_torch

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SampleAndEval:
    def __init__(self, device, num_images=50000, compute_per_class_metrics=False, num_classes=1000, save_dir=None, use_precomputed_stats=False, precomputed_stats_path=None, test_by_train_data=False, exp_name=None, save_only=False):
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            reset_real_features=False,
            compute_unconditional_metrics=False,
            compute_conditional_metrics=True,
            compute_conditional_metrics_per_class=compute_per_class_metrics,
            num_classes=num_classes,
            num_inception_chunks=10,
            manifold_k=3,
            exp_name=exp_name,
        )
        self.num_images = num_images
        self.true_features_computed = False
        self.device = device
        self.inception_metrics.reset()
        self.save_dir = save_dir
        self.save_only = save_only
        assert self.save_only == False or self.save_dir is not None, "save_dir must be specified if save_only is True"
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        # Add the precomputed stats
        self.use_precomputed_stats = use_precomputed_stats
        self.precomputed_stats_path = precomputed_stats_path
        self.precomputed_mu = None
        self.precomputed_sigma = None

        self.test_by_train_data = test_by_train_data
        
        # load the precomputed stats if specified
        if self.use_precomputed_stats and self.precomputed_stats_path:
            self.load_precomputed_stats()

    def load_precomputed_stats(self):
        """fid_stats_imagenet256_guided_diffusion.npz 파일에서 사전 계산된 통계 로드"""
        if not os.path.exists(self.precomputed_stats_path):
            print(f"Warning: Precomputed stats file {self.precomputed_stats_path} does not exist.")
            return
        
        print(f"Loading precomputed stats from {self.precomputed_stats_path}")
        stats = np.load(self.precomputed_stats_path)
        self.precomputed_mu = torch.from_numpy(stats['mu']).to(self.device).double()
        self.precomputed_sigma = torch.from_numpy(stats['sigma']).to(self.device).double()
        print(f"Loaded precomputed stats: mu shape {self.precomputed_mu.shape}, sigma shape {self.precomputed_sigma.shape}")
        self.true_features_computed = True

    def compute_and_log_metrics(self, module):
        with torch.no_grad():
            if self.save_only:
                self.save_images(module, module.train_data) # will push pseudo data here.
                return
            elif module.args.compute_recon_fid:
                self.compute_images_features_reconstruction(module, module.test_data)
            elif module.args.compute_validation_fid:
                self.compute_images_features_validate(module, module.test_data)
            elif module.args.compute_train_recon_fid:
                self.compute_images_features_reconstruction(module, module.train_data)        
            elif self.test_by_train_data:
                imagenet_labeled  = np.load(f"VIRTUAL_imagenet256_labeled.npz", allow_pickle=True)
                prdc_train_data = torch.from_numpy(imagenet_labeled['arr_0']).permute(0, 3, 1, 2).float() / 255.0
                label = [i % 1000 for i in range(10000)]
                data_train = TrainDataset(prdc_train_data, label)
                    
                train_sampler, _ = module.get_distributed_sampler(data_train, data_train)

                train_loader = DataLoader(data_train, batch_size=50,
                                        shuffle=False,
                                        num_workers=module.args.num_workers, pin_memory=True,
                                        drop_last=False, sampler=train_sampler)
                if module.args.test_with_train_x0:
                    self.compute_images_features_from_reflow_dataset(module, train_loader)
                else:
                    self.compute_images_features(module, train_loader)
            else:
                self.compute_images_features(module, module.test_data)
        if self.use_precomputed_stats and self.precomputed_mu is not None and self.precomputed_sigma is not None:
            self.inception_metrics.precomputed_mu = self.precomputed_mu
            self.inception_metrics.precomputed_sigma = self.precomputed_sigma

        metrics = self.inception_metrics.compute()
        metrics = {f"Eval/{k}": v for k, v in metrics.items()}
        print(metrics)
        return metrics
    
    # TODO:
    def compute_images_features_validate(self, module, dataloader):
        if len(dataloader.dataset) < self.num_images:
            max_images = len(dataloader.dataset)
        else:
            max_images = self.num_images
            
        desc = "Computing images features"
        bar = tqdm(dataloader, leave=False, desc=desc)
        for i, (images, labels) in enumerate(bar):
                
            # compute real image features
            real = remap_image_torch(images)
            self.inception_metrics.update(real.to(self.device),
                                          labels.to(self.device),
                                          image_type="real")
            
            self.inception_metrics.update(real.to(self.device),
                                          labels.to(self.device),
                                          image_type="conditional")
        
            

    def compute_images_features_reconstruction(self, module, dataloader):
        if len(dataloader.dataset) < self.num_images:
            max_images = len(dataloader.dataset)
        else:
            max_images = self.num_images
            
        print(f"Computing images features with reconstruction #{max_images}")
        desc = "Computing images features"
        bar = tqdm(dataloader, leave=False, desc=desc)

        num_images = 0
        
        for i, (images, labels) in enumerate(bar):
            if num_images >= max_images:
                break
                
            # compute real image features
            real = remap_image_torch(images)
            self.inception_metrics.update(real.to(self.device),
                                          labels.to(self.device),
                                          image_type="real")
            
            x = images.to(self.device)
            x = 2 * x - 1
            emb, _, [_, _, code] = module.ae.encode(x)
            code = code.view(x.shape[0], module.patch_size, module.patch_size)
            fake = module.ae.decode_code(torch.clamp(code, 0,  module.codebook_size-1))
            fake = fake.float()
            fake = fake.clamp(-1, 1) * 0.5 + 0.5
            fake = remap_image_torch(fake.cpu())
            self.inception_metrics.update(fake.to(self.device),
                                        labels.to(self.device),
                                        image_type="conditional")
            num_images += images.shape[0]


    def compute_images_features(self, module, dataloader):
        if len(dataloader.dataset) < self.num_images:
            max_images = len(dataloader.dataset)
        else:
            max_images = self.num_images
            
        desc = "Computing images features"
        bar = tqdm(dataloader, leave=False, desc=desc)
        
        for i, (images, labels) in enumerate(bar):
                
            # compute real image features
            real = remap_image_torch(images)
            self.inception_metrics.update(real.to(self.device),
                                          labels.to(self.device),
                                          image_type="real")
            
            with torch.no_grad():
                if isinstance(labels, list):
                    labels = clip.tokenize(labels[random.randint(0, 4)]).to(self.device)
                    labels = module.clip.encode_text(labels).float()
                else:
                    labels = labels.to(self.device)
                
                # generate fake image labels: copy existing labels 5 times
                fake_labels = labels.clone()
                nb_sample = images.size(0)

                for _ in range(5):
                    if module.args.interpolate:
                        init_code_rand = torch.randint(0, module.codebook_size, (nb_sample, module.patch_size, module.patch_size))
                        init_code_mask = torch.full((nb_sample, module.patch_size, module.patch_size), module.args.mask_value)
                        # Generate init_code by mixing random and masked code
                        init_code = torch.where(torch.rand(nb_sample, module.patch_size, module.patch_size) < module.args.interpolate_rate, init_code_mask, init_code_rand).to(module.args.device)
                    else:
                        init_code = None
                    
                    images = module.sample(init_code=init_code,
                                        nb_sample=nb_sample,
                                        labels=fake_labels,
                                        sm_temp=module.args.sm_temp,
                                        w=module.args.cfg_w,
                                        randomize="linear",
                                        r_temp=module.args.r_temp,
                                        sched_mode=module.args.sched_mode,
                                        step=module.args.step)[0]
                    images = images.float()
                    images = images.clamp(-1, 1) * 0.5 + 0.5
                    fake = remap_image_torch(images.cpu())
                    self.inception_metrics.update(fake.to(self.device),
                                                fake_labels,
                                                image_type="conditional")
                
                if self.save_dir is not None:
                    # save images
                    for j, (r, f, l) in enumerate(zip(real, fake, labels)):
                        label = int(l.item())
                        if torch.distributed.is_initialized():
                            rank = torch.distributed.get_rank()
                            img_id = f'{rank}_{i}_{j}'
                        else:
                            img_id = f'{i}_{j}'
                        # label as 08d format
                        os.makedirs(os.path.join(self.save_dir, "real", f'{label:08d}'), exist_ok=True)
                        os.makedirs(os.path.join(self.save_dir, "fake", f'{label:08d}'), exist_ok=True)
                        r = r.permute(1, 2, 0).cpu().numpy()
                        f = f.permute(1, 2, 0).cpu().numpy()

                        # convert RGB to BGR
                        r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
                        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)

                        # save image
                        cv2.imwrite(os.path.join(self.save_dir, "real", f'{label:08d}', f"{img_id}.png"), r)
                        cv2.imwrite(os.path.join(self.save_dir, "fake", f'{label:08d}', f"{img_id}.png"), f)

            # barrier
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        if self.use_precomputed_stats and self.precomputed_mu is not None and self.precomputed_sigma is not None:
            self.inception_metrics.precomputed_mu = self.precomputed_mu
            self.inception_metrics.precomputed_sigma = self.precomputed_sigma

        
    def save_images(self, module, dataloader):
        desc = "Generating images for vis"
        bar = tqdm(dataloader, leave=False, desc=desc)
        
        for i, (images, labels) in enumerate(bar):
            # compute real image features
            with torch.no_grad():
                if isinstance(labels, list):
                    labels = clip.tokenize(labels[random.randint(0, 4)]).to(self.device)
                    labels = module.clip.encode_text(labels).float()
                else:
                    labels = labels.to(self.device)
                
                # 가짜 이미지 label 생성: 기존 label을 5배로 복사
                fake_labels = labels.clone()
                nb_sample = images.size(0)
                if module.args.interpolate:
                    init_code_rand = torch.randint(0, module.codebook_size, (nb_sample, module.patch_size, module.patch_size))
                    init_code_mask = torch.full((nb_sample, module.patch_size, module.patch_size), module.args.mask_value)
                    # Generate init_code by mixing random and masked code
                    init_code = torch.where(torch.rand(nb_sample, module.patch_size, module.patch_size) < module.args.interpolate_rate, init_code_mask, init_code_rand).to(module.args.device)
                else:
                    init_code = None
                
                images = module.sample(init_code=init_code,
                                    nb_sample=nb_sample,
                                    labels=fake_labels,
                                    sm_temp=module.args.sm_temp,
                                    w=module.args.cfg_w,
                                    randomize="linear",
                                    r_temp=module.args.r_temp,
                                    sched_mode=module.args.sched_mode,
                                    step=module.args.step)[0]
                images = images.float()
                images = images.clamp(-1, 1) * 0.5 + 0.5
                fake = remap_image_torch(images.cpu())
                self.inception_metrics.update(fake.to(self.device),
                                            fake_labels,
                                            image_type="conditional")
                
                if self.save_dir is not None:
                    for j, (f, l) in enumerate(zip(fake, labels)):
                        label = int(l.item())
                        if torch.distributed.is_initialized():
                            rank = torch.distributed.get_rank()
                            img_id = f'{rank}_{i}_{j}'
                        else:
                            img_id = f'{i}_{j}'
                        # label as 08d format
                        os.makedirs(os.path.join(self.save_dir, "fake", f'{l}', f'{label:08d}'), exist_ok=True)
                        f = f.permute(1, 2, 0).cpu().numpy()
                        # convert RGB to BGR
                        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                        # save image
                        cv2.imwrite(os.path.join(self.save_dir, "fake", f'{l}', f'{label:08d}', f"{img_id}.png"), f)

            # barrier
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
