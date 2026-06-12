# -*- coding: utf-8 -*-
"""Smoke test local: dataset fake + 2 épocas em CPU, wandb offline."""
import os
import shutil

import numpy as np
from PIL import Image
import wandb

os.environ["WANDB_MODE"] = "offline"
root = "tmp_data"
shutil.rmtree(root, ignore_errors=True)
for cls in ["NORMAL", "PNEUMONIA"]:
    d = os.path.join(root, "train", cls)
    os.makedirs(d)
    for i in range(12):
        nome = f"person{i}_bacteria_1.jpeg" if cls == "PNEUMONIA" else f"IM-{i:04d}.jpeg"
        Image.fromarray((np.random.rand(64, 80) * 255).astype("uint8")).save(os.path.join(d, nome))

cfg = dict(data_root=root, channels=1, img_size=64, batch_size=4, num_workers=0,
           use_dp=False, val_frac=0.2, test_frac=0.2, base=16, zdim=16,
           epochs=2, lr_g=2e-4, lr_d=2e-4, beta=0.01, beta_warm_epochs=1,
           disc_start_epoch=1, l_perc=0, l_adv=0.7, amp=True,
           ckpt_every=1, resume=os.environ.get("SMOKE_RESUME") == "1", seed=42)
if cfg["resume"]:
    cfg["epochs"] = 3  # ckpt parou na 2 -> deve retomar e rodar só a 3
wandb.init(project="smoke", mode="offline", config=cfg)

from experiments.vae_gan.experiment import VaeGan
print("OK:", VaeGan(dict(wandb.config)).run())
wandb.finish()
