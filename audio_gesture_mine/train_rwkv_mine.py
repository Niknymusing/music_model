import os
import random
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime


# Set Environment Variables
os.environ["RWKV_CTXLEN"] = '128'
os.environ["RWKV_HEAD_SIZE_A"] = '64' # Ensure this is consistent with head_size_a in args
os.environ["RWKV_FLOAT_MODE"] = 'bf16' # Change to bfloat16 to match CUDA kernel expectations
os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_T_MAX"] = "256"
#os.environ["RWKV_MY_TESTING"] = "x060" # Uncomment this if using the wkv6 CUDA kernel


import argparse
from argparse import Namespace
import pytorch_lightning as pl
from src.model import RWKV


# Configure args
args = Namespace(
    n_embd=128,
    vocab_size=100,  # Adjust to your actual vocabulary size
    n_layer=6,
    dim_att=128,
    dim_ffn=256,
    tiny_att_layer=-1,
    tiny_att_dim=-1,
    dropout=0,
    head_qk=0,
    layerwise_lr=1,
    my_pile_stage=0,
    weight_decay=0.01,
    ctx_len=128,
    lr_init=6e-4,
    accelerator='GPU',
    my_pos_emb=0,
    pre_ffn=0,
    head_size_a=64,  # Ensure this matches RWKV_HEAD_SIZE_A environment variable
    head_size = 64,
    n_head = 1,
    head_size_divisor=1,
    grad_cp=0,
    betas=(0.9, 0.999),
    adam_eps=1e-8,
    precision = 'bf16'  # Match precision with RWKV_FLOAT_MODE
)
model = RWKV(args)


# Load the checkpoint file
checkpoint_path = "/content/drive/MyDrive/rwkv_mine_training/checkpoints/epoch=9-step=115200.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

# Update model state
model.load_state_dict(checkpoint['state_dict'])
# Instantiate the model
#model = RWKV(args)

model.train()
print(sum([p.numel() for p in model.parameters() if p.requires_grad == True])) #### count model parameters

checkpoint_callback = ModelCheckpoint(
    dirpath='/content/drive/MyDrive/rwkv_mine_training/checkpoints',
    filename='{epoch}-{step}',
    every_n_train_steps=300,  # Save a checkpoint every 300 training steps
    save_top_k=-1,            # Set to -1 to keep all checkpoints
    save_last=True            # Optionally, save the last checkpoint at the end of training
)


session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

# Base directory for checkpoints
base_checkpoint_dir = '/content/drive/MyDrive/rwkv_mine_training/checkpoints'

# Create a new directory for this session's checkpoints
session_checkpoint_dir = os.path.join(base_checkpoint_dir, session_id)
os.makedirs(session_checkpoint_dir, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=session_checkpoint_dir,  # Use the session-specific path
    monitor='val_loss',
    filename='{epoch}-{step}-{val_loss:.2f}',
    every_n_train_steps=300,
    save_top_k=1,
    mode='min'
)


# Configure the TensorBoardLogger
logger = TensorBoardLogger(
    save_dir='/content/drive/MyDrive/rwkv_mine_training',
    name='logs'
)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=10,
    precision="bf16",
    callbacks=[checkpoint_callback],
    logger=logger
)

# Assuming model is defined and initialized
# Assuming train_loader and val_loader are defined as per the previous code snippet

# Fit the model using the training and validation data loaders
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)