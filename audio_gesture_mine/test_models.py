###### Correct process dataset and instantiate the dataloader, to load the tensors correctly formated locally on the colab instance
import os
#print("before setting environ var: CUDA_HOME:", os.environ['CUDA_HOME'])
# rest of your script

cuda_path = '/usr/local/cuda'
os.environ['CUDA_HOME'] = cuda_path

# Now, print out the variable to verify it's correctly set
print("CUDA_HOME set to:", os.environ['CUDA_HOME'])
# rest of your script

import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
#### instantiate the model and start the pytorch lightning trainer
from pytorch_lightning.callbacks import Callback
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger#, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from datetime import datetime
import time
# Set Environment Variables
os.environ["RWKV_CTXLEN"] = '128'
os.environ["RWKV_HEAD_SIZE_A"] = '64' # Ensure this is consistent with head_size_a in args
os.environ["RWKV_FLOAT_MODE"] = 'bf16' # Change to bfloat16 to match CUDA kernel expectations
os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_T_MAX"] = "256"
#os.environ["RWKV_MY_TESTING"] = "x060" # Uncomment this if using the wkv6 CUDA kernel
import argparse
from argparse import Namespace
from multimodel import RWKV
#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#if args.precision == "fp32":
#torch.backends.cudnn.allow_tf32 = True#False
#torch.backends.cuda.matmul.allow_tf32 = True# False
import tqdm  # Using tqdm to show progress
import logging
#from logger_config import setup_logging



# Initialize WandbLogger

# Setup logger
#setup_logging()
#logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
logging.info("Starting the training process...")


torch.cuda.empty_cache()



# Configure args
args = Namespace(n_time_scales = 4,
                 n_frequency_features = 4,
    n_embd=128,
    embedding_dim = 64,
    time_scales = [1, 4, 16, 64],
    vocab_size=2048,  # currently not training the head weights of the rwkv just the embedding MI to the inputs, so can set this to something small for now
    n_layer=8,
    dim_att=128,
    dim_ffn=256,
    tiny_att_layer=-1,
    tiny_att_dim=-1,
    dropout=0,
    head_qk=0,
    layerwise_lr=0,
    my_pile_stage=0,
    weight_decay=0.01,
    ctx_len=128,
    lr_init=6e-5, #6e-1,
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



args_head = Namespace(n_time_scales = 4,
                 n_frequency_features = 4,
    n_embd=256,
    embedding_dim = 256,
    time_scales = [1, 4, 16, 64],
    vocab_size=16,  # currently not training the head weights of the rwkv just the embedding MI to the inputs, so can set this to something small for now
    n_layer=8,
    dim_att=128,
    dim_ffn=256,
    tiny_att_layer=-1,
    tiny_att_dim=-1,
    dropout=0,
    head_qk=0,
    layerwise_lr=0,
    my_pile_stage=0,
    weight_decay=0.01,
    ctx_len=128,
    lr_init=6e-5, #6e-1,
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

model = RWKV(args, args_head)
model.generate_init_weight()
model.train()

print('nr of parameters  = ', sum(p.numel() for p in model.parameters() if p.requires_grad))