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
torch.backends.cudnn.allow_tf32 = True#False
torch.backends.cuda.matmul.allow_tf32 = True# False
import tqdm  # Using tqdm to show progress
import logging
#from logger_config import setup_logging
import deeplake

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
#model = RWKV(args)
#model.generate_init_weight()

# Load the checkpoint file
#checkpoint_path = "/content/drive/MyDrive/rwkv_mine_training/checkpoints/epoch=9-step=115200.ckpt"
#checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

# Update model state
#model.load_state_dict(checkpoint['state_dict'])
# Instantiate the model
#model = RWKV(args)

#model.train()
#print(sum([p.numel() for p in model.parameters() if p.requires_grad == True])) #### count model parameters

session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

# Base directory for checkpoints
base_checkpoint_dir = '/home/nikny/music_model/training_data'#'/content/drive/MyDrive/rwkv_mine_training/checkpoints3'


# Instantiate the model
model = RWKV(args, args_head)
model.generate_init_weight()
model.train()

print('nr of parameters  = ', sum(p.numel() for p in model.parameters() if p.requires_grad))




from torch.utils.data import Dataset, DataLoader
import torch

class VideoDataDataset(Dataset):
    def __init__(self, base_path):
        super(VideoDataDataset, self).__init__()
        self.dataset_paths = [os.path.join(base_path, name) for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
        self.data = []
        for path in self.dataset_paths:
            dl_dataset = deeplake.load(path)  # Assuming 'load' is the correct method to access the dataset
            for idx in range(len(dl_dataset['audio'])):
                self.data.append((dl_dataset['audio'][idx], dl_dataset['motion'][idx], dl_dataset['audio_ahead'][idx], dl_dataset['audio_marginal'][idx]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, motion, audio_ahead, audio_marginal = self.data[idx]
        return torch.tensor(audio), torch.tensor(motion), torch.tensor(audio_ahead), torch.tensor(audio_marginal)

# Usage
dataset = VideoDataDataset('/content/drive/MyDrive/AIST_deep_lake/dataset_parts')
loader = DataLoader(dataset, batch_size=10, shuffle=True)









# Check if a checkpoint file exists
best_checkpoint_path = None
if val_checkpoint_callback.best_model_path:
    best_checkpoint_path = val_checkpoint_callback.best_model_path
elif train_checkpoint_callback.best_model_path:
    best_checkpoint_path = train_checkpoint_callback.best_model_path

loaded_epoch = None
loaded_global_step = None

if best_checkpoint_path:
    print(f"Loading best model from checkpoint: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Get the epoch and global step from the checkpoint
    loaded_epoch = checkpoint['epoch']
    loaded_global_step = checkpoint['global_step']
    
    print(f"Resuming training from epoch: {loaded_epoch}, global step: {loaded_global_step}")
else:
    print("No checkpoint found. Starting training from scratch.")

# Configure the TensorBoardLogger
logger = TensorBoardLogger(
    save_dir='/home/nikny/music_model/training_data/logs2',
    name='logs',
    version=loaded_epoch if best_checkpoint_path else None  # Use the loaded epoch as the version if checkpoint exists
)

# Create the trainer
# Create the trainer

trainer = pl.Trainer(
    accelerator='gpu',
    devices=-1,
    max_epochs=100000,
    precision="bf16",
    callbacks=[train_checkpoint_callback, val_checkpoint_callback],
    logger=logger
)

profiler = PyTorchProfiler(
    profile_memory=True,  # This enables memory profiling
    with_stack=True,      # This provides additional stack tracing
    record_shapes=True,   # Records input shapes
    profiled_functions=["forward", "training_step", "validate_step"]  # Specific functions to profile
)

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
    },
    "profiling": {
        "enabled": True,
        "start_step": 5,
        "num_steps": 10,
        "output_dir": "./deepspeed_profiling",
        "timeline": True
    }
}

class MemoryDebuggingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print("\nCUDA Memory Summary at Epoch End:")
        print(torch.cuda.memory_summary(device=pl_module.device, abbreviated=False))
        
    def on_train_end(self, trainer, pl_module):
        print("\nFinal CUDA Memory Summary:")
        print(torch.cuda.memory_summary(device=pl_module.device, abbreviated=False))


# Create the Trainer with the DeepSpeed strategy
import logging
#logging.basicConfig(level=logging.DEBUG) 

_trainer = pl.Trainer(profiler=profiler,
    accelerator='gpu',
    devices=-1,
    max_epochs=100000,
    precision="bf16",
    callbacks=[train_checkpoint_callback, val_checkpoint_callback, MemoryDebuggingCallback()],
    enable_progress_bar=False ,
    logger=True,  # Ensures that logging is enabled
     # Ensures the progress bar updates regularly
)
"""checkpoint_callback_post_validation = ModelCheckpoint(
    monitor='val_loss',  # Assuming you have a 'val_loss' metric
    dirpath='./checkpoints/post_validation/',
    filename='model-post-val-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,  # Only saves the top 1 checkpoint based on val_loss
    save_last=True,  # Save the last checkpoint for each training epoch
    every_n_epochs=1,  # Save checkpoints at the end of these many epochs
    save_on_train_epoch_end=False  # Ensure this saves after validation
)

class PreValidationCheckpoint(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        epoch = trainer.current_epoch
        dirpath = './checkpoints/pre_validation/'
        os.makedirs(dirpath, exist_ok=True)
        filename = f'model-pre-val-epoch={epoch:02d}.ckpt'
        ckpt_path = os.path.join(dirpath, filename)
        trainer.save_checkpoint(ckpt_path)
        print(f"Checkpoint saved before validation at: {ckpt_path}")
#signal_handler = pl.callbacks.SignalHandler(terminate_on_interrupt=True)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=-1,  # Use all available GPUs
    max_epochs=100,
    precision="bf16",
    enable_progress_bar=True,
    logger=True,
    callbacks=[
        
        checkpoint_callback_post_validation,
        PreValidationCheckpoint()  # Adds the custom callback for pre-validation checkpointing
    ]
)

trainer.fit(model, datamodule=data_module)

"""
# Initialize WandbLogger with the log_every_n_steps parameter
wandb_logger = WandbLogger(
    name="Experiment-Name",
    project="mine-estimation",
    entity="musing"
     # Log metrics every 16 steps
)

checkpoint_callback_post_validation = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkpoints/post_validation/',
    filename='model-post-val-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    save_last=True,
    every_n_epochs=1,
    save_on_train_epoch_end=False
)




# Directory where the checkpoints will be saved
checkpoint_dir = '/home/nikny/training_local/checkpoints'

# Ensure the directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(epoch, model, optimizer):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')



class PauseTrainingCallback(Callback):
    def __init__(self, pause_duration=3600, run_duration=57600):
        """
        Args:
            pause_duration (int): Duration to pause training in seconds. Default is 3600 seconds (1 hour).
            run_duration (int): Duration to run training before pausing in seconds. Default is 57600 seconds (16 hours).
        """
        self.pause_duration = pause_duration
        self.run_duration = run_duration
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        """Record the start time when training begins."""
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Check the time at the end of each batch to determine if we need to pause."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= self.run_duration:
            print(f"Pausing training for {self.pause_duration} seconds after running for {self.run_duration} seconds.")
            time.sleep(self.pause_duration)
            print("Resuming training.")
            self.start_time = current_time + self.pause_duration - elapsed_time  # Reset start time accounting for the pause





class PreValidationCheckpoint(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        epoch = trainer.current_epoch
        # Updated directory path to save checkpoints in /home/nikny/training_local
        dirpath = '/home/nikny/training_local/pre_validation/'
        os.makedirs(dirpath, exist_ok=True)
        filename = f'model-pre-val-epoch={epoch:02d}.ckpt'
        ckpt_path = os.path.join(dirpath, filename)
        trainer.save_checkpoint(ckpt_path)
        print(f"Checkpoint saved before validation at: {ckpt_path}")

checkpoint_callback = ModelCheckpoint(
    dirpath='/home/nikny/training_local/checkpoints/',
    filename='model-step-{step:08d}',
    every_n_train_steps=1000,
    save_top_k=-1,  # Save all checkpoints, you can also set this to a fixed number
    save_last=True  # Optionally save the last checkpoint at the end of training
)


pause_callback = PauseTrainingCallback()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=-1,
    max_epochs=100,
    precision="bf16",
    enable_progress_bar=True,
   logger=wandb_logger,  # Use the initialized WandbLogger here
    callbacks=[pause_callback, checkpoint_callback,
        checkpoint_callback_post_validation,
        PreValidationCheckpoint()
    ],
    log_every_n_steps=32 
)

trainer.fit(model, datamodule=data_module)


"""# Set the initial epoch and global step for the trainer (if applicable)
if loaded_epoch is not None and loaded_global_step is not None:
    trainer.fit_loop.epoch_progress.current.completed = loaded_epoch
    trainer.fit_loop.global_step = loaded_global_step
"""
# Fit the model using the training and validation data loaders
#trainer.fit(model, datamodule=data_module, ckpt_path=best_checkpoint_path)


"""try:
    # Fit the model using the training and validation data loaders
    trainer.fit(model, datamodule=data_module)
except RuntimeError as e:
    print(f"An error occurred: {str(e)}")
    # Optionally, handle specific errors, e.g., out of memory
    if "out of memory" in str(e):
        print("Out of GPU memory.")
finally:
    # Print profiling results even if an error occurs
    print(profiler.summary())
    # Save profiling results to a file or further processing
    profiler.describe()
    with open("profiling_summary.txt", "w") as file:
        file.write(profiler.summary())

    # Optionally, cleanup or further error handling
    print("Cleaning up or handling after an error or successful run.")"""
