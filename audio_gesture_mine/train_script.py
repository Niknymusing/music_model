###### Correct process dataset and instantiate the dataloader, to load the tensors correctly formated locally on the colab instance
import os
#print("before setting environ var: CUDA_HOME:", os.environ['CUDA_HOME'])
# rest of your script

os.environ['CUDA_HOME'] = '/usr/local/cuda-12.3'
print("after setting CUDA_HOME:", os.environ['CUDA_HOME'])
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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
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
from multimodel import RWKV
#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#if args.precision == "fp32":
torch.backends.cudnn.allow_tf32 = True#False
torch.backends.cuda.matmul.allow_tf32 = True# False
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
# Example usage
data_dir = '/home/nikny/aist_tensors/aist_tensors'
all_pair_indices = list(range(10))  # Assume 10 pair indices for simplicity
batch_size = 1




# Set parameters
sequence_length = 64 # adapted to current model implementation
batch_size = 1 # can be bigger depending on your sequence length and the length of the videos in your dataset

# Define paths to your dataset of tensors created in the previous step


# Copy files from Drive to local storage
#file_names = os.listdir(temp_save_path)
#or file_name in tqdm.tqdm(file_names, desc="Copying files"):
#    shutil.copy(os.path.join(path, file_name), os.path.join(temp_save_path, file_name))

#print("All files have been copied to local storage.")
in_path = data_dir
out_path = os.getcwd() + '/training_data/batches5'
os.makedirs(out_path, exist_ok=True)

# Collect pairs of mocap and audio files
tensorfiles = os.listdir(in_path)
pairs = [(os.path.join(in_path, f), os.path.join(in_path, f.replace('_mocap', '_audio')))
         for f in tensorfiles if 'mocap' in f]


"""
print('creating raw tensors ...')

# Load and adjust tensors
tensors = []
pair_index = 0  # This will uniquely identify each pair
for mocap_path, audio_path in pairs:
    mocap_tensor = torch.load(mocap_path).squeeze(0)
    audio_tensor = torch.load(audio_path)
    if mocap_tensor.shape[0] > audio_tensor.shape[0]:
        mocap_tensor = mocap_tensor.transpose(0, 2)  # Change shape to (3, 33, N1)
        mocap_tensor = F.adaptive_avg_pool1d(mocap_tensor, audio_tensor.shape[0]).transpose(0, 2)  # Resize and revert shape
    tensors.append((mocap_tensor, audio_tensor, pair_index))
    pair_index += 1

# Forming sequences and batching
n_batches = 0

for mocap_tensor, audio_tensor, video_idx in tensors:
    total_sequences = min(mocap_tensor.shape[0], audio_tensor.shape[0]) // (sequence_length * 2)

    batch_num = 0
    for i in range(0, total_sequences, batch_size):
        sequences = []
        for j in range(i, min(i + batch_size, total_sequences)):
            start_index = j * sequence_length
            end_index = start_index + sequence_length
            start_index_future = end_index
            end_index_future = end_index + sequence_length
            sequences.append((mocap_tensor[start_index:end_index], audio_tensor[start_index:end_index], audio_tensor[start_index_future:end_index_future]))

        if len(sequences) == batch_size:
            mocap_batch, audio_batch, audio_future_batch = zip(*sequences)
            batch_path = os.path.join(out_path, f'pair_{video_idx}_batch_{batch_num}.pt')
            torch.save({'mocap': torch.stack(mocap_batch), 'audio': torch.stack(audio_batch), 'audio_future': torch.stack(audio_future_batch)}, batch_path)
            batch_num += 1
            n_batches+=1

print(f'Batches are successfully created and saved with unique pair indices. n_batches = ', n_batches)

"""

# Set parameters
sequence_length = 64
batch_size = 1

# Define paths to your dataset of tensors created in the previous step
in_path = data_dir
out_path = os.getcwd() + '/training_data/batches5'
os.makedirs(out_path, exist_ok=True)
"""
# Function to verify batch file integrity
def verify_batch(batch_path):
    try:
        data = torch.load(batch_path)
        keys_required = ['mocap', 'audio', 'audio_future']
        if not all(key in data for key in keys_required):
            raise ValueError(f"Data is missing one of the required keys: {keys_required}")
        return True
    except Exception as e:
        print(f"Error verifying batch file {batch_path}: {e}")
        return False

# Collect pairs of mocap and audio files
tensorfiles = os.listdir(in_path)
pairs = [(os.path.join(in_path, f), os.path.join(in_path, f.replace('_mocap', '_audio')))
         for f in tensorfiles if 'mocap' in f]

# Load and adjust tensors
tensors = []
for mocap_path, audio_path in pairs:
    mocap_tensor = torch.load(mocap_path).squeeze(0)
    audio_tensor = torch.load(audio_path)
    if mocap_tensor.shape[0] > audio_tensor.shape[0]:
        mocap_tensor = mocap_tensor.transpose(0, 2)
        mocap_tensor = F.adaptive_avg_pool1d(mocap_tensor, audio_tensor.shape[0]).transpose(0, 2)
    tensors.append((mocap_tensor, audio_tensor))

# Forming sequences and batching
n_batches = 0

for mocap_tensor, audio_tensor in tensors:
    total_sequences = min(mocap_tensor.shape[0], audio_tensor.shape[0]) // (sequence_length * 2)
   
    batch_num = 0
    for i in range(0, total_sequences, batch_size):
        sequences = []
        for j in range(i, min(i + batch_size, total_sequences)):
            start_index = j * sequence_length
            end_index = start_index + sequence_length
            start_index_future = end_index
            end_index_future = end_index + sequence_length
            sequences.append((mocap_tensor[start_index:end_index], audio_tensor[start_index:end_index], audio_tensor[start_index_future:end_index_future]))

        if len(sequences) == batch_size:
            mocap_batch, audio_batch, audio_future_batch = zip(*sequences)
            batch_path = os.path.join(out_path, f'pair_{i}_batch_{batch_num}.pt')
            torch.save({'mocap': torch.stack(mocap_batch), 'audio': torch.stack(audio_batch), 'audio_future': torch.stack(audio_future_batch)}, batch_path)
            if not verify_batch(batch_path):
                os.remove(batch_path)
                print(f"Removed corrupted batch file: {batch_path}")
            else:
                batch_num += 1
                n_batches += 1

print(f'Batches are successfully created and saved with unique pair indices. n_batches = ', n_batches)

"""

print('instantiating DataModule ....')

"""


from scipy.interpolate import interp1d
from torch.utils.data import Dataset

class _VideoDataset(Dataset):
    def __init__(self, directory, video_indices, is_train=True):
        super().__init__()
        self.directory = directory
        self.is_train = is_train
        self.video_indices = video_indices
        self.audio_data = {}



        self.mocap_tensors = {}
        self.audio_tensors = {}

        for video_name in video_indices:
            mocap_file = os.path.join(directory, f"{video_name}_mocap.pt")
            audio_file = os.path.join(directory, f"{video_name}_audio.pt")
            if os.path.isfile(mocap_file) and os.path.isfile(audio_file):
                mocap_tensor = torch.load(mocap_file).float().reshape(-1, 33, 3)
                audio_tensor = torch.load(audio_file).float()

                if mocap_tensor.shape[0] != audio_tensor.shape[0]:  # Sequence lengths differ
                    # Convert tensor to numpy for interpolation
                    mocap_np = mocap_tensor.numpy()
                    x_old = np.linspace(0, 1, mocap_np.shape[0])
                    x_new = np.linspace(0, 1, audio_tensor.shape[0])

                    # Prepare an array to collect interpolated data
                    interpolated_data = np.zeros((audio_tensor.shape[0], mocap_np.shape[1], mocap_np.shape[2]))

                    # Interpolate each feature independently
                    for i in range(mocap_np.shape[1]):  # For each channel
                        for j in range(mocap_np.shape[2]):  # For each feature within the channel
                            interpolator = interp1d(x_old, mocap_np[:, i, j], kind='cubic')
                            interpolated_data[:, i, j] = interpolator(x_new)

                    # Convert interpolated data back to torch tensor
                    mocap_tensor = torch.from_numpy(interpolated_data).float()

                self.mocap_tensors[video_name] = mocap_tensor
                self.audio_tensors[video_name] = audio_tensor
        
        
        if self.is_train:
            random.shuffle(self.video_indices)

    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, idx):
        video_name = self.video_indices[idx]
        mocap = self.mocap_tensors[video_name]
        audio = self.audio_tensors[video_name]

        marginal_indices = random.sample([i for i in self.video_indices if i != video_name], 2)
        marginal_audio = torch.cat([self.audio_tensors[idx] for idx in marginal_indices], dim=0).squeeze(0)
        #marginal_idx = random.choice([i for i in self.video_indices if i != video_name])
        
        #marginal_audio = self.audio_tensors[marginal_idx]

        return (mocap, audio, marginal_audio)
    
    def update_marginal_audio_indices(self):
        if not self.is_train:
            return
        for video_idx in self.video_indices:
            other_videos = [v for v in self.video_indices if v != video_idx]
            random_video_idx = random.choice(other_videos)
            random_audio_idx = random.randint(0, len(self.audio_data[random_video_idx]) - 1)
            self.marginal_audios_idxs[video_idx] = random_audio_idx
"""

class VideoDataset(Dataset):
    def __init__(self, directory, video_indices, is_train=True):
        super().__init__()
        self.directory = directory
        self.is_train = is_train
        self.video_indices = video_indices
        self.mocap_tensors = {}
        self.audio_tensors = {}

        for video_name in video_indices:
            mocap_file = os.path.join(directory, f"{video_name}_mocap.pt")
            audio_file = os.path.join(directory, f"{video_name}_audio.pt")
            if os.path.isfile(mocap_file) and os.path.isfile(audio_file):
                self.mocap_tensors[video_name] = torch.load(mocap_file)
                self.audio_tensors[video_name] = torch.load(audio_file)
                logging.debug(f"Loaded {video_name} data.")
            else:
                logging.warning(f"Missing data for {video_name}.")

        if self.is_train:
            random.shuffle(self.video_indices)
        logging.debug(f"Initializing dataset with {len(video_indices)} videos.")

    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, idx):
        video_name = self.video_indices[idx]
        logging.debug(f"Fetching data for {video_name}.")
        mocap = self.mocap_tensors[video_name]
        audio = self.audio_tensors[video_name]
        marginal_indices = random.sample([i for i in self.video_indices if i != video_name], 2)
        marginal_audio = torch.cat([self.audio_tensors[idx] for idx in marginal_indices], dim=0).squeeze(0)
        logging.debug(f"shapes of data in batch for {video_name} is {mocap.shape, audio.shape, marginal_audio.shape} ")
        return (mocap, audio, marginal_audio)



class VideoDataModule(pl.LightningDataModule):
    def __init__(self, directory, batch_size=1, train_ratio=0.8):
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.audio_data = {}
        self.marginal_audios_idxs = {}


        all_files = [f.replace('_mocap.pt', '').replace('_audio.pt', '') for f in os.listdir(directory) if f.endswith('.pt')]
        

        video_indices = list(set(all_files))  # Ensure unique video indices without extensions
        random.shuffle(video_indices)
        split_idx = int(len(video_indices) * train_ratio)

        self.train_indices = video_indices[:split_idx]
        self.val_indices = video_indices[split_idx:]

        self.train_dataset = VideoDataset(directory, self.train_indices, is_train=True)
        self.val_dataset = VideoDataset(directory, self.val_indices, is_train=False)

        num_train_batches = len(self.train_dataset) // self.batch_size + (1 if len(self.train_dataset) % self.batch_size > 0 else 0)
        num_val_batches = len(self.val_dataset) // self.batch_size + (1 if len(self.val_dataset) % self.batch_size > 0 else 0)

        print(f"Loaded train batches: {num_train_batches}")
        print(f"Loaded validation batches: {num_val_batches}")

        

    def train_dataloader(self):
        logging.debug("Creating train dataloader.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        logging.debug("Creating validation dataloader.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def on_train_epoch_end(self, outputs):
        # Reshuffle training dataset indices at the end of each epoch
        random.shuffle(self.train_dataset.video_indices)

data_module = VideoDataModule(directory='/home/nikny/aist_training_tensors_aligned', batch_size=1)







"""def inspect_val_batches(data_module):
    val_loader = data_module.val_dataloader()
    for i, batch in enumerate(val_loader):
        if batch is None:
            print(f"Batch {i} is None or faulty.")
        else:
            mocap, audio, audio_future, audio_marg = batch
            print(f"Batch {i}: Mocap Shape: {mocap.shape}, Audio Shape: {audio.shape}")
            # Add more detailed inspection as necessary

# Usage
inspect_val_batches(data_module)
"""




"""
def inspect_val_batches(data_module):
    # Fetch the validation dataloader
    val_loader = data_module.val_dataloader()

    # Iterate over each batch in the validation DataLoader
    batch_counter = 0
    for batch in val_loader:
        # Check if the batch is None (which shouldn't happen with the custom collate function)
        if batch is None:
            print(f"Batch {batch_counter} is empty")
            continue
        
        # Print the batch details
        mocap, audio, audio_future, audio_marg = batch
        print(f"Batch {batch_counter}:")
        print(f"  Mocap Tensor Shape: {mocap.shape}")
        print(f"  Audio Tensor Shape: {audio.shape}")
        print(f"  Audio Future Tensor Shape: {audio_future.shape}")
        print(f"  Audio Marginal Tensor Shape: {audio_marg.shape}")
        batch_counter += 1

    print(f"Total batches inspected: {batch_counter}")
"""
"""def inspect_val_batches(data_module, expected_shape=None, expected_dtype=None):
    # Fetch the validation dataloader
    val_loader = data_module.val_dataloader()

    # Define expected shapes and types if not provided (adjust according to your specific needs)
    if expected_shape is None:
        expected_shape = {'mocap': (1, 1, 64, 33, 3), 'audio': (1, 1, 64, 1, 2048), 'audio_future': (1, 1, 64, 1, 2048), 'audio_marg': (1, 1, 64, 1, 2048)}
    if expected_dtype is None:
        expected_dtype = {'mocap': torch.float32, 'audio': torch.float32, 'audio_future': torch.float32, 'audio_marg': torch.float32}

    # Iterate over each batch in the validation DataLoader
    batch_counter = 0
    for batch in val_loader:
        if batch is None:
            print(f"Batch {batch_counter} is empty or missing")
            continue

        # Unpack the batch
        mocap, audio, audio_future, audio_marg = batch
        
        # Initialize flag to check for anomalies
        anomaly_detected = False

        # Check the shapes and data types
        if mocap.shape != expected_shape['mocap'] or mocap.dtype != expected_dtype['mocap']:
            anomaly_detected = True
        if audio.shape != expected_shape['audio'] or audio.dtype != expected_dtype['audio']:
            anomaly_detected = True
        if audio_future.shape != expected_shape['audio_future'] or audio_future.dtype != expected_dtype['audio_future']:
            anomaly_detected = True
        if audio_marg.shape != expected_shape['audio_marg'] or audio_marg.dtype != expected_dtype['audio_marg']:
            anomaly_detected = True

        # Print details if there's an anomaly
        if anomaly_detected:
            print(f"Batch {batch_counter} detected with unexpected shapes or data types:")
            print(f"  Mocap Tensor Shape: {mocap.shape}, Type: {mocap.dtype}")
            print(f"  Audio Tensor Shape: {audio.shape}, Type: {audio.dtype}")
            print(f"  Audio Future Tensor Shape: {audio_future.shape}, Type: {audio_future.dtype}")
            print(f"  Audio Marginal Tensor Shape: {audio_marg.shape}, Type: {audio_marg.dtype}")

        batch_counter += 1

    print(f"Total batches inspected: {batch_counter}")"""

# Example usage, you might want to adjust the expected shapes and types based on your model's input specifications
#inspect_val_batches(data_module)


# Assuming you have a data_module instance ready
# Example usage:
#inspect_val_batches(data_module)


print('datamodule instantiated')



# Configure args
args = Namespace(n_time_scales = 4,
                 n_frequency_features = 4,
    n_embd=128,
    embedding_dim = 64,
    time_scales = [1, 4, 16, 64],
    vocab_size=100,  # currently not training the head weights of the rwkv just the embedding MI to the inputs, so can set this to something small for now
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
    vocab_size=100,  # currently not training the head weights of the rwkv just the embedding MI to the inputs, so can set this to something small for now
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

# Create a new directory for this session's checkpoints
#training_checkpoint_dir = os.path.join(base_checkpoint_dir, session_id)
#os.makedirs(training_checkpoint_dir, exist_ok=True)
#validation_checkpoint_dir = os.path.join(base_checkpoint_dir, session_id)
#os.makedirs(validation_checkpoint_dir, exist_ok=True)



"""train_checkpoint_callback = ModelCheckpoint(
    dirpath=base_checkpoint_dir,
    monitor='train_loss',
    filename='train-{epoch}-{step}-{train_loss:.2f}',
    save_top_k=1,
    mode='min',
    every_n_train_steps=10000,
    save_weights_only=False
)

val_checkpoint_callback = ModelCheckpoint(
    dirpath=base_checkpoint_dir,
    monitor='val_loss',
    filename='val-{epoch}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_weights_only=False
)

# Configure the TensorBoardLogger
logger = TensorBoardLogger(
    save_dir='/home/nikny/music_model/training_data',
    name='logs'
)
"""
"""trainer = pl.Trainer(
    accelerator='gpu',
    devices=-1,
    max_epochs=1000,
    precision="bf16",
    callbacks=[train_checkpoint_callback, val_checkpoint_callback],
    logger=logger
)
"""

# Assuming model is defined and initialized
# Assuming train_loader and val_loader are defined as per the previous code snippet

# Fit the model using the training and validation data loaders


# Instantiate the model
# Instantiate the model
model = RWKV(args, args_head)
model.generate_init_weight()
model.train()

print('nr of parameters  = ', sum(p.numel() for p in model.parameters() if p.requires_grad))






"""





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
)"""
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




import os


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


import time
from pytorch_lightning import Callback

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
