import torch
import numpy as np
import types
from torch import nn
from torch.nn import functional as F
import time
import torch.nn.init as init
#from src.models.s4.s4 import S4
#from src.models.sequence.ss.s4 import S4
import sys
import os
sys.path.append(os.getcwd()+'/src/rave')
from itertools import product
#from models.spiralnet import instantiate_model as instantiate_spiralnet 

#from interface.dancestrument.utils.midi_functions import send_notes

from rave.blocks import EncoderV2
from rave.pqmf import CachedPQMF
#from multimodal_model import SpiralnetRAVEClassifierGRU

#from models.rwkv.src.rwkv.model import RWKV

class RunningMineMean:
    def __init__(self, device):
        self.sum_joints = torch.tensor(0.0, requires_grad=True)
        self.sum_margs = torch.tensor(0.0, requires_grad=True)
        self.count = torch.tensor(0.0, requires_grad=True)

    def update(self, y_joint, y_marg):
        # Avoiding in-place operations and ensuring requires_grad is propagated
        updated_sum_joints = self.sum_joints + y_joint
        updated_sum_margs = self.sum_margs + y_marg
        updated_count = self.count + 1

        # Calculate the running mean without modifying the original tensors
        running_mean = updated_sum_joints / updated_count - torch.log(torch.exp(updated_sum_margs / updated_count))

        # Update the running sums and count
        self.sum_joints = updated_sum_joints.detach().requires_grad_()
        self.sum_margs = updated_sum_margs.detach().requires_grad_()
        self.count = updated_count.detach().requires_grad_()

        return running_mean.detach().requires_grad_()


class MultiscaleSequence_MINE(nn.Module):
    def __init__(self, time_scales_past: list, 
                 time_scales_future: list, 
                 latent_input_dim, 
                 latent_dim):
        super(MultiscaleSequence_MINE, self).__init__()

        self.time_scales_past = time_scales_past
        self.time_scales_future = time_scales_future
        self.nr_past_timescales = len(self.time_scales_past)
        self.nr_future_timescales = len(self.time_scales_future)
        self.latent_input_dim = latent_input_dim
        self.latent_dim = latent_dim

        self.audio_encoder = EncoderV2(data_size = 16, capacity = 24, ratios = [16, 8], 
                            latent_size = self.latent_dim//2, n_out = 1, kernel_size = 3, 
                            dilations = [[1, 3, 9],[1, 3]]) 

        #self.audio_encoder = AudioEncoder()  # Make sure AudioEncoder is defined somewhere
        self.up_proj = nn.ModuleList([nn.Linear(self.latent_input_dim*2, self.latent_dim) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))])
        self.linear_proj1 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))]) 
        self.linear_proj2 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))]) 
        self.down_proj = nn.ModuleList([nn.Linear(self.latent_dim, 1) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))])
        #self.running_means = [RunningMineMean() for _ in range(self.nr_future_timescales * self.nr_past_timescales)]
        device = next(self.parameters()).device
        self.running_means = [RunningMineMean(device=device) for _ in range(self.nr_future_timescales * self.nr_past_timescales)]


    def downsample_pasts(self, input_tensor):
        scales = [1, 4, 16, 64]  # The scales you specified
        downsampled_tensors = []
        for scale in scales:
            if scale == 1:
                # Just append the last matrix for scale 1
                downsampled_tensors.append(input_tensor[-1].unsqueeze(0))
            else:
                # Select the last 'scale' matrices and concatenate them along the 0th dimension
                selected_matrices = input_tensor[-scale:]

                # Permute the dimensions to fit adaptive_avg_pool1d requirements (batch, channel, length)
                permuted_matrices = selected_matrices.permute(1, 2, 0)

                # Downsample the concatenated sequence to the shape (1, 1, 16)
                # This involves reducing the 0th dimension from 'scale' to 1
                # while preserving the other dimensions (1 and 16).
                downsampled = F.adaptive_avg_pool1d(permuted_matrices, 1)

                # Ensure the output is correctly shaped (1, 1, 16)
                downsampled = downsampled.permute(2, 0, 1)  # Back to (1, 1, 16)
                downsampled_tensors.append(downsampled)
        return downsampled_tensors

    def downsample_futures(self, input_tensor):
        scales = [1, 4, 16, 64]  # The scales you specified
        downsampled_tensors = []
        for scale in scales:
            if scale == 1:
                # Just append the last matrix for scale 1
                downsampled_tensors.append(input_tensor[-1].unsqueeze(0))
            else:
                # Select the last 'scale' matrices and concatenate them along the 0th dimension
                selected_matrices = input_tensor[-scale:]

                # Permute the dimensions to fit adaptive_avg_pool1d requirements (batch, channel, length)
                permuted_matrices = selected_matrices.permute(1, 2, 0)

                # Downsample the concatenated sequence to the shape (16, 128, 1)
                # This involves reducing the 0th dimension from 'scale' to 1
                # while preserving the other dimensions (16 and 128).
                downsampled = F.adaptive_avg_pool1d(permuted_matrices, 1)

                # Ensure the output is correctly shaped (1, 16, 128)
                downsampled = downsampled.permute(2, 0, 1)  # Back to (1, 16, 128)
                downsampled_tensors.append(downsampled)
        return downsampled_tensors

    def forward(self, embs, audio_joints, audio_margs):
        # Encoding
        # audio
        # maybe implement rwkv instead of GRU in the generative model 
        # use pytorch lightning for the traning loop 
        #current_mine_count = running_mean[0]
        #current_mine_count = mine_values[0]
        outputs = []
        embs = self.downsample_pasts(embs)
        audio_joints = self.downsample_futures(audio_joints) 
        audio_margs = self.downsample_futures(audio_margs)
        #print('downsampled embs tensors shape : ', embs[0].shape)

        encoded_joints = [self.audio_encoder(joint) for joint in audio_joints]
        #print('encoded_joints.shape=', encoded_joints[0].shape)
        encoded_margs = [self.audio_encoder(marg) for marg in audio_margs]
        #print('encoded_margs.shape = ', encoded_margs[0].shape)
        idx = 0
        
        for i, j in product(range(self.nr_past_timescales), range(self.nr_future_timescales)):
            
            mine_mean = self.running_means[idx]
            z_joint, z_marg = encoded_joints[j], encoded_margs[j]
            #print('z_joint shape', z_joint.shape, 'z_marg shape', z_marg.shape)
            #print('audio_embs[i].shape, z_joint.shape' , audio_embs[i].shape, z_joint.shape)
            #print('z_marg.shape: ', z_marg.shape)
            #print('embs[i].shape, z_joint.shape = ', embs[i].shape, z_joint.shape, flush=True)
            y_joint = torch.cat((embs[i].squeeze(2), z_joint.squeeze(2)), dim=1)
            y_marg = torch.cat((embs[i].squeeze(2), z_marg.reshape(1, self.latent_input_dim)), dim=1)


            #print('y_joint, y_marg shapes', y_joint.shape, y_marg.shape)
            y_joint = F.relu(self.up_proj[idx](y_joint))
            #print('y_joint up', y_joint.shape)
            y_joint = F.relu(self.linear_proj1[idx](y_joint))
            y_joint = F.relu(self.linear_proj2[idx](y_joint))
            #print('y_joint lin', y_joint.shape)
            y_joint = F.relu(self.down_proj[idx](y_joint)).squeeze()
            y_marg = F.relu(self.up_proj[idx](y_marg))
            y_marg = F.relu(self.linear_proj1[idx](y_marg))
            y_marg = F.relu(self.linear_proj2[idx](y_marg))
            y_marg = F.relu(self.down_proj[idx](y_marg)).squeeze()
            #print(y_joint, y_marg)
            outputs.append(mine_mean.update(y_joint, y_marg))
            idx += 1
            
        
        #print(len(outputs))
        return outputs
