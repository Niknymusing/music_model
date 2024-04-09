import sys
import os
sys.path.append(os.getcwd()+'/models')
from models.rave.model import RAVE
from models.rave.blocks import EncoderV2, WasserteinEncoder, VariationalEncoder, GeneratorV2
from models.rave.pqmf import CachedPQMF
from models.rwkv.src.model import RWKV
#from rwkv.model import RWKV
#from models.rwkv_pip_package.src.rwkv.model import RWKV
import torch
from torch.nn import init
import time
from torch import nn
from spiralnet import instantiate_model as instantiate_spiralnet 


#######################################

# Objective : maximize;
# MI[ encoding(current_pose, current_audio_buffer) , next_recorded_audio_buffer ] + human_feedback_reward

#######################################

KERNEL_SIZE = 3
DILATIONS = [
    [1, 3, 9],
    [1, 3, 9],
    [1, 3, 9],
    [1, 3],
]
RATIOS = [4, 4, 4, 2]
CAPACITY = 96
NOISE_AUGMENTATION = 0
LATENT_SIZE = 16
N_BAND = 16

pqmf = CachedPQMF(n_band = N_BAND, attenuation = 100)
encoder = EncoderV2(data_size = N_BAND, capacity = CAPACITY, ratios = RATIOS, 
                    latent_size = LATENT_SIZE, n_out = 1, kernel_size = KERNEL_SIZE, 
                    dilations = DILATIONS) 




class SpiralnetRAVEClassifierGRU(nn.Module):
    def __init__(self, nr_of_classes, embedding_dim=16, nr_spiralnet_layers=4, nr_rnn_layers=2):
        super(SpiralnetRAVEClassifierGRU, self).__init__()
        self.nr_of_gesture_classes = nr_of_classes
        self.embedding_dim = embedding_dim
        self.audio_encoder = encoder
        self.pqmf = pqmf 
        self.spiralnet = instantiate_spiralnet(nr_layers=nr_spiralnet_layers, output_dim= self.embedding_dim)
        self.layer_norm = nn.LayerNorm(2 * self.embedding_dim)
        self.gru = nn.GRU(2 * self.embedding_dim, 2 * self.embedding_dim, nr_rnn_layers, bidirectional=False, batch_first=False)
        self.gelu = nn.GELU()
        self.output_values_ff_list = nn.ModuleList([nn.Linear(2 * self.embedding_dim, 5) 
                                                    for _ in range(self.nr_of_gesture_classes)])
        self.fc = nn.Linear(2 * self.embedding_dim, self.nr_of_gesture_classes)
        self.softmax = nn.Softmax(dim=0)
        

        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)

    def forward(self, pose_tensor, audio_buffer):
        pose_embedding = self.spiralnet(pose_tensor)
        audio_embedding = self.audio_encoder(self.pqmf(audio_buffer)).squeeze(2)
        x = torch.concat((pose_embedding, audio_embedding), dim=1)
        #print('x.shape', x.shape)
        x = self.layer_norm(x)
        x, _ = self.gru(x)
        x = self.gelu(x[-1])
        logits = self.fc(x)  # use this as joint embedding for the MINE
        class_prediction = torch.argmax(logits).item()
        values = self.output_values_ff_list[class_prediction](x)
        softmax_values = self.softmax(values)
        #values = 127 * torch.sigmoid(values) 
        return logits, class_prediction, softmax_values
    

class SpiralnetRaveRWKV(nn.Module):
    def __init__(self, embedding_dim=256, nr_spiralnet_layers=4, rwkv_path='/Users/nikny/Downloads/rwkvstatedict.pth', rwkv_strategy='cpu fp32'):
        super(SpiralnetRaveRWKV, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.rwkv_path = rwkv_path
        self.rwkv_strategy = rwkv_strategy
        self.audio_encoder = EncoderV2(data_size = 16, capacity = 64, ratios = [4, 4, 4, 2], 
                    latent_size = self.embedding_dim, n_out = 1, kernel_size = 3, 
                    dilations = [[1, 3, 9],[1, 3, 9], [1, 3, 9],[1, 3],]) 
        self.pqmf = CachedPQMF(n_band = 16, attenuation = 100)
        self.spiralnet = instantiate_spiralnet(nr_layers=nr_spiralnet_layers, output_dim= self.embedding_dim)
        self.layer_norm = nn.LayerNorm(2 * self.embedding_dim)
        self.rwkv =  RWKV(model=self.rwkv_path, strategy=self.rwkv_strategy)
        

    def forward(self, pose_tensor, audio_buffer, rwkv_state):
        pose_embedding = self.spiralnet(pose_tensor)
        pqmf = self.pqmf(audio_buffer)
        audio_embedding = self.audio_encoder(pqmf).squeeze(2)
        x = torch.concat((pose_embedding, audio_embedding), dim=1)                                                                        
        print('encoder forward x.shape', x.shape)
        x = self.layer_norm(x)
        x, emb, state = self.rwkv(x, rwkv_state)
       
        #softmax_values = self.softmax(values)
        #values = 127 * torch.sigmoid(values) 
        #downsampled_embedding_for_transmission = F.avg_pool1d(pqmf, kernel_size=4, stride=4)
        return x, emb, state, pqmf#softmax_values#, downsampled_embedding_for_transmission