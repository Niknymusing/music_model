import os
import sys
sys.path.append('/Users/nikny/gestures/models/rwkv/RWKV/RWKVv4neo')
sys.path.append('/Users/nikny/gestures/models/rwkv/RWKV/RWKVv4neo/cuda')
import torch
import time
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_T_MAX"] = "1024"
os.environ["RWKV_FLOAT_MODE"] = "bf16"
from types import SimpleNamespace as sn
#from src.model import RWKV
from src.model import RWKV
from types import SimpleNamespace as sn
args = sn(FLOAT_MODE = 'bf16', RUN_DEVICE='cuda', n_embd=512, n_layer=16, 
          vocab_size = 64800, my_pos_emb=0, pre_ffn = 0, ctx_len = 64, dropout = 0, 
          head_qk=0, lr_init = 0.001, accelerator = 'GPU', grad_cp=0)
model = RWKV(args, strategy='cpu fp32')
model.generate_init_weight()
number_of_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
torch.save(model.state_dict(), f='rwkv_statedict.pth')
params = {}
i = 0
for p in model.parameters():
    params[i] = p
    i+=1
torch.save(params, f='rwkv_params.pth')
print('RWKV-v4neo model instantiated, number of parameters =', number_of_parameters)
#x = torch.tensor([[1, 2, 3, 4],[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4] ])
x=torch.rand(10, 3, args.n_embd) # shape (batch_size, number_of_tokens, n_embds)
y = model(x)
t = time.time()
x=torch.rand(1, 3, args.n_embd)
y = model(x)
t=time.time()-t
print('output shape : ', y.shape, 'inference time ;', t )