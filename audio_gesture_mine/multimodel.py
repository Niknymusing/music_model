import os, math, gc, importlib, random, time
from random import randint
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW, SGD
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from torch.optim.lr_scheduler import LambdaLR
from scipy.spatial.distance import cdist

from autoclip.torch import QuantileClip
from collections import deque
import itertools
#from deepspeed.runtime.lr_schedules import OneCycleLR
# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob

from typing import Tuple
import sys
sys.path.append(os.getcwd()+'/src/rave')
sys.path.append('/content/rwkv_mine/audio_gesture_mine/src/rave')
sys.path.append('/content/rwkv_mine/audio_gesture_mine/src')
import gin
gin.enter_interactive_mode()
from rave.blocks import EncoderV2
from mine import MultiscaleSequence_MINE, RunningMineMean
from rave.pqmf import CachedPQMF
from spiralnet import instantiate_model as instantiate_spiralnet
print('new imports worked')
from collections import deque

MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
#from logger_config import setup_logging
import logging

# Setup logger
setup_logging()
logger = logging.getLogger(__name__)

########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.contiguous().to(dtype=torch.bfloat16)
            k = k.contiguous().to(dtype=torch.bfloat16)
            v = v.contiguous().to(dtype=torch.bfloat16)
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            return y
        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)
else:
    wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return y
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return y.half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return y.bfloat16()
        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)

########################################################################################################

class RWKV_TimeMix_RWKV5_Preview(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = 64
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = 8

        self.chunk_len = 512
        assert args.ctx_len % self.chunk_len == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            if 'r3' in os.environ["RWKV_MY_TESTING"]:
                self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)

            # fancy time_decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            if 'r2' in os.environ["RWKV_MY_TESTING"]:
                tmp = torch.zeros(self.n_head)
                for h in range(self.n_head):
                    tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
                self.time_faaaa = nn.Parameter(tmp)
            else:
                self.time_first = nn.Parameter(torch.ones(self.n_head) * (-3.0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    if 'r3' in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def jit_func(self, x):
            B, TT, C = x.size()

            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

            r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
            k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
            v = self.value(xv).view(B, TT, self.n_head, -1).transpose(1, 2)                 # BTC -> BHTS
            g = F.silu(self.gate(xg))

            return r, k, v, g

        @MyFunction
        def jit_func_2(self, r, k, v, g, w, wk, wb, ws):
            B, H, TT, S = r.size()
            T = self.chunk_len

            s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
            x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

            for i in range(TT // T):
                rr = r[:, :, i*T:i*T+T, :]
                kk = k[:, :, :, i*T:i*T+T]
                vv = v[:, :, i*T:i*T+T, :]

                x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

                s = ws * s + (kk * wk) @ vv
           
            x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
            x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S) * g
            return self.output(x)
    else:
        @MyFunction
        def jit_func(self, x):
            B, TT, C = x.size()

            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

            r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
            k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
            v = self.value(xv).view(B, TT, self.n_head, self.head_size).transpose(1, 2)                 # BTC -> BHTS

            return r, k, v

        @MyFunction
        def jit_func_2(self, r, k, v, w, wk, wb, ws):
            B, H, TT, S = r.size()
            T = self.chunk_len

            s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
            x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

            for i in range(TT // T):
                rr = r[:, :, i*T:i*T+T, :]
                kk = k[:, :, :, i*T:i*T+T]
                vv = v[:, :, i*T:i*T+T, :]

                x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

                s = ws * s + (kk * wk) @ vv
           
            x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
            x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S)
            return self.output(x)
   
    def forward(self, x):
        #print('yooo')
        H = self.n_head
        T = self.chunk_len

        if 'r3' in os.environ["RWKV_MY_TESTING"]:
            r, k, v, g = self.jit_func(x)
        else:
            r, k, v = self.jit_func(x)

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)
       
        if 'r2' in os.environ["RWKV_MY_TESTING"]:
            u = self.time_faaaa.float().unsqueeze(-1)
        else:
            u = torch.exp(self.time_first.float()).unsqueeze(-1)

################################################################################
########
        ws = w.pow(T).reshape(1, H, 1, 1)

        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)

        wk = w.reshape(1, H, 1, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(1, H, T, T)
########
################################################################################

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)
        if 'r3' in os.environ["RWKV_MY_TESTING"]:
            return self.jit_func_2(r, k, v, g, w, wk, wb, ws)
        else:
            return self.jit_func_2(r, k, v, w, wk, wb, ws)        


########################################################################################################
# CUDA RWKV5 Kernel
########################################################################################################

if 'r4' in os.environ["RWKV_MY_TESTING"]:
    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
    wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
       
    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                eew = (torch.exp(ew)).contiguous()
                ctx.save_for_backward(r, k, v, eew, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
        return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_TimeMix_RWKV5(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
       
        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
           
            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        if 'a' in os.environ["RWKV_MY_TESTING"]:
            self.register_buffer("att_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
            d_qkv = args.n_embd // 16
            self.qq = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.kk = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.vv = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.oo = nn.Linear(d_qkv, args.n_embd, bias=False)
            with torch.no_grad():
                self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_vv = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)

    if 'a' not in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def jit_func(self, x):
            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            return sr, k, v

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v = self.jit_func(x)
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
            return self.output(rwkv)

    if 'a' in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def QKV(self, q, k, v):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.att_mask == 0, float('-inf'))
            att = F.softmax(att, dim = -1)
            x = att @ v
            return x

        @MyFunction
        def jit_funcQKV(self, x):
            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
            xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
            xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            qq = self.qq(xqq)
            kk = self.kk(xkk)
            vv = self.vv(xvv)
            return sr, k, v, qq, kk, vv

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v, qq, kk, vv = self.jit_funcQKV(x)
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
            rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))
            return rwkv

########################################################################################################

class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
       
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if 'r4' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_TimeMix_RWKV5(args, layer_id)
            elif 'r' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_TimeMix_RWKV5_Preview(args, layer_id)
            else:
                self.att = RWKV_TimeMix(args, layer_id)

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)
       
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
       
    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb

        if self.args.dropout == 0:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)
   

def print_grad_hook(module, grad_input, grad_output):
    # Check if any gradients are None
    grad_input_none = any(g is None for g in grad_input)
    grad_output_none = any(g is None for g in grad_output)

    if grad_input_none or grad_output_none:
        # Print the module name and its unique identifier
        print(f"Module: {module.__class__.__name__}, id: {id(module)}")

        # Check and print grad_input details
        if grad_input_none:
            print("  Grad Input: None detected")
            for idx, g in enumerate(grad_input):
                if g is None:
                    print(f"    grad_input[{idx}]: None")
                else:
                    print(f"    grad_input[{idx}]: Exists, shape: {g.shape}, requires_grad: {g.requires_grad}")

        # Check and print grad_output details
        if grad_output_none:
            print("  Grad Output: None detected")
            for idx, g in enumerate(grad_output):
                if g is None:
                    print(f"    grad_output[{idx}]: None")
                else:
                    print(f"    grad_output[{idx}]: Exists, shape: {g.shape}, requires_grad: {g.requires_grad}")

        # Extra information for debugging
        if hasattr(module, 'weight'):
            weight_requires_grad = module.weight.requires_grad if module.weight is not None else 'No weight'
            print(f"  Module Weight requires_grad: {weight_requires_grad}")
        if hasattr(module, 'bias'):
            bias_requires_grad = module.bias.requires_grad if module.bias is not None else 'No bias'
            print(f"  Module Bias requires_grad: {bias_requires_grad}")


def full_backward_hook(module, grad_input, grad_output):
    print(f"Module: {module.__class__.__name__}, id: {id(module)}")

    # Print details of grad_input (gradients of inputs to the module)
    for idx, g in enumerate(grad_input):
        if g is None:
            print(f"  grad_input[{idx}]: None")
        else:
            print(f"  grad_input[{idx}]: shape: {g.shape}, requires_grad: {g.requires_grad}, norm: {g.norm()}")

    # Print details of grad_output (gradients of outputs from the module)
    for idx, g in enumerate(grad_output):
        if g is None:
            print(f"  grad_output[{idx}]: None")
        else:
            print(f"  grad_output[{idx}]: shape: {g.shape}, requires_grad: {g.requires_grad}, norm: {g.norm()}")




EPS = 1e-6
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean

class SpiralnetRAVEncoder(nn.Module):
    def __init__(self, deq, embedding_dim = 64, nr_spiralnet_layers = 4, n_out = 1):
        super(SpiralnetRAVEncoder, self).__init__()
       
        self.embedding_dim = embedding_dim
        self.audio_encoder = EncoderV2(data_size = N_BAND, capacity = CAPACITY, ratios = RATIOS,
                    latent_size = LATENT_SIZE, n_out = n_out, kernel_size = KERNEL_SIZE,
                    dilations = DILATIONS)
        self.pqmf = CachedPQMF(n_band = N_BAND, attenuation = 100)
        self.spiralnet = instantiate_spiralnet(nr_layers=nr_spiralnet_layers, output_dim= self.embedding_dim)
        self.layer_norm = nn.LayerNorm(2 * self.embedding_dim)
        self.prev_embeddings = deq
   
    #def pqmf(self, x):
    #    return self.pqmf(x)

    def forward(self, pose_tensor, audio_buffer):
        #print('pose_tensor.shape', pose_tensor.shape)
        pose_embedding = self.spiralnet(pose_tensor)
        #print('pose_embedding.shape', pose_embedding.shape)
        audio_embedding = self.audio_encoder(self.pqmf(audio_buffer)).squeeze(2)
        x = torch.concat((pose_embedding, audio_embedding), dim=1)
        #print('x.shape', x.shape)
        #print('encoder forward x.shape', x.shape)
        x = self.layer_norm(x).unsqueeze(0)
        #self.prev_embeddings.append(x)
        #print('x.shape', x.shape)
       
        #values = 127 * torch.sigmoid(values)
        return x#.to(dtype=torch.bfloat16)



KERNEL_SIZE = 3
DILATIONS = [
    [1, 3, 9],
    [1, 3, 9],
    [1, 3, 9],
    [1, 3],
]
RATIOS = [4, 4, 4, 2]
CAPACITY = 64#96#64#
NOISE_AUGMENTATION = 0
LATENT_SIZE = 256
N_BAND = 16#LATENT_SIZE# 16


class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim = 64, n_out = 4):
        super(AudioEncoder, self).__init__()
        self.pqmf = CachedPQMF(n_band = N_BAND, attenuation = 100)
        self.encoder = EncoderV2(data_size = N_BAND, capacity = CAPACITY, ratios = RATIOS,
                    latent_size = embedding_dim, n_out = n_out, kernel_size = KERNEL_SIZE,
                    dilations = DILATIONS)
    def forward(self, audio_buffer):
        pqmf_emb = self.pqmf(audio_buffer)
        audio_embedding = self.encoder(pqmf_emb).squeeze(2)
        return audio_embedding
   
class PoseEncoder(nn.Module):
    def __init__(self, embedding_dim = 64, n_spiralnet_layers = 4):
        super(PoseEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.spiralnet = instantiate_spiralnet(nr_layers=n_spiralnet_layers, output_dim= self.embedding_dim)

    def forward(self, pose_tensor):
        pose_embedding = self.spiralnet(pose_tensor)
        return pose_embedding
   
class MineAudioEncoder(nn.Module):
    def __init__(self, embedding_dim = 64, n_out = 1):
        super(MineAudioEncoder, self).__init__()
        self.pqmf = CachedPQMF(n_band = 16, attenuation = 100)
        self.encoder = EncoderV2(data_size = 16, capacity = 32, ratios = [16, 8],
                    latent_size = embedding_dim, n_out = n_out, kernel_size = 3,
                    dilations =  [[1, 3, 9],[1, 3],])
    def forward(self, audio_buffer):
        pqmf_emb = self.pqmf(audio_buffer)
        audio_embedding = self.encoder(pqmf_emb).squeeze(2)
        return audio_embedding
   
class MINE_net(nn.Module):
    def __init__(self, input_dim = 64, up_dim = 400):
        super(MINE_net, self).__init__()
        self.in_dim = 3 * input_dim
        self.h_dim = up_dim
        #self.mine_net = nn.ModuleList([])
        self.up_proj = nn.Linear(self.in_dim, self.h_dim)
        self.vertical_proj = nn.Linear(self.h_dim, self.h_dim)
        self.down_proj1 = nn.Linear(self.h_dim, self.h_dim // 2)
        self.down_proj2 = nn.Linear(self.h_dim // 2, 1)
        self.audio_encoder = MineAudioEncoder()

    def forward(self, emb_tensor, audio_tensor_joint, audio_tensor_marg):

        #print('emb_joint, emb_marg ',torch.linalg.norm(audio_tensor_joint - audio_tensor_marg))
        audio_tensor_joint = self.audio_encoder(audio_tensor_joint)
        audio_tensor_marg = self.audio_encoder(audio_tensor_marg)
        #print('emb_joint, emb_marg ',torch.linalg.norm(audio_tensor_joint - audio_tensor_marg))
        emb_joint = torch.concat((emb_tensor, audio_tensor_joint), dim=1)
        emb_marg = torch.concat((emb_tensor, audio_tensor_marg), dim=1)
        #print('emb_joint, emb_marg ',torch.linalg.norm(emb_joint - emb_marg))
        #print(' /////////////////////////////////// ',emb_joint.shape, emb_marg.shape)
        #print('emb_joint, emb_marg before relus',emb_joint, emb_marg)
        joint_t = F.relu_(self.up_proj(emb_joint))
        joint_t = F.relu_(self.vertical_proj(joint_t))
        joint_t = F.relu_(self.down_proj1(joint_t))
        joint_t = F.relu_(self.down_proj2(joint_t))

        marg_t = F.relu_(self.up_proj(emb_marg))
        marg_t = F.relu_(self.vertical_proj(marg_t))
        marg_t = F.relu_(self.down_proj1(marg_t))
        marg_t = F.relu_(self.down_proj2(marg_t))
        #print('emb_joint, emb_marg after relus',torch.linalg.norm(emb_joint - emb_marg))
        return joint_t, marg_t





class LoggedModule(nn.Module):
    def __init__(self, module, name):
        super().__init__()
        self.module = module
        self.name = name

    def forward(self, x):
        output = self.module(x)
        # Logging the output statistics
        print(f"{self.name} - Output Mean: {output.mean().item()}, Std: {output.std().item()}, Min: {output.min().item()}, Max: {output.max().item()}")
        return output


from collections import defaultdict, deque

import torch
import torch.nn as nn

class MaskedModel(nn.Module):
    def __init__(self, L, dim1_keeps, dim2_segments):
        super(MaskedModel, self).__init__()
        self.masks = nn.ParameterList()  # Use ParameterList to store masks without gradients
        # Precompute masks
        for keep in dim1_keeps:
            for start, end in dim2_segments:
                mask = torch.zeros((L, 64, 256), dtype=torch.float32)
                mask[:, :keep, start:end] = 1
                self.masks.append(nn.Parameter(mask, requires_grad=False))

    def forward(self, x):
        masked_outputs = []
        for mask in self.masks:
            masked_outputs.append(x * mask)  # Element-wise multiplication
        return masked_outputs

# Constants
L = 80  # Size for L as per the user's requirement
dim1_keeps = [1, 4, 16, 64]
dim2_segments = [(i * 64, (i + 1) * 64) for i in range(4)]

# Create the model
model = MaskedModel(L, dim1_keeps, dim2_segments)

# Generate a random input tensor of the appropriate shape
x = torch.randn(L, 64, 256)

# Apply the model to get the masked tensors
masked_tensors = model(x)

# Print the shape of each masked tensor and inspect a few values
for i, masked_tensor in enumerate(masked_tensors):
    print(f"Mask {i + 1} shape: {masked_tensor.shape}")
    # Printing the non-zero values in the tensor for quick inspection
    non_zeros = masked_tensor.nonzero(as_tuple=False)
    print(f"Non-zero elements count for Mask {i + 1}: {non_zeros.size(0)}")
    # Print a few non-zero elements for inspection
    if non_zeros.size(0) > 0:
        sample_indices = non_zeros[:min(5, non_zeros.size(0))]  # Print up to 5 non-zero elements
        for idx in sample_indices:
            print(f"Mask {i + 1} Value at {idx.tolist()}: {masked_tensor[idx[0], idx[1], idx[2]]}")
    print()  # Newline for better separation of mask outputs


import torch.jit as jit

"""@jit.script
def weighted_sum_update(target_param, source_params, weights):
        target_param.data.zero_()
        for i in range(len(source_params)):
            target_param.data.add_(source_params[i].data, alpha=weights[i])

@jit.script
def update_output_model(source_models, target_model, weights):
        for i, target_param in enumerate(target_model.parameters()):
            source_params = [model.parameters()[i] for model in source_models]
            weighted_sum_update(target_param, source_params, weights)

def validate_model_structure(model):
    if not isinstance(model, nn.Module):
        print("Error: Model is not an instance of nn.Module")
        return

    for idx, submodule in enumerate(model.children()):
        print(f"Submodule {idx}: {submodule.__class__.__name__}")
        for p_idx, param in enumerate(submodule.parameters()):
            print(f"  Param {p_idx}: {param.size()}")

    print("Overall parameter access test:")
    try:
        for p_idx, param in enumerate(model.parameters()):
            print(f"  Global Param {p_idx}: {param.size()}")
    except Exception as e:
        print(f"Error accessing parameters: {e}")
"""
import wandb

def forward_hook(module, input, output):
    try:
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if out is not None:
                    output_mean = out.data.mean()
                    output_std = out.data.std()
                    wandb.log({
                        f'{module.__class__.__name__}_output_{i}_mean': output_mean,
                        f'{module.__class__.__name__}_output_{i}_std': output_std
                    })
        else:
            output_mean = output.data.mean()
            output_std = output.data.std()
            wandb.log({
                f'{module.__class__.__name__}_output_mean': output_mean,
                f'{module.__class__.__name__}_output_std': output_std
            })
    except Exception as e:
        print(f"Error in forward_hook for {module.__class__.__name__}: {e}")


# Define a hook to capture gradients
def backward_hook(module, grad_input, grad_output):
    try:
        grad_input_mean = torch.mean(torch.stack([torch.mean(g) for g in grad_input if g is not None]))
        grad_output_mean = torch.mean(torch.stack([torch.mean(g) for g in grad_output if g is not None]))
        wandb.log({
            f'{module.__class__.__name__}_grad_input_mean': grad_input_mean,
            f'{module.__class__.__name__}_grad_output_mean': grad_output_mean
        })
    except Exception as e:
        print(f"Error in backward_hook for {module.__class__.__name__}: {e}")


class RWKV(pl.LightningModule):
    def __init__(self, args, args_head):
        super().__init__()
        self.args = args
        self.args_head = args_head
        self.n_time_scales = args.n_time_scales
        self.n_frequency_features = args.n_frequency_features
        # Set default attributes if not present
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = args.n_embd * 4
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        # Basic model components
        self.n_parameters = 0
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        #self.n_parameters += sum(p.numel for p in self.emb.parameters if p.requires_grad)
        self.ln_out = nn.LayerNorm(args.n_embd)
        #self.n_parameters += sum(p.numel for p in self.ln_out.parameters if p.requires_grad)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        #self.n_parameters += sum(p.numel for p in self.head.parameters if p.requires_grad)
        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

        # Create multiple instances of blocks based on time and frequency scales
        self.embedding_dim = args.embedding_dim
        self.embdim_step = self.embedding_dim #// self.n_frequency_features # ensure self.n_frequency_features is a multiple of self.embedding_dim
        if not args.time_scales:
            self.time_scales = [1, 4, 16, 64]
        else:
            self.time_scales = args.time_scales
        self.vertical_feature_indices = list(range(len(self.time_scales)))      
        self.feature_indices = list(itertools.product(self.time_scales, self.vertical_feature_indices))  # list of all combinations of vertical & horizontal feature indices        
        self.n_time_scales = len(self.time_scales)
        self.n_frequency_features = len(self.vertical_feature_indices)
        self.n_features = self.n_time_scales * self.n_frequency_features

        self.masks2 = nn.ParameterList()
        self.dim2_segments = [(i * self.embedding_dim, (i + 1) * self.embedding_dim) for i in range(self.n_frequency_features)]
        for keep in self.time_scales:
            for start, end in dim2_segments:
                mask = torch.zeros((1, self.embedding_dim, self.n_frequency_features * self.embedding_dim), dtype=torch.float32)
                mask[:, :keep, start:end] = 1
                self.masks2.append(nn.Parameter(mask, requires_grad=False))

        self.masks = self.create_masks()
        print('self.masks[0].shape, len(self.masks) ----------' , self.masks[0].shape, len(self.masks))

        #self.nr_mine_values = self.nr_future_timescales * self.nr_past_timescales
        self.rwkv_subnetworks = nn.ModuleList([
            nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
            for _ in range(self.n_features)
        ])
       

        self.controller_network = nn.ModuleList([Block(self.args_head, i) for i in range(self.args_head.n_layer)])
        self.controller_head = nn.Linear(self.embedding_dim * self.n_frequency_features, self.n_features)

        self.output_model = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        
        #print('calling validation function : ')
        #validate_model_structure(self.rwkv_subnetworks)
        #validate_model_structure(self.output_model)


        self.layer_norms = [nn.LayerNorm(args.n_embd).cuda()  for _ in range(self.n_features + 2)]
        self.head_layer_norm = nn.LayerNorm(args_head.n_embd).cuda()
        #self.encoder = SpiralnetRAVEncoder(self.embeddings_buffer)#.to(device)#.to(self.device)
        # RWKV initialization with SimpleNamespace arguments
        # all model parameters instantiated below this line should be saved as mine statedict checkpoint
        self.audio_encoder = AudioEncoder(embedding_dim=self.embedding_dim, n_out = self.n_frequency_features).cuda()
        #self.mine_audio_encoder = AudioEncoder(embedding_dim=self.embedding_dim, n_out = self.n_frequency_features).cuda()
        self.pose_encoder = PoseEncoder(embedding_dim=self.embedding_dim).cuda()
        self.mine_nets = nn.ModuleList([MINE_net(input_dim=self.embedding_dim) for _ in range(self.n_features)]).cuda()  #MultiscaleSequence_MINE([1, 4, 16, 64], [1, 4, 16, 64], args.n_embd, 256)#.to(device)
       
        #self.mine_calculator = RunningMineMean()
        #self.register_buffer('accumulated_means', torch.zeros(self.n_features, 2).cuda() )
        #self.register_buffer('count', torch.tensor(1.0).cuda() )
        #self.register_buffer('joint_mean', torch.tensor(0).cuda() )
        #self.register_buffer('marg_mean', torch.tensor(0).cuda() )

        for param in self.parameters():
            param.requires_grad = True

        #self.joint_mean, self.marg_mean, self.count = torch.tensor(0).cuda() , torch.tensor(0).cuda() , torch.tensor(1.0).cuda() #float(0), float(0), int(1)
        self.joint_mean = torch.tensor([0.0], device='cuda')
        self.marg_mean = torch.tensor([0.0], device='cuda')
        self.count = torch.tensor([1.0], device='cuda')

        self.automatic_optimization = False
        self.training_mode = 'init_representation_learning_phase_1' #'init_representation_learning_phase_1''init_representation_learning_1'

        self.log_eps = 1e-10
        self.monitoring_steps = 16
        self.activations_stats = defaultdict(lambda: deque(maxlen=self.monitoring_steps))
        self.gradients_stats = defaultdict(lambda: deque(maxlen=self.monitoring_steps))
        self.attach_hooks()
        self.hooks = False##False#True#
        if self.hooks:
            self.apply_hooks()


        
        """new_state_dict = {}
        weights = list(range(16))
        print('len(self.mine_nets), len(weights) ', len(self.mine_nets), len(weights))
        
        for key in self.output_model.state_dict():
            # Initialize new state dict with zeros
            #new_state_dict[key]
            self.output_model.state_dict()[key] = torch.zeros_like(self.output_model.state_dict()[key])
            #print('output model keys', key)

            # Accumulate weighted state dict from all models
        t = time.time()  

        

        # Example usage
        # Assume `target_model` and `source_models` are instances of the same PyTorch model class
        # `weights` is a list of floats, each corresponding to a model in `source_models`
        t = time.time()  
        self.update_model_parameters_with_weighted_sum(self.output_model, self.rwkv_subnetworks, weights)
        t = time.time()-t
        print('state dict updated successfully in time update_model_parameters_with_weighted_sum(self.output_model, self.rwkv_subnetworks, weights)', t)

        with torch.no_grad():  # Ensures no gradient computations
            for target_param, source_param in zip(self.output_model.parameters(), self.rwkv_subnetworks[0].parameters()):
                target_param.data.copy_(source_param.data) 
        """
        """with torch.no_grad():
            for key in self.output_model.state_dict():
                
                
                self.output_model.state_dict()[key] = self.rwkv_subnetworks[0].state_dict()[key]
            for key in self.output_model.state_dict():
                for model, weight in zip(self.rwkv_subnetworks[1:], weights[1:]):
                        #print([key for key in model.state_dict()])
                        #if key not in model.state_dict():
                        #    raise KeyError(f"Key {key} not found in one of the models.")
                        self.output_model.state_dict()[key] += weight * model.state_dict()[key]
"""
        # Load the new state dict into the first model
        #self.output_model.load_state_dict(new_state_dict)
        
        """t = time.time()
        for key in self.output_model.state_dict():
            # Initialize new state dict with zeros
            new_state_dict[key] = torch.zeros_like(self.output_model.state_dict()[key])
            #print('output model keys', key)

            # Accumulate weighted state dict from all models
            for model, weight in zip(self.rwkv_subnetworks, weights):
                #print([key for key in model.state_dict()])
                if key not in model.state_dict():
                    raise KeyError(f"Key {key} not found in one of the models.")
                new_state_dict[key] += weight * model.state_dict()[key]

        # Load the new state dict into the first model
        self.output_model.load_state_dict(new_state_dict)
        t = time.time()-t"""
        #print('state dict updated successfully in time ', t)
    def attach_hooks(self):
        # Attach hooks to selected components
        self.audio_encoder.register_forward_hook(forward_hook)
        self.audio_encoder.register_backward_hook(backward_hook)
        self.pose_encoder.register_forward_hook(forward_hook)
        self.pose_encoder.register_backward_hook(backward_hook)
        
        for net in self.mine_nets:
            net.register_forward_hook(forward_hook)
            net.register_backward_hook(backward_hook)

        for layer in self.rwkv_subnetworks:
            for block in layer:
                block.register_forward_hook(forward_hook)
                block.register_backward_hook(backward_hook)

        for layer in self.controller_network:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)

        for layer in self.output_model:
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)

    def apply_hooks(self):
        # Apply hooks to each individual layer
        self.emb.apply(self.register_hooks)
        self.ln_out.apply(self.register_hooks)
        self.head.apply(self.register_hooks)
        if hasattr(self, 'head_q'):
            self.head_q.apply(self.register_hooks)
            self.head_k.apply(self.register_hooks)
        if hasattr(self, 'drop0'):
            self.drop0.apply(self.register_hooks)

        self.audio_encoder.apply(self.register_hooks)
        self.pose_encoder.apply(self.register_hooks)
        for mine_net in self.mine_nets:
            mine_net.apply(self.register_hooks)
        for subnetwork in self.rwkv_subnetworks:
            for block in subnetwork:
                block.apply(self.register_hooks)
        for block in self.controller_network:
            block.apply(self.register_hooks)

   
    def register_hooks(self, module):
        if not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict, pl.LightningModule)):
            module.register_forward_hook(self.activation_hook)
            module.register_backward_hook(self.gradient_hook)
   
   
   
   
   
    def activation_hook(self, mod, inp, out):
        """Stores activation statistics; supports both tensor and list/tuple of tensors."""
        if isinstance(out, torch.Tensor):
            self.activations_stats[f"{mod.__class__.__name__}"].append((out.data.mean().item(), out.data.std().item()))
        elif isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    self.activations_stats[f"{mod.__class__.__name__}[{i}]"].append((o.data.mean().item(), o.data.std().item()))

    def gradient_hook(self, mod, grad_input, grad_output):
        """Stores gradient statistics; supports both tensor and list/tuple of tensors."""
        if isinstance(grad_output, torch.Tensor):
            self.gradients_stats[f"{mod.__class__.__name__} output"].append((grad_output.mean().item(), grad_output.std().item()))
        elif isinstance(grad_output, (list, tuple)):
            for i, go in enumerate(grad_output):
                if isinstance(go, torch.Tensor):
                    self.gradients_stats[f"{mod.__class__.__name__}[{i}] output"].append((go.mean().item(), go.std().item()))

        if isinstance(grad_input, torch.Tensor):
            self.gradients_stats[f"{mod.__class__.__name__} input"].append((grad_input.mean().item(), grad_input.std().item()))
        elif isinstance(grad_input, (list, tuple)):
            for i, gi in enumerate(grad_input):
                if isinstance(gi, torch.Tensor):
                    self.gradients_stats[f"{mod.__class__.__name__}[{i}] input"].append((gi.mean().item(), gi.std().item()))

    def _analyze_statistics(self):
        """Analyzes accumulated statistics and logs if mean is close to zero."""
        for name, stats in self.activations_stats.items():
            avg_mean, avg_std = zip(*stats) if stats else (0, 0)
            mean_avg = sum(avg_mean) / len(stats) if stats else 0
            if abs(mean_avg) < 1e-6:  # Threshold for 'close to zero'
                print(f"Activation mean unexpectedly close to zero in {name}: Mean {mean_avg}")
        for name, stats in self.gradients_stats.items():
            avg_mean, avg_std = zip(*stats) if stats else (0, 0)
            mean_avg = sum(avg_mean) / len(stats) if stats else 0
            if abs(mean_avg) < 1e-6:
                print(f"Gradient mean unexpectedly close to zero in {name}: Mean {mean_avg}")

    def analyze_statistics(self):
        """Analyzes accumulated statistics and logs if mean is infinite."""
        print('analysing hooks')
        for name, stats in self.activations_stats.items():
            avg_mean, avg_std = zip(*stats) if stats else (0, 0)
            mean_avg = sum(avg_mean) / len(stats) if stats else 0
            if torch.isinf(torch.tensor(mean_avg)):  # Check for infinite mean
                print(f"Activation mean unexpectedly infinite in {name}: Mean {mean_avg}")
        for name, stats in self.gradients_stats.items():
            avg_mean, avg_std = zip(*stats) if stats else (0, 0)
            mean_avg = sum(avg_mean) / len(stats) if stats else 0
            if torch.isinf(torch.tensor(mean_avg)):  # Check for infinite mean
                print(f"Gradient mean unexpectedly infinite in {name}: Mean {mean_avg}")


        self.activations_stats.clear()
        self.gradients_stats.clear()

    def _configure_optimizers(self):
        args = self.args
        enc_params = list(self.audio_encoder.parameters()) + list(self.pose_encoder.parameters())
        mine_params = list(self.mine_nets.parameters())

        # Use a set to keep track of the parameter IDs to avoid duplicates
        new_params_ids = set(id(p) for p in enc_params + mine_params)
       
        # Gather all other parameters excluding those in new_params
        other_params = [p for p in self.parameters() if id(p) not in new_params_ids]
       
        # Combine all parameters
        model_params = enc_params + other_params + mine_params
       
        if args.weight_decay > 0:
            optimizer = FusedAdam(model_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                bias_correction=True, adam_w_mode=True, weight_decay=args.weight_decay)
        else:
            optimizer = FusedAdam(model_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                bias_correction=True, adam_w_mode=False, weight_decay=0)

        # Define a custom learning rate scheduler
        def lr_lambda(epoch):
            # High initial rate decreasing to lr_init
            if epoch < 10:
                return 1 - epoch * (1 - args.lr_init / high_lr) / 10
            # Oscillate around args.lr_init
            else:
                return args.lr_init + (args.lr_init / 10) * torch.sin(epoch / 10)

        high_lr = 0.01  # Example high initial learning rate
        lr_scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [lr_scheduler]

    def configure_optimizers(self):
        args = self.args
        enc_params = list(self.audio_encoder.parameters()) + list(self.pose_encoder.parameters())
        mine_params = list(self.mine_nets.parameters())

        # Use a set to keep track of the parameter IDs to avoid duplicates
        new_params_ids = set(id(p) for p in enc_params + mine_params)
       
        # Gather all other parameters excluding those in new_params
        other_params = [p for p in self.parameters() if id(p) not in new_params_ids]
       
        # Combine all parameters
        model_params = enc_params + other_params + mine_params

        # Define the base learning rate scheduler function
        def base_lr_lambda(epoch):
            # Assume a higher starting learning rate that decreases to args.lr_init
            high_lr = 0.01  # Example high initial learning rate
            if epoch < 2:
                return 1 - epoch * (1 - args.lr_init / high_lr) / 10
            else:
                return args.lr_init + (args.lr_init / 10) * torch.sin(epoch / 10)

        if args.weight_decay > 0:
            net_optimizer = FusedAdam(model_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                    bias_correction=True, adam_w_mode=True, amsgrad=False, weight_decay=args.weight_decay)
            #mine_optimizer = AdamW(mine_params, lr=args.lr_init)
        else:
            net_optimizer = FusedAdam(model_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                    bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
            #mine_optimizer = SGD(mine_params, lr=0.001, momentum=0.9)

        # Apply QuantileClip gradient clipping
        #mine_optimizer = QuantileClip.as_optimizer(optimizer=mine_optimizer, quantile=0.98, history_length=1000)
        clipper = QuantileClip.as_optimizer(optimizer=net_optimizer, quantile=0.9, history_length=1000)

        # Learning rate schedulers
        """net_lr_scheduler = {
        'scheduler': LambdaLR(net_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch),
        'interval': 'epoch',
        'reduce_on_plateau': False,
        'monitor': 'val_loss', # if you have validation loss to monitor
        'frequency': 1
        }
        mine_lr_scheduler = {
            'scheduler': LambdaLR(mine_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch),
            'interval': 'epoch',
            'reduce_on_plateau': False,
            'monitor': 'val_loss', # optional if using ReduceLROnPlateau
            'frequency': 1
        }"""

        # Return correct structure
        return net_optimizer, clipper

    #################### gpt suggestions for deepspeed configs, to be tested ...

   


   




    #################################################### old tested deepspeed configs below

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, x, model_idx = None, head = False):

        if model_idx != None and not head:
            model = self.rwkv_subnetworks[model_idx]
            layer_norm = self.layer_norms[model_idx]
            
        elif not head:
            
            model = self.output_model
            layer_norm = self.layer_norms[self.n_features + 1]
        
        else:
            model = self.controller_network
            layer_norm = self.head_layer_norm


        args = self.args
        #B, T, _ = x.size()
        #assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        #x = self.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = model.drop0(x)
        if args.tiny_att_dim > 0:
            for block in model.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in model:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)

        output = layer_norm(x).squeeze(0)#model.ln_out(x)

        if head:
            #print(output[-1:])
            output = self.controller_head(output[-1:])
            #output = F.softmax(output, dim=-1).squeeze()#.tolist()
            #output = output[-1:,:].squeeze()
        #print(weights)
        #print('weights.shape :::::::::::::::::' , weights.shape)
        
       
        #print('emb.shape = ', emb.shape )
        #joints, marginals = self.encoder.pqmf(audio_ahead_joints), self.encoder.pqmf(audio_ahead_marginals)
        #print('joints.shape, marginals.shape =', joints.shape, marginals.shape)
        #mine_values = self.mine(emb.reshape(64, -1, 1), joints, marginals)
        #loss = - self.mine_calculator.update(mi)

        #if args.head_qk > 0:
        #    q = self.head_q(x)[:, :T, :]
        #    k = self.head_k(x)[:, :T, :]
        #    c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
        #    c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

        #    if "32" in os.environ["RWKV_FLOAT_MODE"]:
        #        c = c @ F.one_hot(idx, num_classes=args.vocab_size)
        #    elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
        #        c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
        #    elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
        #        c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

        #    x = self.head(x) + c
        #else:
        #    x = self.head(x)

        return output
   
    # vectorize below function across the full sequence formed by concattenating all subsequences of a particular feature idx
   
    @torch.jit.script
    def _compute_mine_means(t_joints: torch.Tensor, t_margs: torch.Tensor, joint_mean: torch.Tensor, marg_mean: torch.Tensor, count: torch.Tensor, seq_len: int):
        #print('in compute mine energy loss ; ------emb.shape, audio_joint.shape, audio_marg.shape = ', emb.shape, audio_ahead_joint.shape, audio_ahead_marg.shape)
        #t_joints, t_margs = self.mine_nets[idx](emb, audio_ahead_joint, audio_ahead_marg)
       
        with torch.no_grad():
            for i in range(seq_len):
                #print('t_joints[i].shape, t_margs[i].shape, joint_mean.shape, marg_mean.shape, count.shape = ', t_joints[i].shape, t_margs[i].shape, joint_mean.shape, marg_mean.shape, count.shape)
                joint_mean += (t_joints[i] - joint_mean)/count
                marg_mean += (torch.exp(t_margs[i]) - marg_mean)/count
                count += 1.0
         #self.compute_loss() #self.accumulated_means[0] - torch.log(torch.exp(self.accumulated_means[1]))
        # update the gradients of the audio encoder only when backpropagating from the past inputs, to on procesing the future audio  values in the mine optimization step
        # that is - add the weights of the audio encoder to the model optimizer only, not to the mine optimizers,
        # but still use the audio encoder to embedd the audio futures when computing the mine values across the features-grid
        return joint_mean, marg_mean, count
   
    #@torch.jit.script
   
    def create_masks(self):
        masks = nn.ParameterList()
        dim2_segments = [(i * self.embedding_dim, (i + 1) * self.embedding_dim) for i in range(self.n_frequency_features)]
        for keep in self.time_scales:
            for start, end in dim2_segments:
                mask = torch.zeros((self.embedding_dim, self.n_frequency_features * self.embedding_dim))
                mask[keep:, start:end] = 1
                masks.append(nn.Parameter(mask, requires_grad=False))

        return masks
       
   
    def apply_masks(self, x):
        #print('apply masks input shapes ::::::::::', x.shape) 
        with torch.no_grad():
            masked_outputs = []
            #L = x.size(0)  # Determine L dynamically from the input tensor
            for mask in self.masks:
                #expanded_mask = mask.expand(L, 256)  # Expand the mask to match the input batch size
                masked_outputs.append(x * mask)  # Element-wise multiplication
        #print('apply masks output shapes ::::::::::', masked_outputs[0].shape) 
        return masked_outputs
   
    def weighted_sum_update(self, target_param, source_params, weights):
            target_param.data.zero_()
            for source_param, weight in zip(source_params, weights):
                target_param.data.add_(source_param.data, alpha=weight)

        # JIT script the function for optimization
        #scripted_weighted_sum_update = torch.jit.script(weighted_sum_update)

    def update_output_model(self, weights):
            with torch.no_grad():  # Prevent gradient computation
                # Iterate through each parameter in the target model
                for target_param, *source_params in zip(self.output_model.parameters(), *(model.parameters() for model in self.rwkv_subnetworks)):
                    # Use the JIT-optimized function for the update
                    self.weighted_sum_update(target_param, source_params, weights)

    @torch.jit.script_method
    def _apply_masks(self, x):
        masked_outputs = torch.jit.annotate(List[torch.Tensor], [])
        for mask in self.masks:
            masked_outputs.append(x * mask)
        return masked_outputs

    
       
    def run_output_model(self,  model_weights, x):
        #print('model_weights.shape _____________________ =', model_weights.shape)
        model_weights = model_weights.tolist()
        self.update_output_model(model_weights)
        x = x.view(1, -1, 128)
        #print('x shape in run output model = ', x.shape)
        output = self(x)
        return output
    
    def run_controller_network(self, x):
        x = x.view(1, -1, 256)
        
        #print('controller inputs shape ===========', x)
        """for block in self.controller_network:
            x = block(x)
        print(x)
        x = self.head_layer_norm(x)
        #print('self.head_layer_norm(x) shape ==============', x.shape)
        logits = self.controller_head(x)
        #print('logits shape  :::::::::::', logits.shape)
        weights = F.softmax(logits, dim=-1).squeeze()#.tolist()
        #print(weights)
        #print('weights.shape :::::::::::::::::' , weights.shape)
        return weights[-1:,:].squeeze()"""
        y = self(x, head=True)
        weights = F.softmax(torch.clamp(y, min=1e-5))
        print(weights)
        return weights.squeeze()
    
    def gram_matrix_loss(self, outputs):
        """Calculate the negative of the Frobenius norm of the Gram matrix of the output vectors."""
        
        outputs_matrix = torch.stack(outputs, dim=1)
        pairwise_distances_matrix = torch.cdist(outputs_matrix, outputs_matrix) 
        gram_matrix = torch.matmul(pairwise_distances_matrix, pairwise_distances_matrix.t())
        # Normalize the gram matrix to have values between 0 and 1
        #with torch.no_grad():
        #    gram_matrix /= torch.norm(gram_matrix, p='fro')
        frobenius_norm = torch.norm(gram_matrix)
        return frobenius_norm#.requires_grad_()


    def _gram_matrix_loss(self, outputs):
        """Calculate the negative of the Frobenius norm of the Gram matrix of the output vectors."""
        outputs_matrix = torch.stack(outputs, dim=1)
        gram_matrix = torch.matmul(outputs_matrix, outputs_matrix.t())
        # Normalize the gram matrix to have values between 0 and 1
        #with torch.no_grad():
        #    gram_matrix /= torch.norm(gram_matrix, p='fro')
        frobenius_norm = torch.norm(gram_matrix)
        return frobenius_norm#.requires_grad_()
   
    def compute_mine_means(self, joint_subsequences, marg_subsequences):
       
        #print('in compute mine energy loss ; ------emb.shape, audio_joint.shape, audio_marg.shape = ', emb.shape, audio_ahead_joint.shape, audio_ahead_marg.shape)
        #t_joints, t_margs = self.mine_nets[idx](emb, audio_ahead_joint, audio_ahead_marg)
       
        with torch.no_grad():
            # Convert lists to tensors
            mean_joint = torch.mean(torch.stack(joint_subsequences)) + self.log_eps
            mean_marg = torch.mean(torch.stack(marg_subsequences))

        mine_value = - mean_joint + torch.log(mean_marg + self.log_eps)
        return mine_value.requires_grad_()
    
    def criterion_model(self, mine_loss, sheaf_gram_loss, final_out_emb = None):
        if self.training_mode == 'init_representation_learning_phase_2':
            #safe_gram_loss = torch.clamp(sheaf_gram_loss, min=1e-6)
            loss = mine_loss * sheaf_gram_loss# torch.log(safe_gram_loss)
            #loss = mine_loss * torch.log(sheaf_gram_loss + 1e-9)
        return loss
   
    def calculate_loop_increments(self, batch_size):
       
        increments = {}
        max_seq = self.time_scales[-1]
       
        maxlen = max_seq * (batch_size // max_seq - 2)
       

        for seq_len in self.time_scales:
            diff = maxlen % seq_len
            seq_max_len = (maxlen - diff) // seq_len
            increments[seq_len] = seq_max_len
           
        random_incr = randint(0, batch_size - maxlen - 2*max_seq)

        return increments, random_incr #start index (random calculated within max range, same for each seq_len), nr of increments

    
   
    def training_step(self, batch, batch_idx):

        ti = time.time()
        # Unpack the batch
       
        try:

            if self.training_mode == 'init_representation_learning_phase_1':
                print('phase1 loop started')
                steps, n_videos = 0, 0
                net_optimizer, clipper = self.optimizers()
                #net_optimizer = optimizers[0]
                #mine_optimizer = optimizers[1]
                #clipper = optimizers[1]
                #lr_schedulers = self.lr_schedulers()
                #net_lr_scheduler = lr_schedulers[0]['scheduler']
                #mine_lr_scheduler = lr_schedulers[1]['scheduler']
                #net_optimizer, mine_optimizer, net_lr_scheduler, mine_lr_scheduler = self.optimizers()
                pose_batch, audio_batch, audio_marginals = batch
                pose_batch, audio_batch, audio_marginals = pose_batch.view(-1, 33, 3), audio_batch.view(-1, 1, 2048), audio_marginals.view(-1, 1, 2048)
                # Reshape tensors to match the required input shapes for the forward pass
                #print(pose_batch.shape, audio_batch.shape, audio_ahead_joints, audio_ahead_marginals)
                #print('pose_batch.shape, audio_batch.shape, audio_marginals.shape = ', pose_batch.shape, audio_batch.shape, audio_marginals.shape)
                pose_emb_seq, audio_emb_seq = self.pose_encoder(pose_batch), self.audio_encoder(audio_batch) # pick out unique encoder feature and sequence start/stop in batch for each idx here                  
               
               
                #with torch.no_grad(): # no grad since outputs should be used only in mine computation, but mine parameter update are separate from model param update
                batch_len = audio_emb_seq.shape[0]
                #encoded_audio_joints, encoded_audio_margs = self.mine_audio_encoder(audio_batch), self.mine_audio_encoder(audio_marginals[batch_len:, :, :]) # ensure in dataset constructor that always len audio_marginals greq batch.shape(0)
                loop_increments, rand_incr = self.calculate_loop_increments(batch_len)


                for idx in range(self.n_features): # len of self.feature_indices list
                    # Create a tuple of the four tensors and pass it to the model
                    #joint_mean, marg_mean, count = self.joint_mean, self.marg_mean, self.count
                    #joint_mean, marg_mean, count = 0, 0, 1

                   

                    seq_len, frequency_feature_idx = self.feature_indices[idx]
                   
                    #batch_len // seq_len - 2
                    joint_subsequences, marg_subsequences = [], []
                    for i in range(loop_increments[seq_len]):

                        feature_idx = self.embdim_step * frequency_feature_idx
                        feature_idx_step = self.embdim_step * (frequency_feature_idx + 1)

                        pose_emb = pose_emb_seq[(seq_len*i + rand_incr):seq_len*(i+1)+rand_incr, :]
                        audio_emb = audio_emb_seq[seq_len*i + rand_incr:seq_len*(i+1)+rand_incr, feature_idx:feature_idx_step]
                        emb = torch.concat((pose_emb, audio_emb), dim = 1).unsqueeze(0)# [frequency_feature] is now obtained by indexing into particular parts of the emb array, n_out in encoder resultts in concatenattion of n_out features across the array dimension
                        #pose_batch = pose_batch[seq_len*i:seq_len*(i+1), :, :]  # Reshape to (64, 33, 3)
                        #audio_batch = audio_batch[seq_len*i:seq_len*(i+1), :, :]   # Reshape to (64, 1, 2048)
                        audio_ahead_joints = audio_batch[seq_len*(i+1)+rand_incr:seq_len*(i+2)+rand_incr, :, :]
                        audio_ahead_marginals =  audio_marginals[seq_len*(i)+rand_incr:seq_len*(i+1)+rand_incr, :, :]
                        #print('audio_ahead_joints.shape , audio_ahead_marginals.shape ===========================', audio_ahead_joints.shape , audio_ahead_marginals.shape)
                        #audio_ahead_joints = encoded_audio_joints[seq_len*(i+1):seq_len*(i+2), feature_idx:feature_idx_step]   # audio target embeddings future step ahead with respect to the pose + audio embs
                        #audio_ahead_marginals = encoded_audio_margs[seq_len*(i):seq_len*(i+1), feature_idx:feature_idx_step]   # marginally distributed audio sequence, audio tensors from another video

                    #inputs = (pose_batch, audio_batch)
                         
                        #print('emb shape before forward:', emb.shape, 'pose_batch.shape, audio_batch.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape = ', pose_batch.shape, audio_batch.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape)
                        emb = self(emb, idx) #.squeeze(0) # model is selcted from time-frequency lattice here by the idx
                        #print('emb shape in phase 1', emb.shape)
                    #t_output = self.apply_mine_net(emb, audio_ahead_joints, audio_ahead_marginals, idx)
                        #print('_____________________emb, audio_ahead_joints, audio_ahead_marginals shapes = ', emb.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape)
                        #try:
                        #if emb.shape == torch.Size([1, 128]) and audio_ahead_joints.shape == torch.Size([seq_len, 1, 2048]) and audio_ahead_marginals.shape == torch.Size([seq_len, 1, 2048]):
                        if emb.shape == torch.Size([seq_len, 128]) and audio_ahead_joints.shape == torch.Size([seq_len, 1, 2048]) and audio_ahead_marginals.shape == torch.Size([seq_len, 1, 2048]):
                            t_joints, t_margs = self.mine_nets[idx](emb, audio_ahead_joints, audio_ahead_marginals)
                            joint_subsequences.append(t_joints)
                            marg_subsequences.append(t_margs)
                   
                    loss = self.compute_mine_means(joint_subsequences, marg_subsequences)    
                    print('train loss phase 1 = ', loss)
                    net_optimizer.zero_grad()
                    self.manual_backward(loss)
                    clipper.step()
                    net_optimizer.step()          
                    steps +=1
                print('steps phase 1 = ', steps)
                #self.training_mode = 'init_representation_learning_phase_2'

                            #joint_mean, marg_mean, count = self.compute_mine_means(t_joints, t_margs, joint_mean, marg_mean, count, seq_len)
                           
                            #
                            #t_joints, t_margs, joint_mean, marg_mean, count, seq_len
                            #print('//////// LOSS ///////////', loss)
                        #except RuntimeError:
                            #print('RuntimeError in self.compute_mine_energy_loss, shapes of emb, audio_ahead_joints, audio_ahead_marginals, idx : ', emb.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape, idx)
                       
                    # some extra pre-cautions (besides gradient-clipping in mine networks), if mine values goes unbounded
                    #self.compute_mine(joint_mean, marg_mean)
                   
            if self.training_mode == 'init_representation_learning_phase_2':
                        print('phase2 loop started')
                        steps, n_videos = 0, 0
                        net_optimizer, clipper = self.optimizers()

                        pose_batch, audio_batch, audio_marginals = batch
                        pose_batch, audio_batch, audio_marginals = pose_batch.view(-1, 33, 3), audio_batch.view(-1, 1, 2048), audio_marginals.view(-1, 1, 2048)
                        pose_emb_seq, audio_emb_seq = self.pose_encoder(pose_batch), self.audio_encoder(audio_batch) # pick out unique encoder feature and sequence start/stop in batch for each idx here                  
                        #print('audio_emb_seq.shape in phase 2 = ', audio_emb_seq.shape)
                        batch_len = audio_emb_seq.shape[0]
                        #print('current batch len phase 1 = ', batch_len)
                        #loop_increment = batch_len - 2*64 - 1
                        loop_increments, rand_incr = self.calculate_loop_increments(batch_len)
                        #rand_incr = 0  
                        loop_increment = batch_len - 3*64 - 1 - rand_incr
                        # comparing sheaf (global structure) loss term and individual mine (local loss terms) . Variables : nr steps and corresponding step sizes,
                        #, relative weights of terms
                        # method: variational methods, compare local structure to global structure  


                        for i in range(loop_increment - rand_incr):
                           
                            input_chunk = audio_emb_seq[( i + self.time_scales[-1] + rand_incr):(self.time_scales[-1] + (i + 1) + rand_incr),:]
                            masked_inputs = self.apply_masks(input_chunk)
                            weights_vectors, outputs = [], []
                            idx = 0
                            for x in masked_inputs:

                                seq_len, frequency_feature_idx = self.feature_indices[idx]
                                joint_subsequences, marg_subsequences = [], []

                                feature_idx = self.embdim_step * frequency_feature_idx
                                feature_idx_step = self.embdim_step * (frequency_feature_idx + 1)

                                pose_emb = pose_emb_seq[(i + rand_incr):seq_len + (i)+rand_incr, :]
                                audio_emb = audio_emb_seq[i + rand_incr:seq_len + (i)+rand_incr, feature_idx:feature_idx_step]
                                #print('audio_emb.shape, pose_emb.shape ---',audio_emb.shape, pose_emb.shape)
                                emb = torch.concat((pose_emb, audio_emb), dim = 1).unsqueeze(0)# [frequency_feature] is now obtained by indexing into particular parts of the emb array, n_out in encoder resultts in concatenattion of n_out features across the array dimension
                                audio_ahead_joints = audio_batch[i + 1 + rand_incr:seq_len + (i+1)+rand_incr, :, :]
                                audio_ahead_marginals =  audio_marginals[i + rand_incr:seq_len + (i)+rand_incr, :, :]
                                # compute the MINE values for current rwkv embedding w.r.t. the current (pose, audio_buffer)
                                
                                
                                #emb = self(emb, idx)

                                weights = self.run_controller_network(x)
                                emb = self.run_output_model(weights, emb)
                                weights_vectors.append(weights)
                                #print('model idx = ', idx, ' in phase 2, shapes of emb, audio_ahead_joints, audio_ahead_marginals =', emb.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape)
                                t_joints, t_margs = self.mine_nets[idx](emb, audio_ahead_joints, audio_ahead_marginals)
                                joint_subsequences.append(t_joints)
                                marg_subsequences.append(t_margs)

                                idx += 1
                                #print(x)
                                """weights = self.run_controller_network(x)
                                final_out_emb = self.run_output_model(weights, x)
                                weights_vectors.append(weights)
                                outputs.append(final_out_emb)
                                """
                                
                           
                            mine_loss = self.compute_mine_means(joint_subsequences, marg_subsequences)
                            sheaf_gram_loss = self.gram_matrix_loss(weights_vectors)
                            
                            loss = self.criterion_model(mine_loss, sheaf_gram_loss) 
                            print('train loss phase 2 = ', loss)
                            net_optimizer.zero_grad()
                            self.manual_backward(loss)
                            clipper.step()
                            net_optimizer.step()  
                            steps +=1

                            

                    # compute outputrun here obtaining sheaf loss / gram matrix loss for the controller network
                    # then compute MINE loss on output model resulting from the addition of mine_model's state dicts

                    
                   
                        # Perform optimization steps
                    #mine_optimizer.zero_grad()
                    #self.manual_backward(loss, retain_graph=True)
                    #mine_optimizer.step()
                   
                    #net_lr_scheduler.step()
                    #mine_lr_scheduler.step()

    # Optionally log values here
                    #current_lr_net = net_lr_scheduler.get_last_lr()[0]
                    #current_lr_mine = mine_lr_scheduler.get_last_lr()[0]
                    #self.log('current_learning_rate_net', current_lr_net)
                    #self.log('current_learning_rate_mine', current_lr_mine)
                        #self.mine_values[idx].append(loss)
                   
                        print('batch_len =', batch_len, ' nr of chunks for seq len', 'and feature ',', loss ===================' , loss)
                        self.log('train_loss', loss)
                                
                        if self.hooks:
                            self.analyze_statistics()
                        print('steps phase 2 = ', steps)
                        self.training_mode = 'init_representation_learning_phase_1'


                    #self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        except torch.cuda.CudaError as e:
                if "out of memory" in str(e):
                    memory_summary = torch.cuda.memory_summary(abbreviated=True)
                    print(f"WARNING: Ran out of memory at batch {batch_idx}. Memory summary:\n{memory_summary}")
                    torch.cuda.empty_cache()
                    raise pl.utilities.exceptions.CancelBatchException()  # This will skip the batch without stopping training
                else:
                    raise e
               
        ti = time.time()-ti
        print('time to process on video = ', ti)
        return loss


    def training_step_end(self, batch_parts):
        if pl.__version__[0]!='2':
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all


   
    def validation_step(self, batch, batch_idx):
        # Unpack the validation batch
        
        try:
            if self.training_mode == 'init_representation_learning_phase_1':
                print('phase1 loop started')
                steps, n_videos = 0, 0
                #net_optimizer, clipper = self.optimizers()
                #net_optimizer = optimizers[0]
                #mine_optimizer = optimizers[1]
                #clipper = optimizers[1]
                #lr_schedulers = self.lr_schedulers()
                #net_lr_scheduler = lr_schedulers[0]['scheduler']
                #mine_lr_scheduler = lr_schedulers[1]['scheduler']
                #net_optimizer, mine_optimizer, net_lr_scheduler, mine_lr_scheduler = self.optimizers()
                pose_batch, audio_batch, audio_marginals = batch
                pose_batch, audio_batch, audio_marginals = pose_batch.view(-1, 33, 3), audio_batch.view(-1, 1, 2048), audio_marginals.view(-1, 1, 2048)
                # Reshape tensors to match the required input shapes for the forward pass
                #print(pose_batch.shape, audio_batch.shape, audio_ahead_joints, audio_ahead_marginals)
                #print('pose_batch.shape, audio_batch.shape, audio_marginals.shape = ', pose_batch.shape, audio_batch.shape, audio_marginals.shape)
                pose_emb_seq, audio_emb_seq = self.pose_encoder(pose_batch), self.audio_encoder(audio_batch) # pick out unique encoder feature and sequence start/stop in batch for each idx here                  
               
               
                #with torch.no_grad(): # no grad since outputs should be used only in mine computation, but mine parameter update are separate from model param update
                batch_len = audio_emb_seq.shape[0]
                print('current batch len phase 1 = ', batch_len)
                #encoded_audio_joints, encoded_audio_margs = self.mine_audio_encoder(audio_batch), self.mine_audio_encoder(audio_marginals[batch_len:, :, :]) # ensure in dataset constructor that always len audio_marginals greq batch.shape(0)
                loop_increments, rand_incr = self.calculate_loop_increments(batch_len)


                for idx in range(self.n_features): # len of self.feature_indices list
                    # Create a tuple of the four tensors and pass it to the model
                    #joint_mean, marg_mean, count = self.joint_mean, self.marg_mean, self.count
                    #joint_mean, marg_mean, count = 0, 0, 1

                   

                    seq_len, frequency_feature_idx = self.feature_indices[idx]
                   
                    #batch_len // seq_len - 2
                    joint_subsequences, marg_subsequences = [], []
                    for i in range(loop_increments[seq_len]):

                        feature_idx = self.embdim_step * frequency_feature_idx
                        feature_idx_step = self.embdim_step * (frequency_feature_idx + 1)

                        pose_emb = pose_emb_seq[(seq_len*i + rand_incr):seq_len*(i+1)+rand_incr, :]
                        audio_emb = audio_emb_seq[seq_len*i + rand_incr:seq_len*(i+1)+rand_incr, feature_idx:feature_idx_step]
                        #print('audio_emb_seq.shape in phase 1 = ', audio_emb_seq.shape)
                        #print('audio_emb.shape, pose_emb.shape ---',audio_emb.shape, pose_emb.shape)
                        emb = torch.concat((pose_emb, audio_emb), dim = 1).unsqueeze(0)# [frequency_feature] is now obtained by indexing into particular parts of the emb array, n_out in encoder resultts in concatenattion of n_out features across the array dimension
                        #pose_batch = pose_batch[seq_len*i:seq_len*(i+1), :, :]  # Reshape to (64, 33, 3)
                        #audio_batch = audio_batch[seq_len*i:seq_len*(i+1), :, :]   # Reshape to (64, 1, 2048)
                        audio_ahead_joints = audio_batch[seq_len*(i+1)+rand_incr:seq_len*(i+2)+rand_incr, :, :]
                        audio_ahead_marginals =  audio_marginals[seq_len*(i)+rand_incr:seq_len*(i+1)+rand_incr, :, :]
                        #print('audio_ahead_joints.shape , audio_ahead_marginals.shape ===========================', audio_ahead_joints.shape , audio_ahead_marginals.shape)
                        #audio_ahead_joints = encoded_audio_joints[seq_len*(i+1):seq_len*(i+2), feature_idx:feature_idx_step]   # audio target embeddings future step ahead with respect to the pose + audio embs
                        #audio_ahead_marginals = encoded_audio_margs[seq_len*(i):seq_len*(i+1), feature_idx:feature_idx_step]   # marginally distributed audio sequence, audio tensors from another video

                    #inputs = (pose_batch, audio_batch)
                         
                        #print('emb shape before forward:', emb.shape, 'pose_batch.shape, audio_batch.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape = ', pose_batch.shape, audio_batch.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape)
                        emb = self(emb, idx) #.squeeze(0) # model is selcted from time-frequency lattice here by the idx
                        #print(emb)
                    #t_output = self.apply_mine_net(emb, audio_ahead_joints, audio_ahead_marginals, idx)
                        #print('_____________________emb, audio_ahead_joints, audio_ahead_marginals shapes = ', emb.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape)
                        #try:
                        #if emb.shape == torch.Size([1, 128]) and audio_ahead_joints.shape == torch.Size([seq_len, 1, 2048]) and audio_ahead_marginals.shape == torch.Size([seq_len, 1, 2048]):
                        if emb.shape == torch.Size([seq_len, 128]) and audio_ahead_joints.shape == torch.Size([seq_len, 1, 2048]) and audio_ahead_marginals.shape == torch.Size([seq_len, 1, 2048]):
                            t_joints, t_margs = self.mine_nets[idx](emb, audio_ahead_joints, audio_ahead_marginals)
                            #print('model idx = ', idx, ' in phase 1, shapes of emb, audio_ahead_joints, audio_ahead_marginals =', emb.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape)
                            joint_subsequences.append(t_joints)
                            marg_subsequences.append(t_margs)
                   
                    val_loss = self.compute_mine_means(joint_subsequences, marg_subsequences)    
                    print('phase 1 val_loss = ', val_loss)
                    #net_optimizer.zero_grad()
                    #self.manual_backward(val_loss)
                    #clipper.step()
                    #net_optimizer.step()          
                    steps +=1
                print('steps phase 1 = ', steps)
                #self.training_mode = 'init_representation_learning_phase_2'

                            #joint_mean, marg_mean, count = self.compute_mine_means(t_joints, t_margs, joint_mean, marg_mean, count, seq_len)
                           
                            #
                            #t_joints, t_margs, joint_mean, marg_mean, count, seq_len
                            #print('//////// LOSS ///////////', loss)
                        #except RuntimeError:
                            #print('RuntimeError in self.compute_mine_energy_loss, shapes of emb, audio_ahead_joints, audio_ahead_marginals, idx : ', emb.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape, idx)
                       
                    # some extra pre-cautions (besides gradient-clipping in mine networks), if mine values goes unbounded
                    #self.compute_mine(joint_mean, marg_mean)
                   
            if self.training_mode == 'init_representation_learning_phase_2':
                        print('phase2 loop started')
                        steps, n_videos = 0, 0
                        net_optimizer, clipper = self.optimizers()

                        pose_batch, audio_batch, audio_marginals = batch
                        pose_batch, audio_batch, audio_marginals = pose_batch.view(-1, 33, 3), audio_batch.view(-1, 1, 2048), audio_marginals.view(-1, 1, 2048)
                        pose_emb_seq, audio_emb_seq = self.pose_encoder(pose_batch), self.audio_encoder(audio_batch) # pick out unique encoder feature and sequence start/stop in batch for each idx here                  
                        #print('audio_emb_seq.shape in phase 2 = ', audio_emb_seq.shape)
                        batch_len = audio_emb_seq.shape[0]
                        print('current batch len phase 1 = ', batch_len)
                        #loop_increment = batch_len - 2*64 - 1
                        loop_increments, rand_incr = self.calculate_loop_increments(batch_len)
                        #rand_incr = 0  
                        loop_increment = batch_len - 3*64 - 1 - rand_incr
                        # comparing sheaf (global structure) loss term and individual mine (local loss terms) . Variables : nr steps and corresponding step sizes,
                        #, relative weights of terms
                        # method: variational methods, compare local structure to global structure  


                        for i in range(loop_increment - rand_incr):
                           
                            input_chunk = audio_emb_seq[( i + self.time_scales[-1] + rand_incr):(self.time_scales[-1] + (i + 1) + rand_incr),:]
                            masked_inputs = self.apply_masks(input_chunk)
                            weights_vectors, outputs = [], []
                            idx = 0
                            for x in masked_inputs:

                                seq_len, frequency_feature_idx = self.feature_indices[idx]
                                joint_subsequences, marg_subsequences = [], []

                                feature_idx = self.embdim_step * frequency_feature_idx
                                feature_idx_step = self.embdim_step * (frequency_feature_idx + 1)

                                pose_emb = pose_emb_seq[(i + rand_incr):seq_len + (i)+rand_incr, :]
                                audio_emb = audio_emb_seq[i + rand_incr:seq_len + (i)+rand_incr, feature_idx:feature_idx_step]
                                #print('audio_emb.shape, pose_emb.shape ---',audio_emb.shape, pose_emb.shape)
                                emb = torch.concat((pose_emb, audio_emb), dim = 1).unsqueeze(0)# [frequency_feature] is now obtained by indexing into particular parts of the emb array, n_out in encoder resultts in concatenattion of n_out features across the array dimension
                                audio_ahead_joints = audio_batch[i + 1 + rand_incr:seq_len + (i+1)+rand_incr, :, :]
                                audio_ahead_marginals =  audio_marginals[i + rand_incr:seq_len + (i)+rand_incr, :, :]
                                # compute the MINE values for current rwkv embedding w.r.t. the current (pose, audio_buffer)
                                
                                
                                #emb = self(emb, idx)

                                weights = self.run_controller_network(x)
                                emb = self.run_output_model(weights, emb)
                                weights_vectors.append(weights)
                                #print('model idx = ', idx, ' in phase 2, shapes of emb, audio_ahead_joints, audio_ahead_marginals =', emb.shape, audio_ahead_joints.shape, audio_ahead_marginals.shape)
                                t_joints, t_margs = self.mine_nets[idx](emb, audio_ahead_joints, audio_ahead_marginals)
                                joint_subsequences.append(t_joints)
                                marg_subsequences.append(t_margs)

                                idx += 1
                                #print(x)
                                """weights = self.run_controller_network(x)
                                final_out_emb = self.run_output_model(weights, x)
                                weights_vectors.append(weights)
                                outputs.append(final_out_emb)
                                """
                                
                           
                            mine_loss = self.compute_mine_means(joint_subsequences, marg_subsequences)
                            sheaf_gram_loss = self.gram_matrix_loss(weights_vectors)
                           
                            val_loss = self.criterion_model(mine_loss, sheaf_gram_loss) # to be implemented
                            print('phase 2 val_loss = ', val_loss, 'step nr ', steps)
                        self.training_mode = 'init_representation_learning_phase_1'           

        # Log the validation loss
                        # Ensure it's synced across devices if using DDP
        except torch.cuda.CudaError as e:
                if "out of memory" in str(e):
                    memory_summary = torch.cuda.memory_summary(abbreviated=True)
                    print(f"WARNING: Ran out of memory at batch {batch_idx}. Memory summary:\n{memory_summary}")
                    torch.cuda.empty_cache()
                    raise pl.utilities.exceptions.CancelBatchException()  # This will skip the batch without stopping training
                else:
                    raise e
           
        return {'val_loss': val_loss}
   
    """def on_after_backward(self):
        # This function is called after the backward pass
        #if self.trainer.global_rank == 0:  # Log only for the first process in DDP
        for name, param in self.named_parameters():
                #print(name, param.grad is not None)
                #print(param.grad)
                # Ensure the parameter has gradients
                if param.grad is not None:
                   self.logger.experiment.add_histogram(f'{name}_grad', param.grad, self.global_step)
                else:
                    print(f'{name}_grad', param.grad, self.global_step)"""

    def on_train_start(self):
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.__enter__()

    def on_train_end(self):
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace("profiling_results.json")

    def on_save_checkpoint(self, checkpoint):
        """
        Called when saving a checkpoint, used for saving different parts of the model.
        """
        # Save RWKV state dict
        rwkv_state_dict = {k: v for k, v in self.state_dict().items() if 'encoder' not in k and 'mine' not in k}
        checkpoint['rwkv_state_dict'] = rwkv_state_dict

        # Save Encoder state dict
        encoder_state_dict = {k: v for k, v in self.state_dict().items() if 'encoder' in k and 'mine' not in k}
        checkpoint['encoder_state_dict'] = encoder_state_dict

        # Save Mine state dict
        mine_state_dict = {k: v for k, v in self.state_dict().items() if 'mine' in k}
        checkpoint['mine_state_dict'] = mine_state_dict

        return checkpoint
   
    def joint_encoder(self, pose_t, audio_t):
        pose_emb = self.pose_encoder(pose_t)
        audio_emb = self.audio_encoder(audio_t)#.reshape(-1, self.n_frequency_features)
        emb = torch.concat((pose_emb, audio_emb), dim = 1)
        return emb
   
    def update_means(self, new_joint, new_marginal):
        # Incremental update of the running means and count
        device = self.accumulated_means.device  # Get the device where the buffer is located
        new_joint = new_joint.to(device)        # Move inputs to the correct device
        new_marginal = new_marginal.to(device)
       
        new_joint_mean = self.accumulated_means[:, 0] + (new_joint.mean() - self.accumulated_means[:, 0]) / self.count
        new_marginal_mean = self.accumulated_means[:, 1] + (new_marginal.mean() - self.accumulated_means[:, 1]) / self.count

        self.count += 1
        self.accumulated_means[:, 0] = new_joint_mean
        self.accumulated_means[:, 1] = new_marginal_mean
   
    def compute_loss(self):
        joint_means = self.accumulated_means[:, 0]
        marginal_means = self.accumulated_means[:, 1]
        mi_estimates = joint_means - torch.log(torch.exp(marginal_means))
        return - torch.mean(mi_estimates).requires_grad_()
   
   
    def _compute_mine_energy_loss(self, emb, audio_ahead_joint, audio_ahead_marg, idx):
        #print('in compute mine energy loss ; ------emb.shape, audio_joint.shape, audio_marg.shape = ', emb.shape, audio_ahead_joint.shape, audio_ahead_marg.shape)
        t_joints, t_margs = self.mine_nets[idx](emb, audio_ahead_joint, audio_ahead_marg)

        with torch.no_grad():
            self.update_means(t_joints, t_margs)

        mine_value = self.compute_loss() #self.accumulated_means[0] - torch.log(torch.exp(self.accumulated_means[1]))
        # update the gradients of the audio encoder only when backpropagating from the past inputs, to on procesing the future audio  values in the mine optimization step
        # that is - add the weights of the audio encoder to the model optimizer only, not to the mine optimizers,
        # but still use the audio encoder to embedd the audio futures when computing the mine values across the features-grid
        return - mine_value
   
    def _configure_optimizers(self):
        args = self.args
        enc_params = list(self.encoder.parameters())
        mine_params = list(self.mine.parameters())
   
        # Use a set to keep track of the parameter IDs in new_params to avoid duplicates
        new_params_ids = set(id(p) for p in enc_params+mine_params)
   
        # Gather all other parameters excluding those in new_params
        other_params = [p for p in self.parameters() if id(p) not in new_params_ids]
   
        # Combine all parameters
        model_params = enc_params + other_params
   
        if args.weight_decay > 0:
            optimizer = FusedAdam(model_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                  bias_correction=True, adam_w_mode=True, amsgrad=False, weight_decay=args.weight_decay)
            mine_optimizer = FusedAdam(mine_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                  bias_correction=True, adam_w_mode=True, amsgrad=False, weight_decay=args.weight_decay)
            mine_optimizer = QuantileClip.as_optimizer(optimizer=mine_optimizer, quantile=0.9, history_length=1000)
        else:
            optimizer = FusedAdam(model_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                  bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
            mine_optimizer = FusedAdam(mine_params, lr=args.lr_init, betas=args.betas, eps=args.adam_eps,
                                  bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
            mine_optimizer = QuantileClip.as_optimizer(optimizer=mine_optimizer, quantile=0.9, history_length=1000)
       
        return optimizer, mine_optimizer

    def p_on_save_checkpoint(self, checkpoint):
        """
        Called when saving a checkpoint, used for saving different parts of the model.
        """

        # Save RWKV state dict
        rwkv_state_dict = {k: v for k, v in self.state_dict().items() if 'encoder' not in k and 'mine' not in k}
        checkpoint['rwkv_state_dict'] = rwkv_state_dict

        # Save Encoder state dict
        encoder_state_dict = {k: v for k, v in self.state_dict().items() if 'encoder' in k and 'mine' not in k}
        checkpoint['encoder_state_dict'] = encoder_state_dict

        # Save Mine state dict
        mine_state_dict = {k: v for k, v in self.state_dict().items() if 'mine' in k}
        checkpoint['mine_state_dict'] = mine_state_dict

        return checkpoint
    """ to implement:

    DONE self.audio_encoder: adapt/separate encoder to only audio for modularity

    DONE self.apply_mine_net: function computing Varadhan-MINE value from previously computed T-values stored in the self.mine_values[idx] list, applying the T-networks for each index to the inputs
   
    DONE T_mine network : standard mine network, computing one value at a time for already concatenated input features tensors, to be added to the deque,
    and for computing a running means in the self.apply_mine_net(args*)

    DONE (just increased number of encoder output channels, then to be implemented enforce "even distribution of MI" across channels, and orthogonality) implement PCA at end of audio encoder to obtain 4 frequency features

    implement self.feature_indices array indexing into features by time_sequence_length value and freaquency_feature_nr

    accomodate validation_step to training_step and implement the necessary logging etc

    implement versions of forward, training_step, validation_step, for the consecutive training_modes 'audio_reconstruction', 'RLHF_finetuning'
    including the controller network and the sheaf network loss, and reward model for the RLHF
    """
    def __configure_optimizers(self):
        args = self.args
       
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if ("time_mix" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_decay" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
       
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)


    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():

            if 'encoder' in n or 'mine' in n or 'accumulated_means' in n or 'count' in n or 'head':
                continue
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    if 'r' in os.environ["RWKV_MY_TESTING"]:
                        zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                    else:
                        zero = [".att.key.", ".att.receptance.", ".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
