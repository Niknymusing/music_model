import torch
from typing import List, Dict
from model import RWKV
import time

n_models = 16

def collect_model_states(models):
    """
    Collects the state dictionaries from a list of trained model instances.
    
    Args:
    models (List[torch.nn.Module]): List of PyTorch model instances.
    
    Returns:
    List[Dict[str, torch.Tensor]]: List of state dictionaries from each model.
    """
    return [model.w for model in models]

@torch.jit.script
def combine_weights(key: str, w_multi: List[Dict[str, torch.Tensor]], w_coeffs: torch.Tensor) -> torch.Tensor:
        """
        JIT-compiled method to compute the linear combination of weights for the given key from multiple dictionaries.
        
        Args:
        key (str): The key for the weight tensors in the dictionaries.
        w_multi (List[Dict[str, torch.Tensor]]): List of weight dictionaries.
        w_coeffs (torch.Tensor): Coefficients for the linear combination.
        
        Returns:
        torch.Tensor: The combined weight tensor.
        """
        tensors = [w[key] for w in w_multi]  # Unpacking tensors from each dictionary
        stacked_tensors = torch.stack(tensors, dim=0)
        return torch.einsum('n,nijk->ijk', w_coeffs, stacked_tensors)

def combine_weights(key: str, w_multi: List[Dict[str, torch.Tensor]], w_coeffs: torch.Tensor) -> torch.Tensor:
        tensors = [w[key] for w in w_multi]  # List of tensors extracted based on the key
        stacked_tensors = torch.stack(tensors, dim=0)  # Stack along a new dimension
        return torch.einsum('n,nijk->ijk', w_coeffs, stacked_tensors)  # Use einsum for weighted sum


checkpoint_path = '/Users/nikny/Library/CloudStorage/GoogleDrive-nilskakoseos@gmail.com/My Drive/rwkv_mine_training/checkpoints3/val-epoch=2-val_loss=-0.00.ckpt'
weights = torch.load(checkpoint_path, map_location=torch.device('cpu') )
rwkv_weights = weights['rwkv_state_dict']
new_path = '/Users/nikny/Downloads/rwkv_mine.pth'
torch.save(rwkv_weights, new_path)
#rwkv1 = RWKV(model=new_path, strategy='cpu fp32')
rwkv_models = [RWKV(model='/Users/nikny/Downloads/rwkv_mine.pth', strategy='cpu fp32') for _ in range(n_models)] #
w_coeffs = torch.rand(16)

models_w = [model.w for model in rwkv_models]

#print(models_w[0].keys())

import torch
from typing import List, Dict

@torch.jit.script
def combine_weights(tensors: List[torch.Tensor], w_coeffs: torch.Tensor) -> torch.Tensor:
    """
    Combine a list of tensors linearly using a list of coefficients.

    Args:
    tensors (List[torch.Tensor]): List of tensors to combine. Each tensor from a different model's same key.
    w_coeffs (torch.Tensor): Coefficients for the linear combination.

    Returns:
    torch.Tensor: The resulting tensor after combining.
    """
    # Stack tensors along a new dimension (0) so that einsum can operate over it
    stacked_tensors = torch.stack(tensors, dim=0)
    # Using einsum to combine tensors along the first dimension using the coefficients
    combined_tensor = torch.einsum('n,n...->...', w_coeffs, stacked_tensors)
    return combined_tensor

def get_combined_weights(models: List[torch.nn.Module], w_coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    For each key in the model's weight dictionary, combine the corresponding tensors from all models.

    Args:
    models (List[torch.nn.Module]): List of model instances.
    w_coeffs (torch.Tensor): Coefficients used for combining weights.

    Returns:
    Dict[str, torch.Tensor]: Dictionary of combined weights.
    """
    # Assume all models have the same structure, so we can use the keys from the first model
    combined_weights = {}
    for key in models[0].w.keys():
        # Gather the same tensor across all models for the current key
        tensors = [model.w[key] for model in models]
        # Combine these tensors using the predefined function
        combined_weights[key] = combine_weights(tensors, w_coeffs)
    return combined_weights


#models = collect_model_states(rwkv_models)s
"""t = time.time()
models_w = get_combined_weights(rwkv_models, w_coeffs)
t = time.time()-t
print('time to combine weights  = ', t)
"""
