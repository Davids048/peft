# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The implementation is based on "Parameter-Efficient Orthogonal Finetuning
# via Butterfly Factorization" (https://arxiv.org/abs/2311.06243) in ICLR 2024.

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

import os

import triton
import triton.ops
from triton.ops.blocksparse import matmul
import gc

# For profiling
from torch.profiler import record_function
from line_profiler import  profile



os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"
curr_dir = os.path.dirname(__file__)

_FBD_CUDA = None


def get_fbd_cuda():
    global _FBD_CUDA

    if _FBD_CUDA is not None:
        return _FBD_CUDA

    curr_dir = os.path.dirname(__file__)
    # need ninja to build the extension
    try:
        fbd_cuda = load(
            name="fbd_cuda",
            sources=[f"{curr_dir}/fbd/fbd_cuda.cpp", f"{curr_dir}/fbd/fbd_cuda_kernel.cu"],
            verbose=True,
            # build_directory='/tmp/'  # for debugging
        )
        # extra_cuda_cflags = ['-std=c++14', '-ccbin=$$(which gcc-7)']) # cuda10.2 is not compatible with gcc9. Specify gcc 7
        import fbd_cuda
    except Exception as e:
        warnings.warn(f"Failed to load the CUDA extension: {e}, check if ninja is available.")
        warnings.warn("Setting boft_n_butterfly_factor to 1 to speed up the finetuning process.")
        fbd_cuda = None

    _FBD_CUDA = fbd_cuda
    return _FBD_CUDA

def unload_cuda_module():
    # print(f"before unload cuda: {torch.cuda.memory_allocated()}")
    global _FBD_CUDA
    if _FBD_CUDA == None: return
    _FBD_CUDA = None
    gc.collect()
    torch.cuda.empty_cache()
    # print(f"after  unload cuda: {torch.cuda.memory_allocated()}")



class FastBlockDiag(Function):
    """
    Implements a custom autograd Function for a fast block diagonal operation using CUDA.

    This function is optimized for 4D tensors where the last two dimensions are equal, representing block diagonal
    matrices for efficient computation on CUDA devices.
    """

    @staticmethod
    def forward(ctx, input):
        """
        The forward method for FastBlockDiag.

        Computes the block diagonal operation on the input tensor using a CUDA-optimized function. This method assumes
        that the input is a 4D tensor where the last two dimensions are equal, which represent the blocks to be
        diagonalized.

        Parameters:
        ctx: A context object that can be used to stash information for backward computation.
        input (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size,
                        `D` represents one additional dimension (In BOFT, the number of BOFT blocks), and `H` is the
                        size of the square blocks along the last two dimensions (In BOFT, the block size).

        Returns:
        Tensor: The resulting tensor after applying the block diagonal operation,
                will have the shape (N, DxH, DxH).
        """
        output = get_fbd_cuda().forward(input)[0]
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = get_fbd_cuda().backward(grad_output, input)[0]
        return grad_input


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for BOFT.
    """

    def __init__(self, p=0.0):
        """
        Initializes the multiplicative dropout layer.

        Parameters:
        p (float): The probability of dropping out a block. Defaults to 0.0.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Applies multiplicative dropout to the input tensor.

        Parameters:
        x (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size, `D` represents
                    one additional dimension (In BOFT, the number of BOFT blocks), and `H` is the size of the square
                    blocks along the last two dimensions (In BOFT, the block size).
        """
        if self.training:
            # Ensure the last two dimensions are the same
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("The last two dimensions of input should be the same!")

            N, D, H, _ = x.shape

            # Randomly select one from N
            n_random = torch.randint(0, N, (1,)).item()

            # Create a mask with 1s for matrices to be replaced with identity and 0s otherwise
            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace

            # Generate a flat tensor with desired number of 1s and 0s
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])

            # Shuffle and reshape the mask
            mask = mask[torch.randperm(D)].view(1, D, 1, 1)

            full_mask = torch.zeros(N, D, 1, 1, device=x.device)
            full_mask[n_random] = mask

            # Use the mask to combine original matrices and identity matrices
            eye_matrix = torch.eye(H, device=x.device).repeat(N, D, 1, 1)
            x = (1 - full_mask) * x + full_mask * eye_matrix
        return x


class BOFTLayer(BaseTunerLayer):
    """
    Implements the BOFT layer.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("boft_R", "boft_s")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("boft_block_size", "boft_block_num", "boft_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Initializes the BOFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be
        added soon.

        Parameters:
        base_layer: the pretrained model layer
        """
        self.base_layer = base_layer
        self.boft_block_size = {}
        self.boft_block_num = {}
        self.n_butterfly_factors = {}
        self.boft_dropout = nn.ModuleDict({})
        self.boft_R = nn.ParameterDict({})
        self.boft_s = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        # fields for batched ops
        self.boft_R_cpu = nn.ParameterDict({})
        self.boft_P_dict = {} # key: (n_factor, n_blocks, block_size), value: boft_P

        base_layer = self.get_base_layer()

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return

        warnings.warn("Scaling operation for BOFT not supported! Automatically set scale to 1.")

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.boft_R.keys():
                continue

            warnings.warn("Scaling operation for BOFT not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.boft_R.keys():
                continue

            warnings.warn("Unscaling operation for BOFT not supported! Keeping scale to 1.")

    def update_layer(
        self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
    ):
        """
        Update the linear layer with trainable BOFT weights. Override for other layer types.
        """
        # to be consistent with the paper notation
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor+1} to be a positive integer number."
            )
        # print(f"=====update layer: block_size: {boft_block_size}, block_num: {boft_block_num}, n_factor: {boft_n_butterfly_factor}")
        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        if boft_block_size == 0 and boft_block_num != 0:
            if self.in_features % boft_block_num != 0:
                raise ValueError(
                    f"in_features ({self.in_features}) must be divisible by boft_block_num ({boft_block_num})!"
                )

            if boft_n_butterfly_factor != 0:
                if boft_n_butterfly_factor > int(math.log2(boft_block_num)):
                    raise ValueError(
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor+1})!"
                    )

            boft_block_size = int(self.in_features // boft_block_num)

        elif boft_block_size != 0 and boft_block_num == 0:
            if self.in_features % boft_block_size != 0:
                raise ValueError(
                    f"in_features ({self.in_features}) must be divisible by boft_block_size ({boft_block_size})!"
                )

            if boft_n_butterfly_factor != 0:
                if self.in_features < (boft_block_size * (2**boft_n_butterfly_factor)):
                    raise ValueError(
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )
                if self.in_features % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of in_features ({self.in_features}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )

            boft_block_num = int(self.in_features // boft_block_size)

        else:
            raise ValueError(
                f"You can only specify either boft_block_size ({boft_block_size}) or boft_block_num ({boft_block_num}), but not both simultaneously or setting both"
                "to be 0, because boft_block_size x boft_block_num != in_features."
            )

        # In OFT you can specify the number of blocks to be 1
        if boft_n_butterfly_factor != 0:
            if boft_block_num % 2 != 0:
                raise ValueError(f"boft_block_num ({boft_block_num}) must be an even number!")

            if boft_block_size % 2 != 0:
                raise ValueError(f"boft_block_size ({boft_block_size}) must be an even number!")

        # If there is no butterfly factor, then permutation matrix P will be an identity matrix.
        # check if we need to update P
        curr_adapter = self.active_adapter if isinstance(self.active_adapter, str) else self.active_adapter[0]

        if curr_adapter in self.boft_block_num.keys():
            
            if boft_block_num != self.boft_block_num[curr_adapter] or \
                boft_block_size != self.boft_block_size[curr_adapter] or \
                boft_n_butterfly_factor != self.n_butterfly_factors[curr_adapter]:
                # P is different, we need to calculate.
                self.create_boft_P(boft_n_butterfly_factor, boft_block_num, boft_block_size)
            else:
                # print("boft_P does not need update")
                pass
        else:
            # First time creating boft_P
            P = self.create_boft_P(boft_n_butterfly_factor, boft_block_num, boft_block_size)
            self.register_buffer("boft_P", P)



        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(int(self.out_features), 1))

        self.reset_boft_parameters(adapter_name, init_weights)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num
        self.n_butterfly_factors[adapter_name] = boft_n_butterfly_factor
        self.set_adapter(self.active_adapters)

        # move to cpu to prevent out of memory error when there are a lot of adapters loaded
        # self.move_boft_R_to_cpu(adapter_name)
        # self.move_boft_s_to_cpu(adapter_name)
        # # self.move_boft_p_to_cpu()


    def create_boft_P(self, boft_n_butterfly_factor, boft_block_num, boft_block_size):
        # P is different, we need to calculate.
        P = torch.empty((boft_n_butterfly_factor + 1, self.in_features, self.in_features))
        for i in range(boft_n_butterfly_factor + 1):
            perm = self.block_butterfly_perm(
                self.in_features, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat
        self.boft_P_dict[(boft_n_butterfly_factor, boft_block_num, boft_block_size)] = P
        return P

    def move_boft_R_to_cpu(self, adapter):
        if self.boft_R[adapter].is_cuda:
            boft_R_gpu = self.boft_R[adapter]
            if adapter in self.boft_R_cpu.keys():
                self.boft_R[adapter] = self.boft_R_cpu[adapter]
            else:
                boft_R_cpu = boft_R_gpu.to('cpu', non_blocking=True)
                self.boft_R[adapter] = boft_R_cpu 
                self.boft_R_cpu[adapter] = boft_R_cpu
            del boft_R_gpu
            torch.cuda.empty_cache()
            # print("boft_R moved to cpu:", self.boft_R[adapter].device)
        else:
            # print("boft_R is not on GPU, not moving to CPU")
            pass
    
    def move_boft_R_to_gpu(self, adapter):
        """
        Move the adapter's boft_R weight to GPU (if it is not on cuda)
        """
        if not self.boft_R[adapter].is_cuda:
            boft_R_cpu = self.boft_R[adapter]
            if self.base_layer.weight.is_cuda:
                boft_R_gpu = boft_R_cpu.to(dtype=self.base_layer.weight.dtype,
                                           device=self.base_layer.weight.device)
                torch.cuda.synchronize(self.base_layer.weight.device)
                self.boft_R[adapter] = boft_R_gpu
                del boft_R_cpu
            else:
                raise RuntimeError(f"base weight is on {self.base_layer.weight.device}- CUDA device error")
            
    def move_boft_s_to_cpu(self, adapter):
        if self.boft_s[adapter].is_cuda:
            boft_s_gpu = self.boft_s[adapter]
            self.boft_s[adapter] = boft_s_gpu.to('cpu', non_blocking=True)
            # torch.cuda.synchronize()
            del boft_s_gpu
            torch.cuda.empty_cache()
            # print(self.boft_s[adapter].device)
        else:
            # print("boft_s is not on GPU, not moving to CPU")
            pass

    def move_boft_s_to_gpu(self, adapter):
        """
        Move the adapter's boft_s weight to GPU (if it is not on cuda)
        """
        if not self.boft_s[adapter].is_cuda:
            boft_s_cpu = self.boft_s[adapter]
            if self.base_layer.weight.is_cuda:
                boft_s_gpu = boft_s_cpu.to(self.base_layer.weight.device)
                torch.cuda.synchronize(self.base_layer.weight.device)
                self.boft_s[adapter] = boft_s_gpu
                del boft_s_cpu
            else:
                raise RuntimeError(f"base weight is on {self.base_layer.weight.device}- CUDA device error")

    def move_boft_p_to_cpu(self):
        """
        TODO: REMOVE THIS FUNCTION if needed 
        Move stuff we don't need to cpu, so that it doesn't occupy gpu
        when updating layers, all the boft_P... etc matrices are moved to GPU,
        this is very costly, as that is a big matrix (e.g. in llama2, each 
        boft_P is 4096 * 4096). 

        Move all boft_P in boft_P_dict (not just self. boft_P) to CPU
        """
        if self.boft_P_dict:
            for key, boft_P in self.boft_P_dict.items():
                if boft_P.is_cuda:
                    boft_P_gpu = boft_P 
                    # print(f"boft_P: {boft_P_gpu.device}")
                    boft_P_cpu = boft_P_gpu.to('cpu', non_blocking=True)
                    # torch.cuda.synchronize()
                    self.boft_P_dict[key] = boft_P_cpu
                    del boft_P_gpu
                    torch.cuda.empty_cache()
                    # print("moved boft_P to cpu")

        # print(f"boft after  clear p mem: {torch.cuda.memory_allocated(self.base_layer.weight.device)}")
    
    def move_boft_p_to_gpu(self):
        if not self.boft_P.is_cuda:
            boft_p_cpu = self.boft_P
            if self.base_layer.weight.is_cuda:
                # print("base layer device: ", self.base_layer.weight.device)
                self.boft_P = self.boft_P.to(dtype=self.base_layer.weight.dtype,
                                             device=self.base_layer.weight.device)
                torch.cuda.synchronize(self.base_layer.weight.device)
                # print(f"moved p to gpu: {self.boft_P.device}")
            else:
                raise RuntimeError(f"base weight is on {self.base_layer.weight.device}- CUDA device error")
        # else:
        #     print("boft_P is on GPU already")
    
    def permute_butterfly_factors(self, diag_factors, adapter_name):
        """
        Given the adapter name, and the calculated factors in dxd full format,
        Calculate the permutation of the factors so they emit the butterfly 
        structures. 
        """
        key = (self.n_butterfly_factors[adapter_name], 
               self.boft_block_num[adapter_name],
               self.boft_block_size[adapter_name])
        if key not in self.boft_P_dict.keys():
            raise Exception(f"Boft_P is not found for adapter: {adapter_name}"
                            f"with key: {key}")
        else:
            boft_P = self.boft_P_dict[key]
            if not boft_P.is_cuda:
                # move boft_P to GPU:
                boft_P = boft_P.to(device=self.base_layer.weight.device,
                                   dtype=self.base_layer.weight.dtype)
            # calculate
            diag_factors = torch.bmm(diag_factors, boft_P.permute(0, 2, 1))
            diag_factors = torch.bmm(boft_P, diag_factors)

            # move boft_P to CPU 
            self.boft_P_dict[key] = boft_P.to(device='cpu', non_blocking=True)
            return diag_factors


    def reset_boft_parameters(self, adapter_name, init_weights):
        """
        Reset the BOFT parameters.
        """
        if init_weights is False:
            nn.init.normal_(self.boft_R[adapter_name], mean=0.0, std=0.1)
            nn.init.normal_(self.boft_s[adapter_name], mean=1.0, std=0.1)
            return

        if adapter_name in self.boft_R.keys():
            if init_weights is True:
                # initialize R to zero
                nn.init.zeros_(self.boft_R[adapter_name])
                nn.init.ones_(self.boft_s[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_weights=}")

    def perm2mat(self, indices):
        """
        Convert permutation indices to permutation matrix.

        Args:
        indices: A list of indices representing the permutation.
        """
        # Number of indices determines the size of the square matrix
        n = len(indices)

        # Initialize a matrix of zeros
        perm_mat = torch.zeros((n, n))

        # Set the 1s according to the indices
        for i, idx in enumerate(indices):
            perm_mat[i, idx] = 1

        return perm_mat

    def block_butterfly_perm(self, n, b, r=3, n_butterfly_factor=1):
        """
        Define the permutation matrix for the block butterfly permutation.

        Args:
        n: size of the permutation matrix
        b: desired number of blocks after multiplying with the permutation matrix
        r: base block size of the block diagonal matrix, e.g. 2x2, 3x3, 5x5 etc.
        """

        if n_butterfly_factor == 0:
            return torch.arange(n)

        if b * r * 2 > n:
            raise ValueError("Invalid number of blocks!")

        block_size = int(n // b)
        indices = torch.arange(n)

        def sort_block(b, r):
            step = b / r
            initial_order = torch.arange(b)
            sorted_order = torch.empty(b, dtype=torch.long)

            evens = torch.arange(0, step, 2)
            odds = torch.arange(1, step, 2)
            sorted_seq = torch.cat((evens, odds), dim=0)
            for i, pos in enumerate(sorted_seq):
                sorted_order[int(i * r) : int(i * r + r)] = initial_order[int(pos * r) : int(pos * r + r)]
            return sorted_order

        sorted_order = sort_block(block_size, r)

        for i in range(0, n, block_size):
            block_end = i + block_size
            tmp_indices = indices[i:block_end]
            indices[i:block_end] = tmp_indices[sorted_order]
        return indices

    def cayley_batch(self, data):
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew_mat = 0.5 * (data - data.transpose(1, 2))
        id_mat = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.linalg.solve(
                id_mat + skew_mat, id_mat - skew_mat, left=False
                ).to(dtype=data.dtype)

        return Q







class Linear(nn.Module, BOFTLayer):
    """
    BOFT implemented in a dense layer.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        boft_n_butterfly_factor: int = 0,
        boft_dropout: float = 0.1,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.batch_op = {} 
        self.batch_layout = {}
        self.batched_adapters = [] # stores the name of the currently batched adapters
        self.sub_block_size = 16 # all blocks will be re-sparsified to blocks of size 16 for unified layout

        # cuda graph vars
        self.forward_mode = "regular"
        self.graph = None
        self.static_weight = None

        # Attempt to load the CUDA extension during model initialization
        if not get_fbd_cuda():
            self.fbd_cuda_available = False
            # If the CUDA extension is not available, set the butterfly factor to 1 to speed up the finetuning process
            boft_n_butterfly_factor = 1
        else:
            self.fbd_cuda_available = True

        self.update_layer(
            adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.boft_R.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.base_layer.weight.data = orig_weight
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)
                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    orig_weight = orig_weight * boft_s

                    self.base_layer.weight.data = orig_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.boft_R.keys():
                butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                orig_weight = self.get_base_layer().weight.data.clone()
                orig_weight = torch.transpose(orig_weight, 0, 1)
                orig_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
                orig_weight = torch.transpose(orig_weight, 0, 1)

                self.get_base_layer().weight.data = orig_weight * (1 / boft_s)

    def get_delta_weight(self, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter]

        N, D, H, _ = boft_R.shape
        boft_R = boft_R.view(N * D, H, H)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
        if self.fbd_cuda_available:
            block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
        else:
            orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
            block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
            block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)

        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # print("forward: x shape: ", x.shape)
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if "batched_adapter" not in self.active_adapters:
                # avoid creating this to save mem
                boft_rotation = torch.eye(self.in_features, device=x.device, dtype=x.dtype)
                boft_scale = torch.ones((int(self.out_features), 1), device=x.device)

            for active_adapter in self.active_adapters:
                # print("forward start====", active_adapter)
                if active_adapter not in self.boft_R.keys():
                    continue
                if active_adapter == "batched_adapter":
                    # NOTE: Assuming we return immediately if the adapter is a 
                    # batched_adapter. Because there is no point of cumulating
                    # results of several adapters if 1 of them is a batched_adapter
                    # (even the shape won't align)
                    with record_function("get_batched_activation"):
                        return self.get_batched_activation(x, *args, **kwargs)
                else:
                    print(" not batched activation")
                    boft_R = self.boft_R[active_adapter]
                    boft_s = self.boft_s[active_adapter]
                    dropout = self.boft_dropout[active_adapter]

                    N, D, H, _ = boft_R.shape
                    # print("boft linear layer: ", boft_R.shape)
                    boft_R = boft_R.view(N * D, H, H)
                    orth_rotate_butterfly = self.cayley_batch(boft_R)
                    # print("orth_rotate_but (after caylay batch)", orth_rotate_butterfly.shape)
                    orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
                    orth_rotate_butterfly = dropout(orth_rotate_butterfly)
                    # print("orth_rotate_butterfly:", orth_rotate_butterfly.shape)
                    # torch.save(orth_rotate_butterfly, "/scratch/js202/boft/boft_dreambooth/orth_rotate_butterfly.pt")

                    if self.fbd_cuda_available:
                        block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
                    else:
                        block_diagonal_butterfly_lst = []
                        for i in range(orth_rotate_butterfly.shape[0]):
                            block_diagonal_butterfly = torch.block_diag(
                                *torch.unbind(orth_rotate_butterfly[i])
                            )
                            block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)
                            block_diagonal_butterfly_lst.append(block_diagonal_butterfly)
                        block_diagonal_butterfly = torch.cat(block_diagonal_butterfly_lst, dim=0)
                        # orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
                        # block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
                        # block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)
                    # print("block_diagonal_butterfly:", block_diagonal_butterfly.shape)
                    # torch.save(block_diagonal_butterfly, "/scratch/js202/boft/boft_dreambooth/block_diagonal_butterfly.pt")

                    butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
                    butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
                    # print("butterfly_oft_mat_batch:", butterfly_oft_mat_batch.shape)
                    # torch.save(butterfly_oft_mat_batch, "/scratch/js202/boft/boft_dreambooth/butterfly_oft_mat_batch.pt")
                    butterfly_oft_mat = butterfly_oft_mat_batch[0]
                    # print("b_oft_mat", butterfly_oft_mat.shape)

                    for i in range(1, butterfly_oft_mat_batch.shape[0]):
                        butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

                    boft_rotation = butterfly_oft_mat @ boft_rotation
                    boft_scale = boft_s * boft_scale

            x = x.to(self.get_base_layer().weight.data.dtype)

            orig_weight = self.get_base_layer().weight.data
            orig_weight = torch.transpose(orig_weight, 0, 1)
            rotated_weight = torch.mm(boft_rotation, orig_weight)
            rotated_weight = torch.transpose(rotated_weight, 0, 1)

            scaled_rotated_weight = rotated_weight * boft_scale

            result = F.linear(input=x, weight=scaled_rotated_weight, bias=self.base_layer.bias)

        result = result.to(previous_dtype)
        # print("====forward end====")
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "boft." + rep

    @profile 
    def batch_adapters(self, adapter_lst):
        """
        batch the boft_R part of each adapter layer of the input adapters
        """
        print("BOFT BATCHING>......")
        with torch.cuda.device(self.base_layer.weight.device):
            for layer_name in self.adapter_layer_names:
                # print("batching adapters: ", layer_name)
                # NOTE: we only batch boft_R for now. 
                if layer_name != "boft_R":
                    # print("batching: not boft_R")
                    continue
                else:
                    # processing boft_R 
                    sparse_boft_R_lst = []
                    batched_adapters_layout_lst = []
                    batched_boft_s = []
                    block_size = 0

                    all_configs = set()
                    for i in range(len(adapter_lst)):
                        adapter = adapter_lst[i]
                        all_configs.add((self.n_butterfly_factors[adapter], self.boft_block_num[adapter], self.boft_block_size[adapter]))
                        try:
                            # Use previously sparsified weights if it exist
                            prev_idx = adapter_lst[:i].index(adapter)
                            sparse_boft_R = sparse_boft_R_lst[prev_idx]
                            boft_s = batched_boft_s[prev_idx]
                            layout = batched_adapters_layout_lst[prev_idx]
                            print("using previous weights")
                        except Exception as e:
                            print("need to calculate sparse boft-R", e)
                            sparse_boft_R, boft_s, layout = self.get_sparse_weights_and_layout(adapter)
                        sparse_boft_R_lst.append(sparse_boft_R)
                        batched_boft_s.append(boft_s)
                        batched_adapters_layout_lst.append(layout)

                        self.move_boft_R_to_cpu(adapter)
                        self.move_boft_s_to_cpu(adapter)
                    unload_cuda_module()

                    # stack adapters together
                    sparse_batched_boft_R = torch.cat(sparse_boft_R_lst, dim=1).to(device=self.base_layer.weight.device) # n_factors, n_adapters, m, n
                    self.boft_R["batched_adapter"] = sparse_batched_boft_R.to(device=self.base_layer.weight.device)

                
                    # stack boft_s
                    self.boft_s["batched_adapter"] = torch.stack(batched_boft_s, dim=0).permute(0,2,1).to(device=self.base_layer.weight.device)

                    # stack layouts
                    unified_layout = torch.cat(batched_adapters_layout_lst, dim=1)
                    n_factors = unified_layout.shape[0]
                    # compile ops
                    ops_lst = []
                    for i in range(n_factors):
                        # only use the first layout (triton only needs 1 unified layout)
                        batch_op = triton.ops.blocksparse.matmul(unified_layout[i], self.sub_block_size, "dds", device=self.base_layer.weight.device)
                        ops_lst.append(batch_op)
                    self.batch_op["batched_adapter"] = ops_lst


                    self.set_adapter("batched_adapter")
                self.move_boft_p_to_cpu()
                return (self.boft_R["batched_adapter"], self.boft_s["batched_adapter"], self.batch_op["batched_adapter"])

    def get_full_boft_R(self, adapter):
        """
        Given the adapter, create the adapter's boft_R in the full format
        """
        if adapter not in self.boft_R.keys():
            raise KeyError(f"adapter {adapter} not found in available adapters")
        else: 
            self.move_boft_R_to_gpu(adapter)
            self.move_boft_p_to_gpu()
            self.batched_adapters.append(adapter)
            # perform preprocessing for each adapter
            boft_R = self.boft_R[adapter]

            N, D, H, _ = boft_R.shape
            # turn boft_R into butterfly structure
            boft_R = boft_R.view(N * D, H, H)
            orth_rotate_butterfly = self.cayley_batch(boft_R)
            orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)


            # move each single adapter out of cuda
            self.move_boft_R_to_cpu(adapter)
            
            torch.cuda.synchronize(self.base_layer.weight.device)
            
            block_diagonal_butterfly_lst = []
            for i in range(orth_rotate_butterfly.shape[0]):
                block_diagonal_butterfly = torch.block_diag(
                    *torch.unbind(orth_rotate_butterfly[i])
                )
                block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)
                block_diagonal_butterfly_lst.append(block_diagonal_butterfly)
            block_diagonal_butterfly = torch.cat(block_diagonal_butterfly_lst, dim=0)


            torch.cuda.synchronize(self.base_layer.weight.device)
            butterfly_oft_mat_batch = self.permute_butterfly_factors(block_diagonal_butterfly, adapter)
            return butterfly_oft_mat_batch
    
    def get_boft_R_layout(self, adapter):
        if adapter not in self.boft_R.keys():
            raise KeyError(f"adapter {adapter} not found in available adapters")
        else: 
            N, D, H, _ = self.boft_R[adapter].shape
            # get the layout for this adapter's factors (need all )
            n_factors = N
            size_ratio = int(H//self.sub_block_size)
            layout = self.create_block_diag_layout(size_ratio, D, n_factors)
            return layout.unsqueeze(1)
    
    def get_sparse_weights_and_layout(self, adapter):
        """
        Return Sparse boft_R, boft_s, and layout (blocksize = 16)
        """
        full_boft_R = self.get_full_boft_R(adapter=adapter) # n_factors, full_side, full_side
        layout = self.get_boft_R_layout(adapter=adapter)
        self.move_boft_s_to_gpu(adapter)
        boft_s = self.boft_s[adapter]
        n_factors, n_blocks, block_size ,_ = self.boft_R[adapter].shape

        # sparsify this weight
        each_step_factors_shape = [1, 1, full_boft_R.shape[1], full_boft_R.shape[2]]
        factors_by_step_lst = []
        for i in range(n_factors):
            cur_step_factors = self.sparsify_tensor(full_boft_R[i].view(each_step_factors_shape), # 1, 1, full_side, full_side
                                    mask=layout[i],
                                    block=self.sub_block_size)
            factors_by_step_lst.append(cur_step_factors)
        sparse_boft_R = torch.cat(factors_by_step_lst, dim=0) # n_factors, 1 adapter, m, n
        return (sparse_boft_R, boft_s, layout)


    
    def create_dummy_batched_adapters(self, adapter_lst):
        """
        Create a batched adapter w/ len(adapter_lst) number of adapters. Each is 
        a clone of the 1st adapter that is initially loaded to the model
        """
        print("BOFT creating dummy batched adapter")
        print(adapter_lst)
        with torch.cuda.device(self.base_layer.weight.device):
            assert self.boft_R # There must be at least 1 adapter in the model

            for adapter in self.active_adapters:
                full_boft_R = self.get_full_boft_R(adapter=adapter) # n_factors, full_side, full_side
                layout = self.get_boft_R_layout(adapter=adapter)
                self.move_boft_s_to_gpu(adapter)
                boft_s = self.boft_s[adapter]
                n_factors, n_blocks, block_size ,_ = self.boft_R[adapter].shape
                sub_block_size = int(block_size//2)

                # sparsify this weight
                each_step_factors_shape = [1, 1, full_boft_R.shape[1], full_boft_R.shape[2]]
                factors_by_step_lst = []
                for i in range(n_factors):
                    cur_step_factors = self.sparsify_tensor(full_boft_R[i].view(each_step_factors_shape), # 1, 1, full_side, full_side
                                            mask=layout[i],
                                            block=int(block_size//2))
                    factors_by_step_lst.append(cur_step_factors)
                sparse_boft_R = torch.cat(factors_by_step_lst, dim=0) # n_factors, 1 adapter, m, n
                break
            unload_cuda_module()

            # stack adapters together
            sparse_boft_R_lst = [sparse_boft_R for _ in range(len(adapter_lst))]
            sparse_batched_boft_R = torch.cat(sparse_boft_R_lst, dim=1).to(device=self.base_layer.weight.device) # n_factors, n_adapters, m, n
            print(f"stacked_full adapters: {sparse_batched_boft_R.shape}") # factor * batch * m * n 
            self.boft_R["batched_adapter"] = sparse_batched_boft_R.to(device=self.base_layer.weight.device)

          
            # stack boft_s
            batched_boft_s = [boft_s for _ in range(len(adapter_lst))]
            self.boft_s["batched_adapter"] = torch.stack(batched_boft_s, dim=0).permute(0,2,1).to(device=self.base_layer.weight.device)

            # compile ops
            ops_lst = []
            for i in range(n_factors):
                # only use the first layout (triton only needs 1 unified layout)
                batch_op = triton.ops.blocksparse.matmul(layout[i], sub_block_size, "dds", device=self.base_layer.weight.device)
                ops_lst.append(batch_op)
            self.batch_op["batched_adapter"] = ops_lst

            self.set_adapter("batched_adapter")
        self.move_boft_p_to_cpu()
        return (self.boft_R["batched_adapter"], self.boft_s["batched_adapter"], self.batch_op["batched_adapter"])
            





    def unbatch_adapters(self, adapter_lst):
        """
        Put the adapters in self.batched_adapters back to cuda
        """
        # batching moved fdb_cuda out of gpu. we load it again here for later use
        global _FBD_CUDA
        if _FBD_CUDA == None:
            get_fbd_cuda()

        if "batched_adapter" not in self.boft_R.keys():
            return
        print("--unbatching adapters--")
        device = self.boft_R["batched_adapter"].device
        for layer_name in self.adapter_layer_names:
            if layer_name == "boft_R":
                for adapter in adapter_lst:

                    if adapter not in self.batched_adapters:
                        raise KeyError(f"adapter {adapter} not found in batched adapters")    
                    self.move_boft_s_to_gpu(adapter)
                    self.move_boft_R_to_gpu(adapter)                    
        self.move_boft_p_to_gpu()
                    
    def sparsify_tensor(self, x, mask, block):
        """
        Create a sparse representation of the original matrix x.
        @params:
            x: the original tensor
            mask: the layout we use to sparsify the tenosr
            block: block size that each mask bit represents
        """
        # print("sparsify: x shape:", x.shape)
        # print("sparsify: mask: ", mask.shape)
        # print(f'mask sum: {mask.sum()}')
        ret = torch.empty((x.size(0), mask.sum(), block, block), dtype=x.dtype, device=x.device)
        for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
            ret[:, idx, :, :] = x[:, h, i * block:(i + 1) * block, j * block:(j + 1) * block]
        return ret

    
    def create_block_diag_layout(self, block_size, num_blocks, n_factors):
        """
        Create the layout for the blocks
        Args:
        block_size: the size of each block in the 1st factor
        num_blocks: the number of blocks in the first factor, which is also the 
            amount of blocks stored in the boft_R attr. 
        n_factors: the number of factors 

        Returns:
        The layout of each factor, stacked together in 1 matrix. 
        """
        # print(f"create_block_diag_layout:block size: {block_size}, num_blocks {num_blocks}, nfactors {n_factors}")
        # Create a block of ones
        block = torch.ones((block_size, block_size))
        
        # Initialize an empty matrix for the final block diagonal matrix
        dim = block_size * num_blocks # side length of the factor layout
        matrix = torch.zeros((dim, dim))
        
        # Place each block on the diagonal
        for i in range(num_blocks):
            start_idx = i * block_size
            matrix[start_idx:start_idx+block_size, start_idx:start_idx+block_size] = block

        matrix_stack = torch.stack([matrix for _ in range(n_factors)])

        perm_lst = []
        for i in range(n_factors):
            # get permutation
            perm = self.block_butterfly_perm(dim, int(num_blocks//2**i), int(block_size/2), n_factors)
            # perm = self.block_butterfly_perm(dim, int(num_blocks//2**i), int(block_size/2), 0)
            perm_mat = self.perm2mat(perm)
            perm_lst.append(perm_mat)
        perm_stack = torch.stack(perm_lst)

        layout = torch.bmm(matrix_stack, perm_stack.permute(0,2,1))
        layout = torch.bmm(perm_stack, layout)

        layout = layout.to(torch.int)
        return layout
    
    @torch.no_grad
    def get_batched_activation(self, x: torch.Tensor, *args, **kwargs):
        """
        Use the batched adapter and compiled triton op, calculate the 
        """
        if self.forward_mode == "capture":
            # capture a cuda graph
            # print("in capture mode")
            # set up graph
            graph = torch.cuda.CUDAGraph()
            graph.enable_debug_mode()
            graph.debug_dump("./graph_debug.dot")
            self.graph = graph
            self.static_weight = torch.rand_like(self.boft_s["batched_adapter"], device=x.device)
            # record the part using triton
            with torch.cuda.device(x.device):
                with torch.cuda.graph(self.graph):
                    self.static_weight = self.batched_triton_activation(self.static_weight)

                result = x * self.static_weight

                result = self.base_layer(result, *args, **kwargs)
                result = result.to(x.dtype)
                self.static_weight.copy_(self.boft_s["batched_adapter"])
                return result
                    
        elif self.forward_mode == "use_graph":
            if self.static_weight is None:
                raise Exception("Static weight is not initialized. Was graph captured?")
            self.graph.replay()
            # print("using cuda graph")
            result = x * self.static_weight
            result = self.base_layer(result, *args, **kwargs)
            result = result.to(x.dtype)
            self.static_weight.copy_(self.boft_s["batched_adapter"], non_blocking=True) # reset static weight to boft s
            return result
        elif self.forward_mode in ["regular", "warm_up"]:
            # print("in regular inference mode")
            with torch.cuda.device(x.device):
                static_weight = self.boft_s["batched_adapter"].clone().detach()
                static_weight = self.batched_triton_activation(static_weight)
            
                result = x * static_weight

                result = self.base_layer(result, *args, **kwargs)
                result = result.to(x.dtype)
                return result

    def batched_triton_activation(self, static_weight):
        # static_weight is boft_s at first.
        # prepare
        boft_R = self.boft_R["batched_adapter"]
        op_lst = self.batch_op["batched_adapter"]
        static_weight = static_weight.unsqueeze(0)
        # S^T x B1 x B2 ... x Bn
        for i in range(len(op_lst)):
            op = op_lst[i]
            op(static_weight, boft_R[i:i+1, :, :, :], static_weight)
        static_weight = static_weight.squeeze(0)

        return static_weight


    def set_forward_mode(self, mode):
        self.forward_mode = mode
    
    def move_batch_op(self, ops, device):
        """
        Move a list of triton matmul ops to the input device, 
        return a copy of the ops for the target device
        Args:
            ops (`List[triton.ops.blocksparse.matmul]`):
                a list of triton matmul objs
            device (`torch.device`):
                the target device
        """
        ret_ops = []
        assert isinstance(device, torch.device)
        for i in range(len(ops)):
            op = ops[i]
            if isinstance(op, matmul):
                new_op = matmul(op.layout.to(device), 
                                op.block, 
                                op.mode, 
                                device, 
                                op.trans_a, 
                                op.trans_b, 
                                op.trans_c)
                ret_ops.append(new_op)
        return ret_ops


        





class Conv2d(nn.Module, BOFTLayer):
    """
    BOFT implemented in a Conv2d layer.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        boft_block_size: int = 8,
        boft_block_num: int = 0,
        boft_n_butterfly_factor: int = 0,
        boft_dropout: float = 0.1,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        BOFTLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name

        # Attempt to load the CUDA extension during model initialization
        if not get_fbd_cuda():
            self.fbd_cuda_available = False
            # If the CUDA extension is not available, set the butterfly factor to 1 to speed up the finetuning process
            boft_n_butterfly_factor = 1
        else:
            self.fbd_cuda_available = True

        self.update_layer(
            adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
        )

    def update_layer(
        self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor, boft_dropout, init_weights
    ):
        """
        Update the conv2d layer with trainable BOFT weights.
        """
        # to be consistent with the paper notation
        boft_n_butterfly_factor = boft_n_butterfly_factor - 1
        if boft_n_butterfly_factor < 0:
            raise ValueError(
                f"You can only specify boft_n_butterfly_factor {boft_n_butterfly_factor+1} to be a positive integer number."
            )

        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        # layer information from the base layer
        base_layer = self.get_base_layer()
        conv_filter_dim = self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]

        # Initialize the BOFT parameters.
        if not (boft_block_size != 0) ^ (boft_block_num != 0):
            raise ValueError(
                f"You can only specify either boft_block_size ({boft_block_size}) or boft_block_num ({boft_block_num}), but not both simultaneously, because boft_block_size x boft_block_num != in_features."
            )

        if boft_block_size == 0 and boft_block_num != 0:
            if conv_filter_dim % boft_block_num != 0:
                raise ValueError(
                    f"Convolutional kernel dimension ({conv_filter_dim}) must be divisible by boft_block_num ({boft_block_num})!"
                )

            if boft_n_butterfly_factor != 0:
                if boft_n_butterfly_factor > int(math.log2(boft_block_num)):
                    raise ValueError(
                        f"Invalid combination of boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_num ({boft_block_num})!"
                    )
                if boft_block_num % (2**boft_n_butterfly_factor) != 0:
                    raise ValueError(
                        f"boft_block_num ({boft_block_num}) must be a multiple of 2 raised to the power of boft_n_butterfly_factor ({boft_n_butterfly_factor+1})!"
                    )

            boft_block_size = int(conv_filter_dim // boft_block_num)

        elif boft_block_size != 0 and boft_block_num == 0:
            if conv_filter_dim % boft_block_size != 0:
                raise ValueError(
                    f"Convolutional kernel dimension ({conv_filter_dim}) must be divisible by boft_block_size ({boft_block_size})!"
                )

            if boft_n_butterfly_factor != 0:
                if conv_filter_dim < (boft_block_size * (2**boft_n_butterfly_factor)):
                    raise ValueError(
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )
                if conv_filter_dim % (boft_block_size * (2**boft_n_butterfly_factor)) != 0:
                    raise ValueError(
                        f"Invalid combination of convolutional kernel dimension ({conv_filter_dim}), boft_n_butterfly_factor ({boft_n_butterfly_factor+1}) and boft_block_size ({boft_block_size})!"
                    )

            boft_block_num = int(conv_filter_dim // boft_block_size)

        else:
            raise ValueError("Unknown error!")

        # In OFT you can specify the number of blocks to be 1
        if boft_n_butterfly_factor != 0:
            if boft_block_num % 2 != 0:
                raise ValueError(f"boft_block_num ({boft_block_num}) must be an even number!")

            if boft_block_size % 2 != 0:
                raise ValueError(f"boft_block_size ({boft_block_size}) must be an even number!")

        # If there is no butterfly factor, then permutation matrix P will be an identity matrix.
        P = torch.empty((boft_n_butterfly_factor + 1, conv_filter_dim, conv_filter_dim))
        for i in range(boft_n_butterfly_factor + 1):
            perm = self.block_butterfly_perm(
                conv_filter_dim, int(boft_block_num / (2 ** (i))), int(boft_block_size / 2), boft_n_butterfly_factor
            )
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat

        self.register_buffer("boft_P", P)

        self.boft_R[adapter_name] = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor + 1, boft_block_num, boft_block_size, boft_block_size)
        )
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(1, int(self.out_features)))

        self.reset_boft_parameters(adapter_name, init_weights)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

        # set the boft block size and number
        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.boft_R.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                    orig_weight = orig_weight.view(
                        self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0], self.out_features
                    )
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = orig_weight * boft_s
                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    self.base_layer.weight.data = orig_weight
                else:
                    butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                    orig_weight = base_layer.weight.data.clone()
                    orig_weight = orig_weight.view(
                        self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0], self.out_features
                    )
                    orig_weight = torch.mm(butterfly_oft_mat, orig_weight)
                    orig_weight = orig_weight * boft_s
                    orig_weight = orig_weight.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    self.base_layer.weight.data = orig_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.boft_R.keys():
                butterfly_oft_mat, boft_s = self.get_delta_weight(active_adapter)

                orig_weight = self.get_base_layer().weight.data.clone()
                orig_weight = orig_weight.view(
                    self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0],
                    self.out_features,
                )
                orig_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
                orig_weight = orig_weight * (1 / boft_s)
                orig_weight = orig_weight.view(
                    self.out_features,
                    self.in_features,
                    self.get_base_layer().kernel_size[0],
                    self.get_base_layer().kernel_size[0],
                )

                self.get_base_layer().weight.data = orig_weight

    def get_delta_weight(self, adapter) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter]

        N, D, H, _ = boft_R.shape
        boft_R = boft_R.view(N * D, H, H)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
        if self.fbd_cuda_available:
            block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
        else:
            orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
            block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
            block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)

        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            boft_rotation = torch.eye(
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0], device=x.device
            )
            boft_scale = torch.ones((1, int(self.out_features)), device=x.device)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.boft_R.keys():
                    continue
                boft_R = self.boft_R[active_adapter]
                boft_s = self.boft_s[active_adapter]
                dropout = self.boft_dropout[active_adapter]

                N, D, H, _ = boft_R.shape
                boft_R = boft_R.view(N * D, H, H)
                orth_rotate_butterfly = self.cayley_batch(boft_R)
                orth_rotate_butterfly = orth_rotate_butterfly.view(N, D, H, H)
                orth_rotate_butterfly = dropout(orth_rotate_butterfly)
                if self.fbd_cuda_available:
                    block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)
                else:
                    orth_rotate_butterfly = orth_rotate_butterfly.squeeze(0)
                    block_diagonal_butterfly = torch.block_diag(*torch.unbind(orth_rotate_butterfly))
                    block_diagonal_butterfly = block_diagonal_butterfly.unsqueeze(0)

                butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
                butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
                butterfly_oft_mat = butterfly_oft_mat_batch[0]

                for i in range(1, butterfly_oft_mat_batch.shape[0]):
                    butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

                boft_rotation = butterfly_oft_mat @ boft_rotation
                boft_scale = boft_s * boft_scale

            x = x.to(self.base_layer.weight.data.dtype)

            orig_weight = self.base_layer.weight.data
            orig_weight = orig_weight.view(
                self.in_features * self.base_layer.kernel_size[0] * self.base_layer.kernel_size[0],
                self.out_features,
            )
            rotated_weight = torch.mm(boft_rotation, orig_weight)

            scaled_rotated_weight = rotated_weight * boft_scale

            scaled_rotated_weight = scaled_rotated_weight.view(
                self.out_features, self.in_features, self.base_layer.kernel_size[0], self.base_layer.kernel_size[0]
            )
            result = F.conv2d(
                input=x,
                weight=scaled_rotated_weight,
                bias=self.base_layer.bias,
                padding=self.base_layer.padding[0],
                stride=self.base_layer.stride[0],
            )

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "boft." + rep
