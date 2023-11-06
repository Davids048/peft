# coding=utf-8
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

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load, CUDA_HOME
from torch.autograd import Function

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"
curr_dir = os.path.dirname(__file__)
fbd_cuda = \
    load(name='fbd_cuda', 
        sources=[f'{curr_dir}/fbd/fbd_cuda.cpp', f'{curr_dir}/fbd/fbd_cuda_kernel.cu'], verbose=True,
        build_directory='/tmp/'
        )
        # extra_cuda_cflags = ['-std=c++14', '-ccbin=$$(which gcc-7)']) # cuda10.2 is not compatible with gcc9. Specify gcc 7 

import fbd_cuda

class FastBlockDiag(Function):
    """
    Implements a custom autograd Function for a fast block diagonal operation using CUDA.

    This function is optimized for 4D tensors where the last two dimensions are equal, 
    representing block diagonal matrices for efficient computation on CUDA devices.
    """

    @staticmethod
    def forward(ctx, input):
        """
        The forward method for FastBlockDiag.

        Computes the block diagonal operation on the input tensor using a CUDA-optimized function.
        This method assumes that the input is a 4D tensor where the last two dimensions are equal,
        which represent the blocks to be diagonalized.

        Parameters:
        ctx: A context object that can be used to stash information for backward computation.
        input (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size,
                        `D` represents one additional dimension (In BOFT, the number of BOFT blocks), 
                        and `H` is the size of the square blocks along the last two dimensions 
                        (In BOFT, the block size).

        Returns:
        Tensor: The resulting tensor after applying the block diagonal operation, 
                will have the shape (N, DxH, DxH).
        """
        output = fbd_cuda.forward(input)[0]
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = fbd_cuda.backward(
            grad_output, input)[0]
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
        super(MultiplicativeDropoutLayer, self).__init__()
        self.p = p

    def forward(self, x):
        """
        The forward method for MultiplicativeDropoutLayer.

        Applies multiplicative dropout to the input tensor.
        Parameters:
        x (Tensor): The input tensor of shape (N, D, H, H), where `N` is the batch size, `D` represents 
                    one additional dimension (In BOFT, the number of BOFT blocks), and `H` is the size 
                    of the square blocks along the last two dimensions (In BOFT, the block size).
        """
        if self.training:
            # Ensure the last two dimensions are the same
            assert x.shape[-1] == x.shape[-2], "The last two dimensions of input should be the same!"

            N, D, H, _ = x.shape

            # Randomly select one from N
            n_random = torch.randint(0, N, (1,)).item()

            # Create a mask with 1s for matrices to be replaced with identity and 0s otherwise
            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace

            # Generate a flat tensor with desired number of 1s and 0s
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])

            # Shuffle and reshape the mask
            mask = mask[torch.randperm(D)].view(1, Z, 1, 1)

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
    def __init__(self, in_features: int, out_features: int, **kwargs):
        """
        Initializes the BOFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be added soon.

        Parameters:
        in_features (int): The dimension of the input tensor.
        out_features (int): The dimension of the output tensor.
        """
        self.boft_block_size = {}
        self.boft_block_num = {}
        self.boft_dropout = nn.ModuleDict({})
        self.boft_R = nn.ParameterDict({})
        self.boft_s = nn.ParameterDict({})
        self.boft_b = nn.ParameterDict({})
        # For Embedding layer
        self.boft_embedding_R = nn.ParameterDict({})
        self.boft_embedding_s = nn.ParameterDict({})
        self.boft_embedding_b = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    def update_layer(self, adapter_name, boft_block_size, boft_block_num, boft_dropout, init_boft_weights):
        """
        Update the linear layer with trainable BOFT weights.
        """
        # Initialize the MultiplicativeDropoutLayer for boft_dropout > 0.0.
        if boft_dropout > 0.0:
            boft_dropout_layer = MultiplicativeDropoutLayer(p=boft_dropout)
        else:
            boft_dropout_layer = nn.Identity()
        self.boft_dropout.update(nn.ModuleDict({adapter_name: boft_dropout_layer}))

        # Initialize the BOFT parameters.
        assert (boft_block_size != 0) ^ (boft_block_num != 0), "You can only specify either boft_block_size or boft_block_num, but not both simultaneously, because boft_block_size x boft_block_num != in_features."
        assert boft_block_size % 2 == 0, "You must set the boft_block_size to be an even number!"

        if boft_block_size == 0 and boft_block_num != 0:
            assert self.in_features % boft_block_num == 0, "in_features must be divisible by boft_block_num"
            if self.kwargs["boft_n_butterfly_factor"] != 0:
                assert self.kwargs["boft_n_butterfly_factor"] <= int(math.log2(boft_block_num)), "invalid combination of boft_n_butterfly_factor and boft_block_num"
                assert boft_block_num % (2**self.kwargs["boft_n_butterfly_factor"]) == 0, "boft_block_num must be a power of 2"
            boft_block_size = int(self.in_features // boft_block_num)

        elif boft_block_size != 0 and boft_block_num == 0:
            assert self.in_features % boft_block_size == 0, "in_features must be divisible by boft_block_size"
            if self.kwargs["boft_n_butterfly_factor"] != 0:
                assert self.in_features >= boft_block_size * (2**self.kwargs["boft_n_butterfly_factor"]), "invalid combination of boft_n_butterfly_factor and boft_block_size"
                assert self.in_features % (boft_block_size * (2**self.kwargs["boft_n_butterfly_factor"])) == 0, "invalid combination of boft_n_butterfly_factor and boft_block_size"
            boft_block_num = int(self.in_features // boft_block_size)

        else:
            print('Unknown error!')
            sys.exit()

        # If there is no butterfly factor, then permutation matrix P will be an identity matrix.
        P = torch.empty((self.kwargs["boft_n_butterfly_factor"]+1, self.in_features, self.in_features))
        for i in range((self.kwargs["boft_n_butterfly_factor"]+1)):
            perm = self.block_butterfly_perm(self.in_features, int(boft_block_num/(2**(i))), int(boft_block_size / 2))
            perm_mat = self.perm2mat(perm)
            P[i] = perm_mat

        self.register_buffer('boft_P', P)

        self.boft_R[adapter_name] = nn.Parameter(torch.zeros(self.kwargs["boft_n_butterfly_factor"]+1, boft_block_num, boft_block_size, boft_block_size))
        self.boft_s[adapter_name] = nn.Parameter(torch.ones(int(self.out_features), 1))
        self.boft_b[adapter_name] = nn.Parameter(torch.ones(int(self.out_features)))

        if init_boft_weights:
            self.reset_boft_parameters(adapter_name)

        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

        self.boft_block_size[adapter_name] = boft_block_size
        self.boft_block_num[adapter_name] = boft_block_num

    def reset_boft_parameters(self, adapter_name):
        """
        Reset the BOFT parameters.
        """
        if adapter_name in self.boft_R.keys():
            # initialize R to zero   
            nn.init.zeros_(self.boft_R[adapter_name])
            nn.init.ones_(self.boft_s[adapter_name])
            nn.init.ones_(self.boft_b[adapter_name])

    def perm2mat(self, indices):
        """
        Convert permutation indices to permutation matrix.
        """
        # Number of indices determines the size of the square matrix
        n = len(indices)
        
        # Initialize a matrix of zeros
        perm_mat = torch.zeros((n, n))
        
        # Set the 1s according to the indices
        for i, idx in enumerate(indices):
            perm_mat[i, idx] = 1
        
        return perm_mat

    def block_butterfly_perm(self, n, b, r=3):
        """
        Define the permutation matrix for the block butterfly permutation.

        Args:
        n: size of the permutation matrix
        b: desired number of blocks after multiplying with the permutation matrix
        r: base block size of the block diagonal matrix, e.g. 2x2, 3x3, 5x5 etc.
        """

        assert b * r * 2 <= n, "Invalid number of blocks!"

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
                sorted_order[int(i*r):int(i*r+r)] = initial_order[int(pos*r):int(pos*r+r)]
            return sorted_order

        sorted_order = sort_block(block_size, r)

        for i in range(0, n, block_size):
            block_end = i + block_size
            tmp_indices = indices[i:block_end]
            indices[i:block_end] = tmp_indices[sorted_order]
        return indices


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Linear, LoraLayer):
    # BOFT implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        # this gets the init from nn.Linear's super perspective, i.e.
        # nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        # Note that we don't use self._init_empty_weights() for Linear because it is a bit slower and the benefit of
        # added robustness is not big enough for Linear.

        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.set_adapter(adapter_name)

    def merge(self) -> None:
        if self.active_adapter not in self.boft_R.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.boft_block_size[self.active_adapter] > 0 and self.boft_block_num[self.active_adapter] > 0:
            # self.weight.data += self.get_delta_weight(self.active_adapter)
            orig_weight = self.weight.data
            butterfly_oft_mat, boft_s, boft_b = self.get_delta_weight(self.active_adapter)

            orig_weight = torch.transpose(orig_weight, 0, 1)
            rotated_weight = torch.mm(butterfly_oft_mat, orig_weight)
            rotated_weight = torch.transpose(rotated_weight, 0, 1)

            self.weight.data = rotated_weight * boft_s
            self.bias.data = self.bias.data * boft_b
            self.merged = True

    def unmerge(self) -> None:
        if self.active_adapter not in self.boft_R.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.boft_block_size[self.active_adapter] > 0 and self.boft_block_num[self.active_adapter] > 0:
            # self.weight.data -= self.get_delta_weight(self.active_adapter)
            orig_weight = self.weight.data 
            butterfly_oft_mat, boft_s, boft_b = self.get_delta_weight(self.active_adapter)

            orig_weight = torch.transpose(orig_weight, 0, 1)
            rotated_weight = torch.mm(butterfly_oft_mat.t(), orig_weight)
            rotated_weight = torch.transpose(rotated_weight, 0, 1) 

            self.weight.data = rotated_weight * (1 / boft_s)
            self.bias.data = self.bias.data / boft_b
            self.merged = False

    def get_delta_weight(self, adapter) -> torch.Tensor:
        boft_R = self.boft_R[adapter]
        boft_s = self.boft_s[adapter]
        boft_b = self.boft_b[adapter]

        N, Z, b, _ = boft_R.shape
        boft_R = boft_R.view(N * Z, b, b)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, Z, b, b)
        block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        return butterfly_oft_mat, boft_s, boft_b

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))
        # Q = torch.linalg.solve(I + skew, I - skew, left=False)
        
        return Q
    
    def angle2rot(self, alphas):
        c = torch.cos(alphas)
        s = torch.sin(alphas)
        rot_mats = torch.cat([c, -s, s, c], dim=-1).view(alphas.shape[0], alphas.shape[1], 2, 2)
        return rot_mats
    
    def is_orthogonal(self, R, eps=1e-3):
        R = R.float()
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter not in self.boft_R.keys():
            return self._linear(x)

        previous_dtype = x.dtype

        # if self.disable_adapters:
        #     if (self.boft_block_size[self.active_adapter] > 0) and self.merged:
        #         self.unmerge()
        #     result = self._linear(x)
        # elif (self.boft_block_size[self.active_adapter] == 0) or self.merged:
        #     result = self._linear(x)
        # else:

        boft_R = self.boft_R[self.active_adapter]
        boft_s = self.boft_s[self.active_adapter]
        if self.boft_bias_fit:
            boft_b = self.boft_b[self.active_adapter]
        dropout = self.boft_dropout[self.active_adapter]

        # oft_mat = self.cayley_batch(boft_R)
        # oft_mat = dropout(oft_mat)
        # oft_mat = FastBlockDiag.apply(oft_mat.unsqueeze(0)).squeeze(0)

        # if self.kwargs["boft_n_butterfly_factor"] != 0:
        #     boft_R_butterfly = self.boft_R_butterfly[self.active_adapter]

        N, Z, b, _ = boft_R.shape
        boft_R = boft_R.view(N * Z, b, b)
        orth_rotate_butterfly = self.cayley_batch(boft_R)
        orth_rotate_butterfly = orth_rotate_butterfly.view(N, Z, b, b)
        orth_rotate_butterfly = dropout(orth_rotate_butterfly)
        block_diagonal_butterfly = FastBlockDiag.apply(orth_rotate_butterfly)

        # with torch.cuda.amp.autocast(enabled=False):

        butterfly_oft_mat_batch = torch.bmm(block_diagonal_butterfly, self.boft_P.permute(0, 2, 1))
        butterfly_oft_mat_batch = torch.bmm(self.boft_P, butterfly_oft_mat_batch)
        butterfly_oft_mat = butterfly_oft_mat_batch[0]

        for i in range(1, butterfly_oft_mat_batch.shape[0]):
            butterfly_oft_mat = butterfly_oft_mat_batch[i] @ butterfly_oft_mat

        # if self.optim_idx % 5 == 0:
        #     oft_mat = butterfly_mat @ oft_mat #.detach()
        #     self.optim_idx = self.optim_idx + 1
        # else:
        #     oft_mat = butterfly_mat.detach() @ oft_mat
        #     self.optim_idx = self.optim_idx + 1

        # print('percentage of non-zero before', (torch.nonzero(oft_mat).size(0) / oft_mat.numel()) * 100)
        # oft_mat = butterfly_mat @ oft_mat
        # print('percentage of non-zero after ', (torch.nonzero(oft_mat).size(0) / oft_mat.numel()) * 100)

        # scaling = self.scaling[self.active_adapter]

        # print('oft_mat is orthogonal', self.is_orthogonal(oft_mat))
        # print('oft_mat is identity', self.is_identity_matrix(oft_mat))
        # print('butterfly_mat is identity', self.is_identity_matrix(butterfly_mat))

        # print('boft_R', boft_R.norm(p='fro'))
        # print('boft_R_butterfly', boft_R_butterfly.norm(p='fro'))

        x = x.to(boft_R.data.dtype)
        
        orig_weight = self.weight.data
        orig_weight = torch.transpose(orig_weight, 0, 1)
        rotated_weight = torch.mm(butterfly_oft_mat, orig_weight)
        rotated_weight = torch.transpose(rotated_weight, 0, 1)

        scaled_rotated_weight = rotated_weight * boft_s

        # Apply the trainable identity matrix
        if self.boft_bias_fit:
            if self.bias is not None:
                bias_term = self.bias.data * boft_b
            else:
                bias_term = None
        else:
            bias_term = self.bias.data if self.bias is not None else None

        result = F.linear(input=x, weight=scaled_rotated_weight, bias=bias_term)

        result = result.to(previous_dtype)
        return result


class Embedding(nn.Embedding, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self._init_empty_weights(nn.Embedding, num_embeddings, embedding_dim, **kwargs)
        LoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)
        self.update_layer_embedding(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_embedding_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _embed(self, input: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight if weight is None else weight
        return F.embedding(
            input,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._embed(x)
        elif self.merged:
            result = self._embed(x)
        else:
            result = self._embed(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result += (after_A @ embedding_B) * scaling

        return result


class Conv2d(nn.Conv2d, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        adapter_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self._init_empty_weights(nn.Conv2d, in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        LoraLayer.__init__(
            self,
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.update_layer_conv2d(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = self.weight.data.copy()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    self.weight.data = orig_weights
                else:
                    self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _conv2d(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._conv2d(x)
        elif self.merged:
            result = self._conv2d(x)
        else:
            result = self._conv2d(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result