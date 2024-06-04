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
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from peft.tuners.lycoris_utils import LycorisLayer, check_adapters_to_merge


class OFTLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("oft_r",)
    # other_param_names is defined on parent class

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # OFT info
        self.oft_r = nn.ParameterDict({})
        self.coft = {}
        self.eps = {}
        self.block_share = {}
        
        # for batch adapter operations:
        self.batch_op = {}

    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.oft_r}

    def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...], block_share: bool):
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(1, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))
        else:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(r, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))

    def reset_adapter_parameters(self, adapter_name: str):
        nn.init.zeros_(self.oft_r[adapter_name])

    def reset_adapter_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.oft_r[adapter_name], a=math.sqrt(5))

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        module_dropout: float,
        init_weights: bool,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
            coft (`bool`): Whether to use the constrained variant of OFT or not.
            eps (`float`):
                The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
            block_share (`bool`): Whether to share the OFT parameters between blocks or not.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.module_dropout[adapter_name] = module_dropout
        self.coft[adapter_name] = coft
        self.block_share[adapter_name] = block_share

        # Determine shape of OFT weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            shape = (
                base_layer.out_channels,
                base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
            )
        else:
            raise TypeError(f"OFT is not implemented for base layers of type {type(base_layer).__name__}")

        self.eps[adapter_name] = eps * math.ceil(shape[0] / r) * math.ceil(shape[0] / r)

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape, block_share)

        # Initialize weights
        if init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)

        # Move new weights to device
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def unscale_layer(self, scale=None) -> None:
        # scale is not used
        pass

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()

                orig_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                        ]
                    )
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if orig_weights.shape[1] != delta_weight.shape[1]:
                    # when in channels is not divisible by r
                    delta_weight = delta_weight[: orig_weights.shape[1], : orig_weights.shape[1]]
                new_weights = torch.mm(orig_weights, delta_weight)
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = torch.transpose(new_weights, 0, 1)
                    new_weights = new_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )

                if safe_merge and not torch.isfinite(new_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = new_weights
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
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                new_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = new_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                        ]
                    )
                    new_weights = torch.transpose(new_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if new_weights.shape[1] != delta_weight.shape[1]:
                    # when in channels is not divisible by r
                    delta_weight = delta_weight[: new_weights.shape[1], : new_weights.shape[1]]
                delta_inv = torch.inverse(delta_weight)
                orig_weights = torch.mm(new_weights, delta_inv)

                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights.reshape(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )
                base_layer.weight.data = orig_weights

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        rank = self.r[adapter_name]
        coft = self.coft[adapter_name]
        eps = self.eps[adapter_name]
        opt_r = self.oft_r[adapter_name]

        if coft:
            with torch.no_grad():
                opt_r.copy_(self._project_batch(opt_r, eps=eps))

        orth_rotate = self._cayley_batch(opt_r)
        weight = self._block_diagonal(orth_rotate, rank)

        return weight

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L144
    def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)  # noqa: E741

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L155
    def _block_diagonal(self, oft_r: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_r.shape[0] == 1:
            # block share
            blocks = [oft_r[0, ...] for i in range(rank)]
        else:
            blocks = [oft_r[i, ...] for i in range(rank)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L52
    def _project_batch(self, oft_r, eps=1e-5):
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
        I = (  # noqa: E741
            torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype)
            .unsqueeze(0)
            .expand_as(oft_r)
        )
        diff = oft_r - I
        norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
        return out

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            if len(result.shape) == 4:
                result = result.permute(0, 2, 3, 1)

            base_layer = self.get_base_layer()
            base_bias = base_layer.bias
            if base_bias is not None:
                # Bias should be added after OFT forward
                result = result - base_bias.data

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    # differentiate between batched adapters and single adapters.
                    if active_adapter == "batched_adapter":
                        result = self.get_batched_activation(active_adapter, result)
                    else:
                        result = self._get_delta_activations(active_adapter, result, *args, **kwargs)

            if base_bias is not None:
                result = result + base_bias.data
            if len(result.shape) == 4:
                result = result.permute(0, 3, 1, 2)

        result = result.to(previous_dtype)
        return result

    def get_batched_activation(self, active_adapter:str, input:torch.tensor):
        """
        Same function as get_delta_activation. But with the following difference:
        1. use block sparse format and triton ops to comput.
        Args:
            active_adapter: the batched adapter's name 
            NOTE: currently assuming the name is always "batched_adapter"
            input: the data from the previous step in forward, before this function 
                is called. 

        Returns:
            The result after multiplying the weights from the batched adapter
        """
        if active_adapter != "batched_adapter":
            raise Exception("active adapter is not batched_adapter, should not use get_batched_activation")
        
        op = self.batch_op[active_adapter]
        batched_oft_r = self.oft_r[active_adapter]
        # process input to fit triton format (assuming 3D input tensor, triton requires 4D)
        input = input.unsqueeze(0)
        result = op(input, batched_oft_r)
        # return the first element (assuming 3D input tensor)
        return result[0]



    def batch_adapters(self, adapter_lst):
        """
        Given a list of adapter names:
        1. stack the adapters together, and 
        2. add the new adapter and its triton op to the current layer. 
        3. Set the current active adapter to this batched adapter. 

        Params:
        adapter_lst: a list of adapter names. All adapters must be already loaded
            to the layer. And their block size must be the same.
        
        Returns:
            Nothing. 

        NOTE: currently assuming the input adapters will have the same layout.
            If they don't have the same layout (rank * r * r), this won't work,
            as we can't stack the adapters together. 
        """
        # print("oft: batching adapters")
        # print("oft: layer_names:", self.adapter_layer_names)

        for layer_name in self.adapter_layer_names:
            batched_adapters_lst = []
            batched_adapters_rank_lst = []
            module_dict = getattr(self, layer_name)
            for adapter in adapter_lst:
                if adapter not in module_dict.keys():
                    raise KeyError(f"adapter {adapter} not found in available adapters")
                else: 
                    layer = module_dict[adapter]
                    # NOTE: before the layers are stacked together, need to transform each
                    # of them to be a orth matrix (cayley batch only applies to 3d 
                    # tensors)
                    orth_rotate_layer = self._cayley_batch(layer)
                    # add the layer to adapter list
                    batched_adapters_lst.append(orth_rotate_layer)
                    batched_adapters_rank_lst.append(self.r[adapter])
                    
                    # clear memory on cuda
                    layer_cpu = layer.to('cpu')
                    module_dict[adapter] = layer_cpu
                    del layer
                    torch.cuda.empty_cache()  # Clear CUDA cache if necessary
                    
            # each adapter shape is n_useful_blocks * m (n_rows) * n (n_cols)
            # we stack on the 0'th dimention
            stacked_adapters = torch.cat(batched_adapters_lst, dim=0)
            # NOTE: here we assume r==c (i.e. square blocks)
            block_size = stacked_adapters.shape[1]

            # This process the 3D weight to 4D to fit triton 
            stacked_adapters = stacked_adapters.unsqueeze(0)
            self.oft_r["batched_adapter"] = stacked_adapters.to(self.base_layer.weight.dtype)

            # Add OFT layout (simply diagonal):
            # Original _block_diagonal transform the block layout to be on the diagonal
            # created by rank. So the layout is a rank x rank identity matrix
            layout_lst = [torch.eye(r,dtype=torch.bool) for r in batched_adapters_rank_lst]
            layout = torch.stack(layout_lst)
            # Compile a Triton OP
            import triton
            import triton.ops
            batch_op = triton.ops.blocksparse.matmul(layout, block_size, "dds", device="cuda")

            self.batch_op["batched_adapter"] = batch_op

            self.set_adapter("batched_adapter")

    def unbatch_adapters(self, adapter_lst):
        """
        Put the adapters in adapter_lst back to cuda
        """
        print("--unbatching adapters--")
        for layer_name in self.adapter_layer_names:
            batched_adapters_lst = []
            batched_adapters_rank_lst = []
            module_dict = getattr(self, layer_name)
            for adapter in adapter_lst:
                if adapter not in module_dict.keys():
                    raise KeyError(f"adapter {adapter} not found in available adapters")
                else: 
                    layer = module_dict[adapter]
                    # put the layer back to cuda

                    print(f"before clear mem: allocated: {torch.cuda.memory_allocated()}, reserved: {torch.cuda.memory_reserved()}")
                    layer_gpu = layer.to('cuda')
                    module_dict[adapter] = layer_gpu
                    del layer
                    torch.cuda.empty_cache()  # Clear CUDA cache if necessary
                    # print("original layer: ", type(layer), "device:", layer.device)
                    print(f"after clear mem:  allocated: {torch.cuda.memory_allocated()}, reserved: {torch.cuda.memory_reserved()}")
                    



class Linear(OFTLayer):
    """OFT implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, module_dropout, init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()
        base_weight = base_layer.weight.data
        delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

        # don't add bias here, because the bias will be added after OFT forward
        return torch.matmul(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


class Conv2d(OFTLayer):
    """OFT implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, module_dropout, init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()
        base_weight = base_layer.weight.data
        delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

        # don't add bias here, because the bias will be added after OFT forward
        return torch.matmul(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep
