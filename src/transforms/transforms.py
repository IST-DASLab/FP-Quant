import math
from abc import abstractmethod
from typing import Optional, List

import torch
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform

from .matrix import (
    GeneralMatrix,
    OrthogonalMatrix,
    SVDMatrix,
    l2norm_along_axis1, 
    sample_chi
)
from ..helpers import decompose_dim, split_dim
from ..utils.common_utils import filter_kwarg_dict


MATRIX_PARAMETRIZATIONS = {
    "general": GeneralMatrix,
    "orthogonal": OrthogonalMatrix,
    "svd": SVDMatrix, # TODO also general, but in different format - rename?
}


class BaseTransform(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        pass

    @abstractmethod
    def remove_parametrizations(self) -> None:
        pass


class IdentityTransform(BaseTransform):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        return x
    
    def remove_parametrizations(self) -> None:
        pass


class FullTransform(BaseTransform):

    def __init__(
        self, 
        size: int, 
        init: str = "orthogonal",
        parametrization: str = "general",
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.matrix = MATRIX_PARAMETRIZATIONS[parametrization](size, init, device, dtype)

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        t = self.matrix() if not inv_t else self.matrix.inv_t()
        return torch.tensordot(x, t, dims=((dim,), (0,)))
    
    def remove_parametrizations(self) -> None:
        self.matrix.remove_parametrizations()


class HadamardTransform(BaseTransform):

    def __init__(self, group_size: int = 128):
        super().__init__()
        self.group_size = group_size
        self.scale = 1 / math.sqrt(self.group_size)

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # Hadamard transform is it own inverse
        x_shape = x.shape
        return hadamard_transform(x.view(-1, self.group_size), scale=self.scale).view(x_shape)
    
    def remove_parametrizations(self) -> None:
        pass


class KroneckerFactorizedTransform(BaseTransform):

    def __init__(
        self, 
        size: int, 
        init: str = "orthogonal",
        parametrization: str = "general",
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.lsize, self.rsize = decompose_dim(size)
        self.lmatrix = MATRIX_PARAMETRIZATIONS[parametrization](self.lsize, init, device, dtype)
        self.rmatrix = MATRIX_PARAMETRIZATIONS[parametrization](self.rsize, init, device, dtype)
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if dim == -1:
            dim = x.ndim - 1
            
        l = self.lmatrix() if not inv_t else self.lmatrix.inv_t()
        r = self.rmatrix() if not inv_t else self.rmatrix.inv_t()
        x = split_dim(x, self.lsize, dim)
        x = torch.matmul(x.movedim(dim, -1), l).movedim(-1, dim)
        x = torch.matmul(x.movedim(dim + 1, -1), r).movedim(-1, dim + 1)
        return x.flatten(dim, dim + 1)
    
    def remove_parametrizations(self) -> None:
        self.lmatrix.remove_parametrizations()
        self.rmatrix.remove_parametrizations()


class BlockDiagonalTransform(BaseTransform):
    pass

class IdentityLowRankTransform(BaseTransform):

    def __init__(
        self, 
        size: int, 
        rank: int,
        alpha: Optional[float] = None,
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha or rank
        self.lora_A = nn.Parameter(torch.empty(rank, size, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(size, rank, device=device, dtype=dtype)) 
        # Following LoRA paper
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if dim == -1:
            dim = x.ndim - 1

        if inv_t:
            # Woodbury Matrix identity
            inv = torch.eye(self.rank, device=x.device, dtype=x.dtype) + self.lora_A.mm(self.lora_B)
            res = torch.tensordot(x, self.lora_A.T, dims=((dim,), (0,)))
            res = torch.tensordot(res, inv, dims=((dim,), (0,)))
            res = -torch.tensordot(res, self.lora_B.T, dims=((dim,), (0,)))
        else:
            res = torch.tensordot(x, self.lora_A.T, dims=((dim,), (0,)))
            res = torch.tensordot(res, self.lora_B.T, dims=((dim,), (0,)))

        return x + (self.alpha / self.rank) * res
    
    def remove_parametrizations(self) -> None:
        pass


class CompositeTransform(BaseTransform):
    
    def __init__(self, transforms: List[BaseTransform]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        for transform in self.transforms:
            x = transform(x, inv_t, dim)
        return x
    
    def remove_parametrizations(self) -> None:
        for transform in self.transforms:
            transform.remove_parametrizations()

class DCTTransform(BaseTransform):
    
    def __init__(self, group_size: int = 128):
        super().__init__()
        import numpy as np
        from scipy.fftpack import dct
        self.group_size = group_size
        self.block_dct = torch.from_numpy(
                dct(np.eye(group_size), type=2, norm='ortho')
                )
        self.mat = None

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # DCT transform is it own inverse
        x_shape = x.shape
        if self.mat is None:
            self.mat = torch.block_diag(
                *[self.block_dct] * (x_shape[-1] // self.group_size),
            ).to(x.device).to(x.dtype)
        return torch.matmul(x, self.mat)

    def remove_parametrizations(self) -> None:
        pass
    
class DSTransform(BaseTransform):
    
    def __init__(self, group_size: int = 128):
        super().__init__()
        import numpy as np
        from scipy.fftpack import dst
        self.group_size = group_size
        self.block_dct = torch.from_numpy(
                dst(np.eye(group_size), type=2, norm='ortho')
                )
        self.mat = None

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # DST transform is it own inverse
        x_shape = x.shape
        if self.mat is None:
            self.mat = torch.block_diag(
                *[self.block_dct] * (x_shape[-1] // self.group_size),
            ).to(x.device).to(x.dtype)
        return torch.matmul(x, self.mat)
    
    def remove_parametrizations(self) -> None:
        pass


class FastFoodTransform(BaseTransform):
    def __init__(self, group_size: int = 128):
        '''
        Implemented based on the FastFood transform paper:
            - https://arxiv.org/pdf/1408.3060
            - https://scikit-learn-extra.readthedocs.io/en/stable/_modules/sklearn_extra/kernel_approximation/_fastfood.html#Fastfood
        '''
        
        super().__init__()
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        sigma = 1 / math.sqrt(2)
        B = torch.diag(torch.randint(0, 2, (group_size,)).float() * 2 - 1)
        G = torch.diag(torch.randn(group_size))
        H = hadamard_transform(torch.eye(group_size, device=device)).cpu()
        S = torch.diag((1 / l2norm_along_axis1(G)) *  sample_chi(group_size))
        P = torch.eye(group_size)[torch.randperm(group_size, device='cpu')].to(torch.float32)
        self.block_mat = (1 / sigma) *(1 / math.sqrt(group_size) )*(S@H@G@P@H@B)
        self.block_inv_mat = torch.linalg.inv(self.block_mat)
        self.group_size = group_size
        self.mat = None
        self.inv_mat = None
        
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if inv_t:
            x_shape = x.shape
            if self.inv_mat is None:
                self.inv_mat = torch.block_diag(
                    *[self.block_inv_mat] * (x_shape[-1] // self.group_size),
                ).to(x.device).to(x.dtype).T
                del self.block_inv_mat
                
            return torch.matmul(x, self.inv_mat)
        else:
            x_shape = x.shape
            if self.mat is None:
                self.mat = torch.block_diag(
                    *[self.block_mat] * (x_shape[-1] // self.group_size),
                ).to(x.device).to(x.dtype)
                del self.block_mat
            return torch.matmul(x, self.mat)


class GSRTransform(BaseTransform):
    
    def __init__(self, group_size: int = 128):
        super().__init__()
        from scipy.linalg import hadamard
        self.group_size = group_size
        q_ = torch.tensor(hadamard(self.group_size), dtype=torch.float64)
        sign_changes = torch.diff(q_, dim=0).ne(0).sum(dim=0) 
        sorted_indices = torch.argsort(sign_changes)
        q_ = q_[:, sorted_indices]
        self.block_gsr= q_ / torch.tensor(q_.shape[-1]).sqrt()

        self.mat = None

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # GSR transform is it own inverse
        x_shape = x.shape
        if self.mat is None:
            self.mat = torch.block_diag(
                *[self.block_gsr] * (x_shape[-1] // self.group_size),
            ).to(x.device).to(x.dtype)
        return torch.matmul(x, self.mat)

    def remove_parametrizations(self) -> None:
        pass
        

TRANSFORMS = {
    "identity": IdentityTransform,
    "full": FullTransform,
    "hadamard": HadamardTransform,
    "kronecker": KroneckerFactorizedTransform,
    "identity_low_rank": IdentityLowRankTransform,
    "dct": DCTTransform,
    "dst": DSTransform,
    "fast_food": FastFoodTransform,
    "gsr": GSRTransform
}


def build_transform(transform_class: str, **transform_kwargs) -> BaseTransform:
    transform = TRANSFORMS[transform_class]
    return transform(**filter_kwarg_dict(transform.__init__, transform_kwargs))

def get_transform_matrix(
    transform_class: str, 
    size: int, 
    device: torch.device = None, 
    dtype: torch.dtype = None
) -> torch.Tensor:
    if transform_class == "hadamard":
        return hadamard_transform(torch.eye(size, device=device, dtype=dtype), scale=1 / math.sqrt(size))
    elif transform_class == "identity":
        return torch.eye(size, device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"get_transform_matrix is implemented only for Hadamard and Identity transforms")
