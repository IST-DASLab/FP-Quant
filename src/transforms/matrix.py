import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import _Orthogonal
from fast_hadamard_transform import hadamard_transform


MATRIX_INITS = ("identity", "orthogonal", "hadamard", "xavier_normal")


def init_matrix(
    size: int, 
    init: str, 
    device: torch.device = None, 
    dtype: torch.dtype = None
):
    assert init in MATRIX_INITS, f"Invalid matrix initialization {init}"

    if init == "identity":
        m = torch.eye(size, device=device, dtype=dtype)
    elif init == "orthogonal":
        m = torch.empty(size, size, device=device, dtype=dtype)
        nn.init.orthogonal_(m)
    elif init == "hadamard":
        m = torch.eye(size, device=device, dtype=dtype)
        m = hadamard_transform(m, scale=1.0 / math.sqrt(size))
    elif init == "xavier_normal":
        m = torch.empty(size, size, device=device, dtype=dtype)
        nn.init.xavier_normal_(m)
    return m


class BaseMatrix(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self) -> torch.Tensor:
        pass

    @abstractmethod
    def inv_t(self) -> torch.Tensor:
        pass

    @abstractmethod
    def remove_parametrizations(self) -> None:
        pass


class GeneralMatrix(BaseMatrix):

    def __init__(
        self, 
        size: int,
        init: str,
        device: torch.device = None, 
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(init_matrix(size, init, device, dtype))

    def forward(self) -> torch.Tensor:
        return self.weight

    def inv_t(self) -> torch.Tensor:
        return self.weight.pinverse().T
    
    def remove_parametrizations(self) -> None:
        pass


class OrthogonalMatrix(BaseMatrix):

    def __init__(
        self, 
        size: int,
        init: str,
        device: torch.device = None, 
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(init_matrix(size, init, device, dtype))
        orth = _Orthogonal(self.weight, "cayley", use_trivialization=False)
        parametrize.register_parametrization(self, "weight", orth)

    def forward(self) -> torch.Tensor:
        return self.weight

    def inv_t(self) -> torch.Tensor:
        return self.weight
    
    def remove_parametrizations(self) -> None:
        parametrize.remove_parametrizations(self, 'weight', leave_parametrized=True)
    

class SVDMatrix(BaseMatrix):

    def __init__(
        self, 
        size: int,
        init: str,
        device: torch.device = None, 
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.u  = nn.Parameter(init_matrix(size, init, device, dtype))
        self.s  = nn.Parameter(torch.ones(size, device=device, dtype=dtype))
        self.vT = nn.Parameter(init_matrix(size, init, device, dtype))
        # Register orthogonal parametrization
        orth_u = _Orthogonal(self.u, "cayley", use_trivialization=False)
        parametrize.register_parametrization(self, "u", orth_u)
        orth_vT = _Orthogonal(self.vT, "cayley", use_trivialization=False)
        parametrize.register_parametrization(self, "vT", orth_vT)            

    def forward(self) -> torch.Tensor:
        return self.u.mul(self.s[:, None]).mm(self.vT)

    def inv_t(self) -> torch.Tensor:
        return self.u.div(self.s[:, None]).mm(self.vT)
    
    def remove_parametrizations(self) -> None:
        parametrize.remove_parametrizations(self, 'u', leave_parametrized=True)
        parametrize.remove_parametrizations(self, 'vT', leave_parametrized=True)


def l2norm_along_axis1(X: torch.Tensor) -> torch.Tensor:
        return torch.norm(X, p=2, dim=1)

def sample_chi(d, rng=None, device='cpu'):
    """
    Samples from a Chi distribution with `d` degrees of freedom.
    
    Args:
        d (int): The degrees of freedom for the Chi distribution. Also determines the shape of the matrix.
        rng (np.random.RandomState or np.random.Generator, optional): 
            A NumPy random number generator for seeding. If None, uses PyTorch's default RNG.
        device (str or torch.device): The device on which to perform computation ('cpu', 'cuda' or 'xpu').

    Returns:
        torch.Tensor: A 1D tensor of length `d`, where each entry is a sample from Chi(d).
    """
    
    if rng is None:
        # Case 1: No external RNG provided → use PyTorch's default RNG
        # Generate a (d x d) matrix of standard normal samples
        normal_samples = torch.randn((d, d), device=device)
    else:
        # Case 2: A NumPy RNG is provided
        # Create a PyTorch generator and seed it using a random 32-bit integer from the NumPy RNG
        g = torch.Generator(device=device)
        g.manual_seed(rng.randint(0, 2**32 - 1))
        
        # Generate a (d x d) matrix of standard normal samples using the seeded generator
        normal_samples = torch.randn((d, d), generator=g, device=device)

    # Compute the L2 norm (Euclidean norm) of each row
    # This gives `d` samples from a Chi distribution with `d` degrees of freedom
    chi_samples = torch.norm(normal_samples, dim=1)

    return chi_samples
