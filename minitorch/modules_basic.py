"""
For additional transformer related

Sequential
Embedding

"""
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .module import Module, Parameter
from .tensor_functions import (Mul, LT, View, zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor


def UniformRandomParameter(backend, w, *shape):
    r = (2*w)*rand(shape, backend=backend) - w
    return Parameter(r)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(tensor_from_numpy(np.random.normal(0, 1, (num_embeddings, embedding_dim)), backend=backend, requires_grad=True))
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        # flatten sequences for lookup
        lookups = one_hot(x.view(bs * seq_len), self.num_embeddings) @ self.weights.value
        return lookups.view(bs, seq_len, self.embedding_dim)
        ### END ASSIGN3_2


class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes:
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor:
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)

        Args:
            x : Tensor of shape (*)

        Returns:
            output : Tensor of shape (*)
        """
        ### BEGIN ASSIGN3_2
        if not self.training or self.p_dropout <= 0:
            return x
        if self.p_dropout >= 1:
            return x.zeros()
        return Mul.apply(
            x * 1/(1 - self.p_dropout),
            tensor_from_numpy(np.random.binomial(1, 1 - self.p_dropout, x.shape), backend=x.backend)
        )
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        w = 1/math.sqrt(in_size)
        self.weights = UniformRandomParameter(backend, w, in_size, out_size)
        self.bias = UniformRandomParameter(backend, w, out_size) if bias else None
        ### END ASSIGN3_2


    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.

        Args:
            x : Tensor of shape (n, in_size)

        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        activation = (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size)
        return (activation + self.bias.value) if self.bias is not None else activation
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.

        Args:
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.

        Attributes:
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(ones((dim,), backend=backend))
        self.bias = Parameter(zeros((dim,), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs.
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.

        Input:
            x - Tensor of shape (bs, dim)

        Output:
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        return x.layernorm(self.weights.value, self.bias.value)

        """
        ### BEGIN ASSIGN3_2
        difs = x - x.mean(1)
        var = x.var(1) + self.eps
        return (difs / (var ** 0.5)) * self.weights.value + self.bias.value
        ### END ASSIGN3_2
        """
