import flax.linen as nn
import jax.numpy as jnp

from rlpd.networks import default_init
from typing import Sequence


class Conv2DTransposeBlock(nn.Module):
    features: int
    kernel_size: int
    stride: int
    padding: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.ConvTranspose(
            self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            kernel_init=default_init(),
            padding=self.padding
        )(x)
        x = nn.GroupNorm(num_groups=self.features)(x)
        x = nn.leaky_relu(x, negative_slope=0.3)
        # 2nd block
        # x = nn.ConvTranspose(
        #     self.features,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     padding=1
        # )(x)
        # x = nn.GroupNorm(num_groups=self.features)(x)
        # x = nn.leaky_relu(x, negative_slope=0.3)
        return x


class QuantizedDecoder(nn.Module):
    mlp_dims: Sequence[int] = (256, 256, 4096)
    transition_shape: Sequence[int] = (4, 4, 256)
    features: Sequence[int] = (128, 64, 32, 16)
    kernel_sizes: Sequence[int] = (4, 4, 4, 4)
    strides: Sequence[int] = (2, 2, 2, 2)
    padding: Sequence[int] = (2, 2, 2, 2)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.mlp_dims:
            x = nn.Dense(dim, kernel_init=default_init())(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], *self.transition_shape))

        for feature, kernel_size, stride, padding in zip(self.features, self.kernel_sizes, self.strides, self.padding):
            x = Conv2DTransposeBlock(feature, kernel_size, stride, padding)(x)

        # project back to image space
        x = nn.ConvTranspose(3,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             kernel_init=default_init(),
                             padding=1)(x)
        return x