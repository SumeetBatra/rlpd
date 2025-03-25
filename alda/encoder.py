import flax.linen as nn
import jax.lax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict
from rlpd.networks import default_init
from typing import Sequence, Type, Tuple, Union, Dict, Optional

LATENT_MAPPING = {  # map num_latents -> in_features for 1D CNN encoder
    6: 32,
    12: 64,
    20: 96,
    32: 144,
    40: 176,
    64: 272,
    128: 528,
}

class Conv2DBlock(nn.Module):
    features: int
    kernel_size: int
    stride: int
    padding: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            kernel_init=default_init(),
            padding=self.padding
        )(x)
        x = nn.GroupNorm(num_groups=self.features)(x)
        x = nn.leaky_relu(x, negative_slope=0.3)
        # 2nd block
        x = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1
        )
        x = nn.GroupNorm(num_groups=self.features)(x)
        x = nn.leaky_relu(x, negative_slope=0.3)
        return x


class QuantizedEncoder(nn.Module):
    num_latents: int
    features: Sequence[int] = (32, 64, 128, 256),
    kernel_sizes: Sequence[int] = (4, 4, 4, 4)
    strides: Sequence[int] = (2, 2, 2, 2)
    padding: Sequence[int] = (1, 1, 1, 1)

    mlp_hidden_layers: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for features, kernel_size, stride, padding in zip(self.features, self.kernel_sizes, self.strides, self.padding):
            x = Conv2DBlock(features, kernel_size, stride, padding)(x)

        x = x.reshape((*x.shape[:-3], -1))
        for hidden_layer in self.mlp_hidden_layers:
            x = nn.Dense(hidden_layer, kernel_init=default_init())(x)
            x = nn.relu(x)

        # final layer
        x = nn.Dense(self.num_latents, kernel_init=default_init())(x)
        return x


class HistoryEncoder(nn.Module):
    num_latents: int
    emb_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            8,
            kernel_size=4,
            strides=2,
            kernel_init=default_init(),
            padding=1
        )(x)
        x = nn.gelu(x)
        x = nn.Conv(16,
                    kernel_size=4,
                    strides=2,
                    kernel_init=default_init(),
                    padding=2
                    )(x)
        x = nn.gelu(x)
        x = x.reshape((x.shape[0], *x.shape[1:]))
        x = nn.Dense(self.emb_dim, kernel_init=default_init())(x)
        return x


class ALDAEncoder(nn.Module):
    encoder_cls: Type[nn.Module]
    history_encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module] # actor or critic model
    latent_dim: int
    stop_gradient: bool = False 
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()
    
    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        observations = FrozenDict(observations)

        xs = []
        for i, pixel_key in enumerate(self.pixel_keys):
            x = observations[pixel_key].astype(jnp.float32) / 255.0
            x = self.encoder_cls(x)
            x = self.history_encoder_cls(x)
            if self.stop_gradient:
                x = jax.lax.stop_gradient(x)

            x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)
        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)