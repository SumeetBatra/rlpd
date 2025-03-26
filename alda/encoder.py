import flax.linen as nn
import jax.lax
import jax.numpy as jnp
import einops

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
        # x = nn.Conv(
        #     self.features,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     padding=1
        # )(x)
        # x = nn.GroupNorm(num_groups=self.features)(x)
        # x = nn.leaky_relu(x, negative_slope=0.3)
        return x


class QuantizedEncoder(nn.Module):
    num_latents: int
    features: Sequence[int] = (32, 64, 128, 256)
    kernel_sizes: Sequence[int] = (3, 3, 3, 3)
    strides: Sequence[int] = (2, 2, 2, 2)
    padding: Sequence[int] = (1, 1, 1, 1)

    mlp_hidden_layers: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, observations: Union[FrozenDict, Dict]) -> jnp.ndarray:
        observations = FrozenDict(observations)
        x = observations['pixels'].astype(jnp.float32) / 255.0  # b h w c f
        if len(x.shape) == 4:
            x = x[None]
        f = x.shape[-1]
        x = einops.rearrange(x, 'b h w c f -> (b f) h w c', f=f)
        # x = jnp.reshape(x, (*x.shape[:-2], -1))
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
        x = x.reshape(*x.shape[:-2], -1)
        x = nn.Dense(self.emb_dim, kernel_init=default_init())(x)
        return x


class ALDAEncoder(nn.Module):
    history_encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]  # actor or critic model
    latent_dim: int
    stop_gradient: bool = False 
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()

    def setup(self):
        self.history_encoder = self.history_encoder_cls()

    def __call__(self,
                 base_encoder: nn.Module,
                 latent_model: Optional[nn.Module],
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        x = self.compute_embeddings(base_encoder, latent_model, observations)
        return self.compute_output(x, actions, training)

    def compute_embeddings(self,
                           base_encoder: nn.Module,
                           latent_model: nn.Module,
                           observations: Union[FrozenDict, Dict]):
        observations = FrozenDict(observations)
        f = observations['pixels'].shape[-1]
        x = base_encoder.apply_fn({"params": base_encoder.params}, observations)  # (b f) e
        x = latent_model.apply_fn({"params": latent_model.params}, x)['z_hat']
        x = einops.rearrange(x, '(b f) e -> b e f', f=f)
        x = self.history_encoder(x)
        return x

    def compute_temporal_embeddings(self, embeds: jnp.ndarray, batch_size: int):
        """
        :param embeds: shape (b f) e tensor where f is framestack, e is embed dim
        :param batch_size: batch_size
        """
        x = einops.rearrange(embeds, '(b f) e -> b e f', b=batch_size)
        x = self.history_encoder(x)
        if self.stop_gradient:
            x = jax.lax.stop_gradient(x)
        return x

    @nn.compact
    def compute_output(self, embeds: jnp.ndarray, actions: Optional[jnp.ndarray] = None, training: bool = False):
        x = nn.Dense(self.latent_dim, kernel_init=default_init())(embeds)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        if x.shape[0] == 1:
            x = x[0]

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)


