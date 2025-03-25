import flax.linen as nn
import jax.numpy as jnp

from jax.lax import stop_gradient
from typing import List


class AssociativeLatent(nn.Module):
    num_latents: int
    num_values_per_latent: List[int]
    _values_per_latent: List[jnp.ndarray]
    beta: float = 100.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.num_values_per_latent, int):
            self.num_values_per_latent = [self.num_values_per_latent] * self.num_latents

        self._values_per_latent = [jnp.linspace(-1.0, 1.0, self.num_values_per_latent[i])
                                   for i in range(self.num_latents)]
        self._values_per_latent = jnp.array(self._values_per_latent)

    @property
    def values_per_latent(self):
        return self._values_per_latent

    def associate(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.expand_dims(x, -1).repeat((1, 1, self.num_values_per_latent.shape[-1]))
        dists = jnp.abs(x - self.values_per_latent[None])
        score = nn.softmax(-dists * self.beta, axis=-1)
        return (score * self.values_per_latent).sum(-1)

    @nn.compact
    def __call__(self, x: jnp.array):
        z_quantized = self.associate(stop_gradient(x))
        z_hat = x + stop_gradient(z_quantized - x)
        outs = {
            'z_continuous': x,
            'z_quantized': z_quantized,
            'z_hat': z_hat
        }
        return outs

