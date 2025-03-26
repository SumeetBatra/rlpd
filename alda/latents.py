import flax.linen as nn
import jax.numpy as jnp

from jax.lax import stop_gradient
from typing import List


class AssociativeLatent(nn.Module):
    num_latents: int
    num_values_per_latent: List[int]
    beta: float = 100.

    def setup(self):
        init_values = [jnp.linspace(-1.0, 1.0, self.num_values_per_latent[i])
                                   for i in range(self.num_latents)]
        init_values = jnp.stack(init_values, axis=0)
        self._values_per_latent = self.param(
            "values_per_latent",
            lambda rng, shape: init_values,
            init_values.shape
        )

    @property
    def values_per_latent(self):
        return self._values_per_latent

    def associate(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.expand_dims(x, -1).repeat(self.values_per_latent.shape[-1], axis=-1)
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

