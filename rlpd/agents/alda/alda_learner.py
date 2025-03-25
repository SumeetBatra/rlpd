import gym
import jax
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from rlpd.agents.drq.augmentations import batched_random_crop
from rlpd.agents.sac.sac_learner import SACLearner
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import MLP, Ensemble, PixelMultiplexer, StateActionValue
from rlpd.networks.encoders import D4PGEncoder
from alda import *

from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple


# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][..., :-1]
            next_obs_pixels = batch["observations"][pixel_key][..., 1:]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )

    batch = batch.copy(
        add_or_replace={"observations": obs, "next_observations": next_obs}
    )

    return batch


def _share_encoder(source, target):
    replacers = {}

    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v

    # Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)


class ALDALearner(SACLearner):
    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        latent_dim: int = 50,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        num_latents: int = 10,
        values_per_latent: int = 12,
        beta: int = 100,
        transition_shape: Tuple[int] = (256, 4, 4)
    ):

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, decoder_key, temp_key = jax.random.split(rng, 5)

        encoder_cls = partial(
            QuantizedEncoder,
            num_latents=num_latents,
            mlp_hidden_layers=hidden_dims
        )

        history_encoder_cls = partial(
            HistoryEncoder,
            num_latents=num_latents,
            emb_dim=hidden_dims[-1]
        )

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = ALDAEncoder(
            encoder_cls=encoder_cls,
            history_encoder_cls=history_encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            stop_gradient=True,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )

        actor_params = actor_def.init(actor_key, observations)["params"]
        actor_params = FrozenDict(actor_params)
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = ALDAEncoder(
            encoder_cls=encoder_cls,
            history_encoder_cls=history_encoder_cls,
            network_cls=critic_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_params = FrozenDict(critic_params)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp_params = FrozenDict(temp_params)
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        # ALDA
        decoder_def = QuantizedDecoder(
            transition_shape=transition_shape
        )
        sample_latent = jax.random.normal(rng, (1, num_latents))
        decoder_params = decoder_def.init(decoder_key, sample_latent)["params"]
        decoder_params = FrozenDict(critic_params)

        ae_params = {'encoder': actor_params, 'decoder': decoder_params}