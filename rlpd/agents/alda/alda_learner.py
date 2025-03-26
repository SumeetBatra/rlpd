import gym
import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from jax.lax import stop_gradient

from rlpd.agents.drq.augmentations import batched_random_crop
from rlpd.agents.sac.sac_learner import SACLearner
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.agents.agent import Agent
from rlpd.networks import MLP, Ensemble, PixelMultiplexer, StateActionValue, subsample_ensemble
from rlpd.networks.encoders import D4PGEncoder
from alda import *

from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple, Dict


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, encoder, latent_model, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, encoder, latent_model, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, encoder, latent_model, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, encoder, latent_model, observations)
    return dist.mode()


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
        if "Encoder" in k:
            replacers[k] = v

    # Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)


def bce_w_logits(x, y):
    """
    Binary Cross Entropy Loss
    Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751

    :param x: Input tensor
    :param y: Target tensor
    :param weight: Vector of example weights
    :param average: Boolean to average resulting loss vector
    :return: Scalar value
    """
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))
    return loss


class ALDALearner(SACLearner):
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    encoder: TrainState
    decoder: TrainState
    latent_model: TrainState
    target_latent_model: TrainState

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
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
        transition_shape: Tuple[int] = (4, 4, 256),
        **kwargs,
    ):

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, encoder_key, decoder_key, latent_key, temp_key = jax.random.split(rng, 7)

        encoder_def = QuantizedEncoder(
            num_latents=num_latents,
            mlp_hidden_layers=hidden_dims
        )
        encoder_params = encoder_def.init(encoder_key, observations)["params"]
        encoder_params = FrozenDict(encoder_params)
        quantized_encoder = TrainState.create(
            apply_fn=encoder_def.apply,
            params=encoder_params,
            tx=optax.adamw(learning_rate=1e-3, weight_decay=0.1)
        )

        history_encoder_cls = partial(
            HistoryEncoder,
            num_latents=num_latents,
            emb_dim=hidden_dims[-1]
        )

        latent_model_def = AssociativeLatent(
            num_latents=num_latents,
            num_values_per_latent=[values_per_latent] * num_latents
        )
        sample_embeds = jax.random.normal(rng, (1, num_latents))
        latent_model_params = latent_model_def.init(latent_key, sample_embeds)["params"]
        latent_model_params = FrozenDict(latent_model_params)
        latent_model = TrainState.create(
            apply_fn=latent_model_def.apply,
            params=latent_model_params,
            tx=optax.adam(learning_rate=1e-3)
        )

        target_latent_model = TrainState.create(
            apply_fn=latent_model_def.apply,
            params=latent_model_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = ALDAEncoder(
            history_encoder_cls=history_encoder_cls,
            network_cls=actor_cls,
            latent_dim=latent_dim,
            stop_gradient=True,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )

        actor_params = actor_def.init(actor_key, quantized_encoder, latent_model, observations)["params"]
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
            history_encoder_cls=history_encoder_cls,
            network_cls=critic_cls,
            latent_dim=latent_dim,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )
        critic_params = critic_def.init(critic_key, quantized_encoder, latent_model, observations, actions)["params"]
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
        decoder_params = FrozenDict(decoder_params)
        decoder = TrainState.create(
            apply_fn=decoder_def.apply,
            params=decoder_params,
            tx=optax.adamw(learning_rate=1e-3, weight_decay=0.1)
        )

        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(key, observations, pixel_key)
                if depth_key is not None:
                    observations = batched_random_crop(key, observations, depth_key)
            return observations

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            encoder=quantized_encoder,
            decoder=decoder,
            latent_model=latent_model,
            target_latent_model=target_latent_model,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            data_augmentation_fn=data_augmentation_fn,
        )

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, self.encoder, self.latent_model, observations)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, self.encoder, self.latent_model, observations
        )
        return np.asarray(actions), self.replace(rng=new_rng)

    def update_alda(self, obs: FrozenDict):
        # use the batch of latest frames only for training
        rgb = obs["pixels"][..., (-1,)]
        rgb_target = rgb.astype(jnp.float32) / 255.
        obs = obs.copy(add_or_replace={'pixels': rgb, 'rgb_target': rgb_target})

        def encoder_loss(encoder_params, latent_params, decoder_params):
            pre_z = self.encoder.apply_fn({"params": encoder_params}, obs).squeeze()
            outs_latent = self.latent_model.apply_fn({"params": latent_params}, pre_z)
            x_hat_logits = self.decoder.apply_fn({"params": decoder_params}, outs_latent['z_quantized'])
            outs = {'pre_z': pre_z, **outs_latent}

            quant_loss = jnp.mean(jnp.square(stop_gradient(outs['z_continuous']) - outs['z_quantized']), axis=1)
            commit_loss = jnp.mean(jnp.square(stop_gradient(outs['z_continuous']) - stop_gradient(outs['z_quantized'])), axis=1)
            bce_loss = bce_w_logits(x_hat_logits, obs['rgb_target'].squeeze()).mean(axis=(1, 2, 3))

            total_loss = jnp.mean(0.01 * quant_loss + 0.01 * commit_loss + 1.0 * bce_loss)
            return total_loss, {'quant_loss': quant_loss, 'commit_loss': commit_loss, **outs}

        grads, info = jax.grad(encoder_loss, argnums=(0, 1, 2), has_aux=True)(self.encoder.params, self.latent_model.params, self.decoder.params)
        encoder_grads, latent_grads, decoder_grads = grads
        encoder = self.encoder.apply_gradients(grads=encoder_grads)
        latent_model = self.latent_model.apply_gradients(grads=latent_grads)
        decoder = self.decoder.apply_gradients(grads=decoder_grads)

        target_latent_model_params = optax.incremental_update(
            latent_model.params, self.target_latent_model.params, self.tau
        )
        target_latent_model = self.target_latent_model.replace(params=target_latent_model_params)

        return self.replace(encoder=encoder, decoder=decoder, latent_model=latent_model, target_latent_model=target_latent_model), info


    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, self.encoder, self.latent_model, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                self.encoder,
                self.latent_model,
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, self.encoder, self.latent_model, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            self.encoder,
            self.latent_model,
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params, encoder_params, latent_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            embeds = self.encoder.apply_fn({"params": encoder_params}, batch["observations"])
            embeds = self.latent_model.apply_fn({"params": latent_params}, embeds)['z_hat']
            batch_size = batch["observations"]["pixels"].shape[0]
            embeds = self.critic.apply_fn({"params": critic_params}, embeds, batch_size,
                                          method=ALDAEncoder.compute_temporal_embeddings)
            qs = self.critic.apply_fn({"params": critic_params}, embeds, batch["actions"], True,
                                      method=ALDAEncoder.compute_output)
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(self.critic.params, self.encoder.params, self.latent_model.params)
        critic_grads, encoder_grads = grads
        critic = self.critic.apply_gradients(grads=critic_grads)
        encoder = self.encoder.apply_gradients(grads=encoder_grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng, encoder=encoder), info


    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent = self

        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)

        actor = _share_encoder(source=new_agent.critic, target=new_agent.actor)
        new_agent = new_agent.replace(actor=actor)

        obs = batch['observations']
        next_obs = batch['next_observations']
        rng, key = jax.random.split(new_agent.rng)
        obs_aug = self.data_augmentation_fn(key, obs)
        rng, key = jax.random.split(rng)
        next_obs_aug = self.data_augmentation_fn(key, next_obs)
        new_agent = new_agent.replace(rng=rng)
        batch = batch.copy(
            add_or_replace={
                "observations": obs_aug,
                "next_observations": next_obs_aug,
                "obs_orig": obs
            }
        )

        for i in range(utd_ratio):
            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)

            new_agent, alda_info = new_agent.update_alda(mini_batch["obs_orig"])
            new_agent, critic_info = new_agent.update_critic(mini_batch)
            new_agent, actor_info = new_agent.update_actor(mini_batch)
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

            return new_agent, {**actor_info, **critic_info, **temp_info, **alda_info}
