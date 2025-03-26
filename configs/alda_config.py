from ml_collections.config_dict import config_dict

from configs import rlpd_pixels_config


def get_config():
    config = rlpd_pixels_config.get_config()
    config.model_cls = "ALDALearner"

    config.num_latents = 20
    config.values_per_latent = 12
    config.beta = 100

    return config