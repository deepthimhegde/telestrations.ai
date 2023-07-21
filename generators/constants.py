import random

import jax


class DalleConstants:

    class GithubParams:
        # dalle Model references

        # dalle-mega
        DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
        DALLE_COMMIT_ID = None

        # VQGAN model
        VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
        VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    class GeneratorParams:
        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        GEN_TOP_K = None
        GEN_TOP_P = None
        TEMPERATURE = None
        COND_SCALE = 10.0

    class ClipParams:
        CLIP_REPO = "openai/clip-vit-base-patch32"
        CLIP_COMMIT_ID = None

    class Random:
        # create a random key
        SEED = random.randint(0, 2**32 - 1)
        KEY = jax.random.PRNGKey(SEED)
