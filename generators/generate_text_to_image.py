import jax
import jax.numpy as jnp
from jax.lib import xla_bridge

print('Device: ', xla_bridge.get_backend().platform)

from functools import partial

import numpy as np
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard, shard_prng_key
from PIL import Image
from tqdm.notebook import trange
from transformers import CLIPProcessor, FlaxCLIPModel
from vqgan_jax.modeling_flax_vqgan import VQModel

from generators.constants import DalleConstants


class ImageGenerator:
    """This class provides functions to generate an image based on the given text prompt using dalle model."""
    def __init__(self, prompt: str):
        self.prompt = prompt

    def tokenize(self):
        processor = DalleBartProcessor.from_pretrained(DalleConstants.GithubParams.DALLE_MODEL,
                                                       revision=DalleConstants.GithubParams.DALLE_COMMIT_ID)
        tokenized_prompts = processor([self.prompt])
        # Replicate it onto each device for faster inference.
        return replicate(tokenized_prompts)

    def generate_images(self, num_predictions: int):
        # Load models & tokenizer
        # Load dalle-mini
        model, params = DalleBart.from_pretrained(
            DalleConstants.GithubParams.DALLE_MODEL,
            revision=DalleConstants.GithubParams.DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )

        # Load VQGAN
        vqgan, vqgan_params = VQModel.from_pretrained(
            DalleConstants.GithubParams.VQGAN_REPO,
            revision=DalleConstants.GithubParams.VQGAN_COMMIT_ID, _do_init=False
        )

        # Model parameters are replicated on each device for faster inference.
        params = replicate(params)
        vqgan_params = replicate(vqgan_params)

        tokenized_prompt = self.tokenize()

        print(f"Prompt: {self.prompt}\n")
        # generate images
        images = []
        for _ in trange(max(num_predictions // jax.device_count(), 1)):
            # get a new key
            key, subkey = jax.random.split(DalleConstants.Random.KEY)
            # generate images
            encoded_images = self.p_generate(
                tokenized_prompt,
                model,
                params,
                shard_prng_key(subkey)
            )

            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = self.p_decode(encoded_images, vqgan_params, vqgan)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
        return images

    def reranked_order(self, image_list):
        # Load CLIP model
        clip, clip_params = FlaxCLIPModel.from_pretrained(
            DalleConstants.ClipParams.CLIP_REPO, revision=DalleConstants.ClipParams.CLIP_COMMIT_ID,
            dtype=jnp.float16, _do_init=False
        )
        clip_processor = CLIPProcessor.from_pretrained(DalleConstants.ClipParams.CLIP_REPO,
                                                       revision=DalleConstants.ClipParams.CLIP_COMMIT_ID)
        clip_params = replicate(clip_params)

        # get clip scores
        clip_inputs = clip_processor(
            text=[self.prompt] * jax.device_count(),
            images=image_list,
            return_tensors="np",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).data
        logits = self.p_clip(shard(clip_inputs), clip_params, clip)
        logits = logits.squeeze().flatten()
        sorted_logits = logits.argsort()[::-1]
        return sorted_logits

    def get_top_1(self, num_predictions):
        generated_images = self.generate_images(num_predictions)
        reranked_order = self.reranked_order(generated_images)
        # return top 1 image
        generated_images[reranked_order[0]].save("./tmp.jpg")
        return generated_images[reranked_order[0]]

    # model inference
    @staticmethod
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(tokenized_prompt, model, params, key):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=DalleConstants.GeneratorParams.GEN_TOP_K,
            top_p=DalleConstants.GeneratorParams.GEN_TOP_P,
            temperature=DalleConstants.GeneratorParams.TEMPERATURE,
            condition_scale=DalleConstants.GeneratorParams.COND_SCALE,
        )

    # decode image
    @staticmethod
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params, vqgan_model):
        return vqgan_model.decode_code(indices, params=params)

    # score images
    @staticmethod
    @partial(jax.pmap, axis_name="batch")
    def p_clip(inputs, params, clip_model):
        logits = clip_model(params=params, **inputs).logits_per_image
        return logits
