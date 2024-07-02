# Copy from diffusers.examples.textual_inversion.textual_inversion.py
#
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import time
import functools
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import dnnlib
import misc
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    CLIPImageProcessor, CLIPVisionModelWithProjection,
    AutoImageProcessor, AutoModel
)

from utils import (
    attn_maps,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    preprocess,
    visualize_and_save_attn_map,
    init_attn_maps_cache
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
from imagenet_vocabulary import ImageNet_vocabulary
# ------------------------------------------------------------------------------

logger = get_logger(__name__)

sub_vocabulary = {
    0: 'person',
    1: 'cat',
    2: 'dog',
    3: 'rabbit',
    4: 'car',
    5: 'plane',
    6: 'tree',
    7: 'flower',
    8: 'building',
    9: 'house',
    10: 'table',
    11: 'chair',
    12: 'food',
    13: 'fruit',
    14: 'vegetable',
    15: 'animal'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--test_rabbit",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--attn_map",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--global_emb",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--class_token",
        type=str,
        default="|class|",
        required=True,
        help="Class token for tokenizer.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default='cls',
        required=False,
        help="Class token for initialize infer.",
    )
    parser.add_argument(
        "--num_embeds",
        type=int,
        default=1,
        required=False,
        help="Number of class token embeddings to optimize.",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=1000,
        required=False,
        help="Number of test images to run.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=False, help="A folder containing the training data."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--do_CFG",
        action="store_true",
        default=False,
        help="Whether or not to use CFG.",
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.train_data_dir is None:
    #     raise ValueError("You must specify a train data directory.")

    return args


def encode_prompt(prompt_batch, text_encoders, tokenizers, get_cls_idx=False):
    prompt_embeds_list = []

    # ViT-L encode
    with torch.no_grad():
        text_inputs = tokenizers[0](
            prompt_batch,
            padding="max_length",
            max_length=tokenizers[0].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoders[0](
            text_input_ids.to(text_encoders[0].device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        # pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds_list.append(prompt_embeds)

    # ViT-bigG encode
    text_inputs = tokenizers[1](
        prompt_batch,
        padding="max_length",
        max_length=tokenizers[1].model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    if get_cls_idx:
        tokens = tokenizers[1].convert_ids_to_tokens(text_input_ids[0])
        for i, token in enumerate(tokens):
            if "<cls>" in token:
                return i
    prompt_embeds = text_encoders[1](
        text_input_ids.to(text_encoders[1].device),
        output_hidden_states=True,
    )

    # We are only ALWAYS interested in the pooled output of the final text encoder
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
def encode_image(image, device, num_images_per_prompt, output_hidden_states=None):
    dtype = next(self.image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = self.feature_extractor(image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_enc_hidden_states = self.image_encoder(
            torch.zeros_like(image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds


@torch.no_grad()
def compute_vocabulary_embedding(cls_prompt, compute_embeddings_fn, cls_idx=None, sub_vocab=False):
    embedding_dict = {}
    vocab = sub_vocabulary if sub_vocab else ImageNet_vocabulary
    for idx, class_phrase in vocab.items():
        prompt = [cls_prompt.replace("<cls>", class_phrase)]
        prompt_embedding, addtional_conditions = compute_embeddings_fn(prompt)
        if cls_idx:
            emb_l, emb_bigG = torch.split(prompt_embedding, [768, 1280], dim=-1)
            embedding = emb_bigG[0][cls_idx]
        else:
            embedding = addtional_conditions["text_embeds"][0]
        embedding_dict[idx] = embedding
    return embedding_dict


@torch.no_grad()
def do_classification(cls_prompt, compute_embeddings_fn, embedding_dict, cls_idx=None, sub_vocab=False):
    min_dis = 1.0
    pred_cls = -1
    cls_scores = []
    prompt_embedding, addtional_conditions = compute_embeddings_fn([cls_prompt])
    vocab = sub_vocabulary if sub_vocab else ImageNet_vocabulary

    # Get embedding
    if cls_idx:
        emb_l, emb_bigG = torch.split(prompt_embedding, [768, 1280], dim=-1)
        embedding = emb_bigG[0][cls_idx]
    else:
        embedding = addtional_conditions["text_embeds"][0]

    # Calculate distances
    for idx, class_phrase in vocab.items():
        dis = (embedding - embedding_dict[idx]).pow(2).mean().sqrt()
        cls_scores.append(dis.item())
        # print(class_phrase, ": ", cls_scores[idx])
    if sub_vocab:
        return
    else:
        indexed_list = list(enumerate(cls_scores))
        sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
        sorted_elements = [element for index, element in sorted_indexed_list]
        # print(cls_scores)
        original_indices = [index for index, element in sorted_indexed_list]
        # print(original_indices)
        cls_scores_var = np.var(cls_scores)
        return original_indices[0], cls_scores_var


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def main():

    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # Text encoder.
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_2 = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant
    )        

    # Insert tokens to be optimized into tokenizer.
    cls_tokens = [args.class_token]

    # if args.num_embeds < 1:
    #     raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")
    # # add dummy tokens for multi-vector
    # additional_tokens = []
    # for i in range(1, args.num_embeds):
    #     additional_tokens.append(f"{args.class_token}_{i}")
    # cls_tokens += additional_tokens

    num_added_tokens_2 = tokenizer_2.add_tokens(cls_tokens)
    if num_added_tokens_2 != args.num_embeds:
        raise ValueError(
            f"The tokenizer already contains the token {args.class_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    # Directly retrive initializer token as it is in the vocabulary.
    cls_token_ids = tokenizer_2.convert_tokens_to_ids(cls_tokens)
    token_ids = tokenizer_2.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")
    initializer_token_id = token_ids[0]

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))

    # Initialise the newly added class token with the embeddings of the initializer token
    token_embeds = text_encoder_2.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in cls_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    prompt = f"a photo of {args.class_token}"
    with torch.no_grad():
        cls_idx = encode_prompt(prompt, [text_encoder, text_encoder_2], [tokenizer, tokenizer_2], get_cls_idx=True)

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder_2.text_model.encoder.requires_grad_(False)
    text_encoder_2.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_2.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder_2.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=args.train_data_dir, use_labels=True)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    if not dataset_obj.has_labels:
        raise ValueError('--cond=True, but no labels found in the dataset')
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
        seed=args.seed, start_idx=0
    )
    data_loader_kwargs = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=1, **data_loader_kwargs))

    @torch.no_grad()
    def read_image(image_dir):
        image_pt = transform(Image.open(image_dir).convert("RGB"))
        return image_pt

    def compute_embeddings(prompts, text_encoders, tokenizers):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompts, text_encoders, tokenizers
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompts), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return prompt_embeds, unet_added_cond_kwargs

    @torch.no_grad()
    def convert_to_latent(pixels):
        model_input = vae.encode(pixels).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
        if args.pretrained_vae_model_name_or_path is None:
            model_input = model_input.to(weight_dtype)
        return model_input

    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=[text_encoder, text_encoder_2],
        tokenizers=[tokenizer, tokenizer_2],
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    # Prepare everything with our `accelerator`.
    text_encoder_2, optimizer, lr_scheduler = accelerator.prepare(
        text_encoder_2, optimizer, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # GO!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running *****")
    logger.info(f"  Num examples = {len(dataset_obj)}")
    # logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    # logger.info(f"  Total batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    test_num = min(args.test_num, len(dataset_obj))
    num_correct = 0
    # progress_bar = tqdm(
    #     range(0, len(test_num)),
    #     initial=0,
    #     desc="Steps",
    #     # Only show the progress bar once on each machine.
    #     disable=not accelerator.is_local_main_process,
    # )

    # Cache original embedding for reference.
    if args.global_emb:
        embedding_dict = compute_vocabulary_embedding(prompt, compute_embeddings_fn, sub_vocab=args.test_rabbit)
        do_classification(prompt, compute_embeddings_fn, embedding_dict, sub_vocab=args.test_rabbit)
    else:
        embedding_dict = compute_vocabulary_embedding(prompt, compute_embeddings_fn, cls_idx=cls_idx, sub_vocab=args.test_rabbit)
        do_classification(prompt, compute_embeddings_fn, embedding_dict, cls_idx=cls_idx, sub_vocab=args.test_rabbit)
    orig_embeds_params = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight.data.clone()

    for i, (image, label) in enumerate(dataset_iterator):
        if i == test_num:
            return
        numpy_array = image.numpy()
        numpy_array = numpy_array.squeeze(0).astype(np.uint8)
        numpy_array = np.transpose(numpy_array, (1, 2, 0))
        pil_image = Image.fromarray(numpy_array)
        pil_image.save('imagenet_image.png')
        # print(f"class phrase: {ImageNet_vocabulary[np.argmax(label).item()]}")
        image = (image.to(dtype=torch.float32) - 127.5) / 127.5
        if args.test_rabbit:
            # Test with JourneyDB rabbit.
            image = read_image("examples/0.jpg").unsqueeze(0)
        image = image.to(accelerator.device, dtype=vae.dtype)

        # Move inputs to latent space.
        latent = convert_to_latent(image)
        if args.pretrained_vae_model_name_or_path is None:
            latent = latent.to(weight_dtype)
        bsz = latent.shape[0]

        # 4. Prepare timesteps
        timesteps = reversed(noise_scheduler.timesteps)
        curve = []
        timesteps = [100] * 300

        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        tic = time.time()
        for id_t, t in enumerate(timesteps):

            if id_t == len(timesteps) - 1 and args.attn_map:
                cross_attn_init()
                unet = set_layer_with_name_and_path(unet)
                unet = register_cross_attention_hook(unet)
            # Sample noise that we'll add to the latents.
            noise = torch.randn_like(latent)
            ts = torch.tensor([t]*bsz, dtype=torch.int64, device=accelerator.device)
            noisy_model_input = noise_scheduler.add_noise(latent, noise, ts)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([noisy_model_input] * 2) if args.do_CFG else noisy_model_input

            prompt_embeds, add_conds = compute_embeddings_fn([prompt])

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,                # [B, 77, 2048]
                added_cond_kwargs=add_conds,                # {[B, 1280], [B, 6]}
                return_dict=False,
            )[0]

            # perform guidance
            if args.do_CFG:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latent, noise, ts)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            accelerator.backward(loss, retain_graph=True)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer_2),), dtype=torch.bool)
            index_no_updates[min(cls_token_ids) : max(cls_token_ids) + 1] = False
            with torch.no_grad():
                accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]

                # Loss curve
                if args.test_rabbit:
                    test_prompt_embedding, test_addtional_conditions = compute_embeddings_fn(
                        [prompt, prompt.replace("<cls>", 'rabbit')]
                    )
                    if args.global_emb:
                        test_error = (test_addtional_conditions["text_embeds"][0] - test_addtional_conditions["text_embeds"][1]).pow(2).mean().sqrt()
                    else:
                        emb_L, emb_bigG = torch.split(test_prompt_embedding, [768, 1280], dim=-1)
                        test_error = (emb_bigG[0][cls_idx] - emb_bigG[1][cls_idx]).pow(2).mean().sqrt()
                    curve.append(test_error.item())
                    # if id_t == 0 and args.attn_map:
                    #     attn_map = preprocess(max_height=args.resolution, max_width=args.resolution,)
                    #     visualize_and_save_attn_map(attn_map, tokenizer_2, [prompt], postfix=f"before")

            # compute the previous noisy sample x_t -> x_t-1
            # latents_dtype = latents.dtype
            # latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            # if latents.dtype != latents_dtype:
            #     if torch.backends.mps.is_available():
            #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            #         latents = latents.to(latents_dtype)

        toc = time.time()
        # print("Time: ", toc-tic)

        if args.test_rabbit:
            print(curve)
            if args.global_emb:
                do_classification(prompt, compute_embeddings_fn, embedding_dict, sub_vocab=args.test_rabbit)
            else:
                do_classification(prompt, compute_embeddings_fn, embedding_dict, cls_idx=cls_idx, sub_vocab=args.test_rabbit)
            if args.attn_map:
                # Extract attention map
                attn_map = preprocess(max_height=args.resolution, max_width=args.resolution,)
                visualize_and_save_attn_map(attn_map, tokenizer_2, [prompt], postfix=f"after")
            return
        else:
            if args.global_emb:
                pred_cls, score_var = do_classification(prompt, compute_embeddings_fn, embedding_dict, sub_vocab=args.test_rabbit)
            else:
                pred_cls, score_var = do_classification(prompt, compute_embeddings_fn, embedding_dict, cls_idx=cls_idx, sub_vocab=args.test_rabbit)
            if pred_cls == np.argmax(label).item():
                num_correct += 1
            # print(f"Pred cls: {pred_cls} | {ImageNet_vocabulary[pred_cls]}")
            # print(f"Ground truth: {np.argmax(label).item()} | {ImageNet_vocabulary[np.argmax(label).item()]}")
            # print(f"acc: {num_correct}/{i+1}", score_var)
            print(score_var, end=", ")


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main()