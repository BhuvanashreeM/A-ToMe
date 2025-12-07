import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from configs.demo_config import RunConfig1, RunConfig2, RunConfig3, RunConfig4, RunConfig5, RunConfig5_SpaCy, RunConfig6
from pipe_tome import tomePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
from prompt_utils import PromptParser
import spacy
import os

# Import new parser system
from parsers import SpaCyParser, LLMParser

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def read_prompt(path):
    with open(path, "r") as f:
        prompt_ls = f.readlines()

    all_prompt = []

    for idx, prompt in enumerate(prompt_ls):
        prompt = prompt.replace("\n", "")
        all_prompt.append([idx, prompt])
    return all_prompt


def load_model(config, device):

    stable_diffusion_version = "stabilityai/stable-diffusion-xl-base-1.0"

    if hasattr(config, "model_path") and config.model_path is not None:
        stable_diffusion_version = config.model_path

    # Use float32 for CPU, float16 for CUDA (MPS also supports float16 but can be unstable)
    if device == "cpu":
        stable = tomePipeline.from_pretrained(
            stable_diffusion_version,
            torch_dtype=torch.float32,
            safety_checker=None,
        ).to(device)
    elif device == "mps":
        # MPS supports float16 but load models without variant
        stable = tomePipeline.from_pretrained(
            stable_diffusion_version,
            torch_dtype=torch.float32,  # Use float32 for better stability on MPS
            safety_checker=None,
        ).to(device)
    else:
        stable = tomePipeline.from_pretrained(
            stable_diffusion_version,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
        ).to(device)
    # stable.enable_xformers_memory_efficient_attention()
    stable.unet.requires_grad_(False)
    stable.vae.requires_grad_(False)
    # stable.enable_model_cpu_offload()

    prompt_parser = PromptParser(stable_diffusion_version)

    return stable, prompt_parser


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {
        idx: stable.tokenizer.decode(t)
        for idx, t in enumerate(stable.tokenizer(prompt)["input_ids"])
        if 0 < idx < len(stable.tokenizer(prompt)["input_ids"]) - 1
    }
    pprint.pprint(token_idx_to_word)
    token_indices = input(
        "Please enter the a comma-separated list indices of the tokens you wish to "
        "alter (e.g., 2,5): "
    )
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(
    prompt: List[str],
    model: tomePipeline,
    controller: AttentionStore,
    token_indices: List[int],
    prompt_anchor: List[str],
    seed: torch.Generator,
    config,
) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(
        prompt=prompt,
        guidance_scale=config.guidance_scale,
        generator=seed,
        num_inference_steps=config.n_inference_steps,
        attention_store=controller,
        indices_to_alter=token_indices,
        prompt_anchor=prompt_anchor,
        attention_res=config.attention_res,
        run_standard_sd=config.run_standard_sd,
        thresholds=config.thresholds,
        scale_factor=config.scale_factor,
        scale_range=config.scale_range,
        prompt3=config.prompt_merged,
        prompt_length=config.prompt_length,
        token_refinement_steps=config.token_refinement_steps,
        attention_refinement_steps=config.attention_refinement_steps,
        tome_control_steps=config.tome_control_steps,
        eot_replace_step=config.eot_replace_step,
        use_pose_loss=config.use_pose_loss,
        negative_prompt="low res, ugly, blurry, artifact, unreal",
    )
    image = outputs.images[0]
    return image


def filter_text(token_indices, prompt_anchor):
    final_idx = []
    final_prompt = []
    for i, idx in enumerate(token_indices):
        if len(idx[1]) == 0:
            continue
        final_idx.append(idx)
        final_prompt.append(prompt_anchor[i])
    return final_idx, final_prompt


def main():
    config = RunConfig5_SpaCy() #edit this to change the config
    config_name = config.__class__.__name__  # Get config class name (e.g., "RunConfig3")
    # Support MPS (Apple Silicon) if available
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    stable, prompt_parser = load_model(config, device)
    # ------------------parser prompt-------------------------
    # NEW: Support for LLM-based parser
    if hasattr(config, 'use_llm_parser') and config.use_llm_parser:
        print("Using LLM-based parser (Qwen 2.5-14B)")
        llm_parser = LLMParser(model_path=config.model_path)
        token_indices, prompt_anchor = llm_parser.parse(config.prompt)
        llm_parser.cleanup()  # Free LLM memory before loading SDXL
    elif config.use_nlp:
        import en_core_web_trf

        nlp = en_core_web_trf.load()  # load spacy

        doc = nlp(config.prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(config.prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)
        token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)
    else:
        token_indices = config.token_indices
        prompt_anchor = config.prompt_anchor
    # ------------------parser prompt-------------------------

    # token_indices = get_indices_to_alter(stable, config.prompt) if config.token_indices is None else config.token_indices

    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        print(f"Original Prompt: {config.prompt}")
        print(f"Anchor Prompt: {prompt_anchor}")
        print(f"Indices of merged tokens: {token_indices}")
        g = torch.Generator(device).manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(
            prompt=config.prompt,
            model=stable,
            controller=controller,
            token_indices=token_indices,
            prompt_anchor=prompt_anchor,
            seed=g,
            config=config,
        )
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)

        # Add config name to filename for easy identification
        method = "standard" if config.run_standard_sd else "tome"
        filename = f'{seed}_{method}_{config_name}.png'

        image.save(prompt_output_path / filename)
        print(f"Saved: {prompt_output_path / filename}")
        images.append(image)

    joined_image = vis_utils.get_image_grid(images)

    # Add config name to grid image filename
    method = "standard" if config.run_standard_sd else "tome"
    grid_filename = f'{config.prompt}_{method}_{config_name}.png'

    joined_image.save(config.output_path / grid_filename)
    print(f"Saved grid: {config.output_path / grid_filename}")


if __name__ == "__main__":
    main()
