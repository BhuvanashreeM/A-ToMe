from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig1:
    # Guiding text prompt
    prompt: str = "a cat wearing sunglasses and a dog wearing hat"
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    # whether use LLM-based parser (Qwen 2.5-14B) instead of SpaCy
    use_llm_parser: bool = False
    # Which token indices to merge
    token_indices: List[int] = field(
        default_factory=lambda: [[[2], [3, 4]], [[7], [8, 9]]]
    )
    # Spilt prompt
    # prompt_anchor: List[str] = field(default_factory=lambda:['Musk with black sunglasses', 'Trump with blue suit'])
    prompt_anchor: List[str] = field(
        default_factory=lambda: [
            "a cat wearing sunglasses",
            "a dog wearing hat",
        ]
    )
    # prompt after token merge
    prompt_merged: str = "a cat and a dog"
    # words of the prompt
    prompt_length: int = 9
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = 0
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


@dataclass
class RunConfig2:
    # Guiding text prompt
    prompt: str = "a white cat and a black dog"
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = True
    # whether use LLM-based parser (Qwen 2.5-14B) instead of SpaCy
    use_llm_parser: bool = False
    # Which token indices to merge
    token_indices: List[int] = field(default_factory=lambda: [[[2], [3]], [[6], [7]]])
    # Spilt prompt
    # prompt_anchor: List[str] = field(default_factory=lambda:['Musk with black sunglasses', 'Trump with blue suit'])
    prompt_anchor: List[str] = field(
        default_factory=lambda: [
            "a white cat",
            "a black dog",
        ]
    )
    # prompt after token merge
    prompt_merged: str = "a cat and a dog"
    # words of the prompt
    prompt_length: int = 7
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [43, 198])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [5, 5])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [4, 4])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = False
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


@dataclass
class RunConfig3:
    """Test config for LLM-based parser (Qwen 2.5-14B)."""
    # Guiding text prompt
    prompt: str = "a white cat and a black dog"
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    use_nlp: bool = True
    use_llm_parser: bool = True  # ENABLE LLM PARSER
    token_indices: List[int] = field(default_factory=lambda: [[[2], [3]], [[6], [7]]])
    prompt_anchor: List[str] = field(default_factory=lambda: ["a white cat", "a black dog"])
    prompt_merged: str = "a cat and a dog"
    prompt_length: int = 7
    seeds: List[int] = field(default_factory=lambda: [43])  # Single seed for testing
    output_path: Path = Path("./demo")
    n_inference_steps: int = 50
    guidance_scale: float = 7.5
    attention_res: int = 32
    run_standard_sd: bool = False
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {0: 26, 1: 25, 2: 24, 3: 23, 4: 22.5, 5: 22, 6: 21.5, 7: 21, 8: 21, 9: 21}
    )
    tome_control_steps: List[int] = field(default_factory=lambda: [5, 5])
    token_refinement_steps: int = 3
    attention_refinement_steps: List[int] = field(default_factory=lambda: [4, 4])
    eot_replace_step: int = 60
    use_pose_loss: bool = False
    scale_factor: int = 3
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
