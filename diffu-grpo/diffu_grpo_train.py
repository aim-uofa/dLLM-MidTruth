# set wandb offline mode
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "outputs/grpo"

import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig, PeftModel, get_peft_model

# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
    temporal_semantic_entropy_reward,
    temporal_semantic_entropy_with_ground_truth_reward,
    arc_correctness_reward_func,
    hellaswag_correctness_reward_func,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
    get_svamp_questions,
    get_arc_questions,
    get_wino_questions, 
    get_hellaswag_questions,
)


def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train", prompt_type=grpo_config.prompt_type)
        reward_functions = [
            correctness_reward_func,
        ]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from GSM8K dataset")
            reward_functions = []
        if grpo_config.with_format_reward == 1:
            reward_functions.extend([
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
            ])
    elif grpo_config.dataset == "combined":
        dataset = get_combined_questions("train", prompt_type=grpo_config.prompt_type)
        reward_functions = [
            combined_correctness_reward_func,
        ]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from Combined dataset")
            reward_functions = []
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train", prompt_type=grpo_config.prompt_type)
        reward_functions = [countdown_reward_func]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from Countdown dataset")
            reward_functions = []
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from Sudoku dataset")
            reward_functions = []
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train", prompt_type=grpo_config.prompt_type)
        reward_functions = [
            correctness_reward_func_math,
        ]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from Math dataset")
            reward_functions = []
        if grpo_config.with_format_reward == 1:
            reward_functions.extend([
                boxed_and_answer_tags_format_reward,
            ])
    elif grpo_config.dataset == "svamp":
        dataset = get_svamp_questions("train")
        reward_functions = [
            correctness_reward_func,
        ]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from SVAMP dataset")
            reward_functions = []
        if grpo_config.with_format_reward == 1:
            reward_functions.extend([
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
            ])
    elif grpo_config.dataset == "arc-c" :
        dataset = get_arc_questions("train")
        reward_functions = [
            arc_correctness_reward_func,
        ]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from ARC/wino dataset")
            reward_functions = []
    elif grpo_config.dataset == "wino":
        dataset = get_wino_questions("train")
        reward_functions = [
            arc_correctness_reward_func,
        ]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from ARC/wino dataset")
            reward_functions = []
    elif grpo_config.dataset == "hellaswag":
        dataset = get_hellaswag_questions("train")
        reward_functions = [
            hellaswag_correctness_reward_func,
        ]
        if grpo_config.remove_gt_reward == 1:
            print("Removing ground truth reward from HellaSwag dataset")
            reward_functions = []

    if grpo_config.reward_weights is None:
        grpo_config.reward_weights = [1.0] * len(reward_functions)
    else:
        assert len(reward_functions) == len(grpo_config.reward_weights)

    if grpo_config.temporal_reward_only:
        reward_functions = []
        grpo_config.reward_weights = []

    if grpo_config.temporal_reward_weight > 0.0:
        reward_functions.append(temporal_semantic_entropy_reward)
        grpo_config.reward_weights.append(grpo_config.temporal_reward_weight)

    if grpo_config.temporal_reward_with_gt:
        # remove all rewards that name contains "correctness" or "temporal" or "countdown"
        reward_functions = [rf for rf in reward_functions if not any(
            name in rf.__name__ for name in ["correctness", "temporal", "countdown"]
        )]
        grpo_config.reward_weights = [1.0] * len(reward_functions)
        reward_functions.append(temporal_semantic_entropy_with_ground_truth_reward)
        grpo_config.reward_weights.append(grpo_config.temporal_reward_weight)

    print("reward_functions:", reward_functions)
    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # First load the base model
    base_model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    print(grpo_config.pretrained_sft_lora_path)
    if grpo_config.pretrained_sft_lora_path is not None:
        print(f"Loading pretrained SFT LoRA weights from {grpo_config.pretrained_sft_lora_path}")
        # Load the sft LoRA weights
        model = PeftModel.from_pretrained(
            base_model,
            grpo_config.pretrained_sft_lora_path,
            torch_dtype=torch.bfloat16,
        )

        # Merge LoRA weights with base model
        model = model.merge_and_unload()
        model = model.to(device)
    else:
        print("No pretrained SFT LoRA weights provided, using base model directly.")
        model = base_model


    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        # ours
        temporal_reward_weight=grpo_config.temporal_reward_weight,
        temporal_reward_type=grpo_config.temporal_reward_type,
        combine_method=grpo_config.temporal_gt_combine_method,
        dataset_name=grpo_config.dataset,
        use_coupled_masking=grpo_config.use_coupled_masking,
        use_token_sampling=grpo_config.use_token_sampling,
    )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()

    # check configurations
    assert grpo_config.temporal_reward_only in [0, 1], f"temporal_reward_only must be 0 or 1, got {grpo_config.temporal_reward_only}"
    if grpo_config.temporal_reward_only:
        assert grpo_config.temporal_reward_weight > 0, "When temporal_reward_only is True, temporal_reward_weight must be greater than 0"
        assert grpo_config.with_format_reward == 0, "When temporal_reward_only is True, with_format_reward must be 0"
    assert grpo_config.dataset in ["gsm8k", "countdown", "sudoku", "math", "svamp", "arc-c", "wino", "hellaswag"], f"Unsupported dataset: {grpo_config.dataset}"
    assert grpo_config.prompt_type in [0, 1], f"Unsupported prompt type: {grpo_config.prompt_type}"
    assert grpo_config.use_coupled_masking in [0, 1], f"use_coupled_masking must be 0 or 1, got {grpo_config.use_coupled_masking}"
    assert grpo_config.with_format_reward in [0, 1], f"with_format_reward must be 0 or 1, got {grpo_config.with_format_reward}"
    assert grpo_config.remove_gt_reward in [0, 1], f"remove_gt_reward must be 0 or 1, got {grpo_config.remove_gt_reward}"
    assert grpo_config.temporal_reward_type in ["fixed", "linear", "exp", "pairwise", "token_entropy"], f"Unsupported temporal reward type: {grpo_config.temporal_reward_type}"
    assert grpo_config.temporal_reward_weight >= 0, f"Temporal reward weight must be non-negative, got {grpo_config.temporal_reward_weight}"
    assert grpo_config.entropy_include_none in [0, 1], f"entropy_include_none must be 0 or 1, got {grpo_config.entropy_include_none}"
    assert grpo_config.temporal_reward_with_gt in [0, 1], f"temporal_reward_with_gt must be 0 or 1, got {grpo_config.temporal_reward_with_gt}"
    assert grpo_config.temporal_gt_combine_method in ["fixed", "brier", "logarithmic", "spherical", "logarithmic_plus", "spherical_plus"], f"Unsupported temporal_gt_combine_method: {grpo_config.temporal_gt_combine_method}"
    assert grpo_config.use_token_sampling in [0, 1], f"use_token_sampling must be 0 or 1, got {grpo_config.use_token_sampling}"

    main(grpo_config=grpo_config, model_config=model_config)
