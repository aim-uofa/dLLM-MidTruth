



# <img src="assets/logo.png" width="8%" alt="" align=center /> Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-green)](https://arxiv.org/abs/2508.09138)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://aim-uofa.github.io/dLLM-MidTruth/)
[![Code](https://img.shields.io/badge/Code-GitHub-orange)](#)
[![License](https://img.shields.io/badge/License-BSD%202--clause-lightgrey)](https://opensource.org/license/bsd-2-clause)

## 📣 News

- [2026-03-05] Training code of Temporal Consistency Reinforcement Released!
- [2026-01-27] [dLLM-MidTruth](https://arxiv.org/abs/2508.09138) is accepted by ICLR 2026.
- [2025-08-21] Code of Temporal Self-Consistency Voting and Evaluation Released!
- [2025-08-13] Paper Released!


## 🚀 Overview

<div align="center" >
<img src="assets/demo.png"/>
</div>

## 📖 Description

Diffusion large language models (dLLMs) generate text through iterative denoising, yet current decoding strategies discard rich intermediate predictions in favor of the final output. Our work here reveals a critical phenomenon, <b>temporal oscillation</b>, where correct answers often emerge in the middle process, but are overwritten in later denoising steps.
To address this issue, we introduce two complementary methods that exploit temporal consistency: 
- **Temporal Self-Consistency Voting**, a training-free, test-time decoding strategy that aggregates predictions across denoising steps to select the most consistent output; 
- **Temporal Consistency Reinforcement**, a post-training method that uses Temporal Semantic Entropy (TSE), a measure of semantic stability across intermediate predictions, as a reward signal to encourage stable generations. 


## ⚙️ Getting Started

### Environment Setup

To setup the environment, run:
```bash
conda env create -f env.yml
conda activate tiaf
```

Then install the dependencies:
```bash
bash install.sh
```

This will install required packages and set up the output directories. You also need to:
1. Download the base model [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) to `pretrained_weights/GSAI-ML/LLaDA-8B-Instruct`
2. Download the required datasets (they will be loaded from `dataset/` directory):
   - `openai/gsm8k`, `Jiayi-Pan/Countdown-Tasks-3to4`, `ankner/math-500`, `ChilleD/SVAMP`, `simplescaling/s1K`

### Temporal Majority Voting

We provide a simple plug-and-play generate script to perform temporal self-consistency voting in `eval/generate.py`.

You can enable the voting mechanism by passing the `--enable_vote` flag together with both `--tokenizer` and `--parse_answer_func`.
Optionally, you may also specify `--vote_method` (fixed, linear, or exp) and, if using the exp method, control the weighting strength with `--alpha`.



### Temporal Consistency Reinforcement (Training)

We provide training scripts for three stages:

#### Stage 1: SFT (Supervised Fine-Tuning)

Fine-tune the base dLLM model on the [s1K dataset](https://huggingface.co/datasets/simplescaling/s1K) with LoRA adapters:

```bash
bash scripts/train_sft.sh
```

Key hyperparameters (configurable in the script):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LR` | `1e-5` | Learning rate |
| `NUM_EPOCHS` | `20` | Number of training epochs |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM_STEPS` | `4` | Gradient accumulation steps |
| `NUM_GPUS` | `8` | Number of GPUs |

#### Stage 2: RFT with Temporal Reward Only

Train with GRPO using only the Temporal Semantic Entropy (TSE) as reward signal — **no ground-truth labels needed**:

```bash
bash scripts/train_rft_temporal.sh
```

Key hyperparameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET` | `countdown` | Training dataset (`countdown`, `gsm8k`, `math`, `svamp`, `sudoku`) |
| `SFT_PATH` | `outputs/sft/...` | Path to SFT checkpoint (from Stage 1) |
| `TEMPORAL_REWARD_TYPE` | `exp` | TSE weighting: `fixed`, `linear`, or `exp` |
| `TEMPORAL_REWARD_WEIGHT` | `2.0` | Weight for the temporal reward |
| `ENTROPY_SKIP_STEPS_RATIO` | `0.5` | Skip first 50% denoising steps for entropy |
| `ENTROPY_EXP_ALPHA` | `12.8` | Exponential decay factor for time weighting |
| `NUM_ITERATIONS` | `12` | GRPO inner update iterations (μ) |
| `NUM_GENERATIONS` | `6` | Group size for GRPO |
| `DIFFUSION_STEPS` | `128` | Number of diffusion denoising steps |

#### Stage 2 (Alternative): RFT with Temporal + Accuracy Reward

Train with GRPO using both TSE and ground-truth accuracy, combined via a proper scoring rule:

```bash
bash scripts/train_rft_temporal_acc.sh
```

Additional hyperparameters (on top of the temporal-only setting):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEMPORAL_GT_COMBINE_METHOD` | `spherical_plus` | Scoring rule: `brier`, `logarithmic`, `spherical`, `logarithmic_plus`, `spherical_plus` |
| `TEMPORAL_REWARD_WITH_GT` | `1` | Enable combined temporal + accuracy reward |

### Evaluation

You can easily evaluate our method by running the following command (using GSM8K with a generation length of 128 as an example):

```bash
cd eval

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node 4 \
    --master_port 29173 \
    eval.py \
    --dataset gsm8k \
    --batch_size 8 \
    --gen_length 128 \
    --diffusion_steps 64 \
    --output_dir "outputs/LLaDA-8B-Instruct/gsm8k_gen_128_steps_64_temp_0_vote" \
    --model_path <your_model_path> \
    --enable_vote \
    --vote_method exp \
    --alpha 5.0 
```

For convenience, we also provide a wrapper script `eval/run_eval.sh` that allows easier evaluation management.
You can modify the parameters in this script to test different datasets, and voting strategies.

To compute accuracy, run `get_acc.py` after the evaluation is done.

## 🚩 Plan
- [x] source code of temporal self-consistency voting and evaluation
- [x] source code of temporal consistency reinforcement (SFT + RFT training)

## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 🖊️ Citation

If you find this work useful, please consider citing:

```bibtex 
@article{wang2025temporaldynamics,
    title={Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models},
    author={Wen, Wang and Bozhen, Fang and Chenchen, Jing and Yongliang, Shen and Yangyi, Shen and Qiuyu, Wang and Hao, Ouyang and Hao, Chen and Chunhua, Shen},
    journal={arXiv preprint arXiv:2508.09138},
    year={2025}
}
```
