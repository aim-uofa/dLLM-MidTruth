import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
import random
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb
import os, json
from packaging import version
import transformers

from parse_and_calculate import parse_answer, calculate_entropy


if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def selective_log_softmax_coupled(logits, index, weights=None, mask=None):
    """
    Memory-efficient implementation for coupled masking with three versions of probabilities.
    
    Args:
        logits: [num_iterations * 3 * batch_size, seq_len, vocab_size]
        index: [num_iterations * batch_size, seq_len] 
        weights: [num_iterations * 3] weights for different versions
        mask: [num_iterations * batch_size, seq_len] completion mask
    
    Returns:
        [num_iterations * batch_size, seq_len] log probabilities
    """
    full_batch_size = logits.size(0) // 3
    num_iterations = weights.size(0) // 3
    batch_size = full_batch_size // num_iterations
    per_token_logps = []
    
    for i in range(full_batch_size):
        seq_labels = index[i]
        
        chunk, offset = divmod(i, batch_size)
        base = chunk * 3 * batch_size
        logits_index = torch.tensor([base + 0*batch_size + offset, base + 1*batch_size + offset, base + 2*batch_size + offset], device=logits.device)
        
        seq_logits = logits[logits_index]  # [3, seq_len, vocab_size]
        seq_logps = F.log_softmax(seq_logits, dim=-1)
        seq_per_token_logps = seq_logps.gather(dim=-1, index=seq_labels.unsqueeze(0).unsqueeze(-1).expand(3, -1, 1)).squeeze(-1)
        
        if weights is not None and mask is not None:
            weight_idx = i // batch_size
            seq_weights = weights[weight_idx*3:(weight_idx+1)*3]
            seq_mask = mask[i]
            
            weighted_logps = torch.where(
                seq_mask,
                seq_per_token_logps[1] * seq_weights[1],
                seq_per_token_logps[2] * seq_weights[2]
            )
            
            final_logps = (seq_per_token_logps[0] + weighted_logps) / 2
        else:
            final_logps = seq_per_token_logps[0]
            
        per_token_logps.append(final_logps)
    
    return torch.stack(per_token_logps)


class DiffuGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
        # ours
        temporal_reward_weight: bool = False,
        temporal_reward_type: str = "fixed",  # "fixed", "linear", "exp"
        combine_method: str = "brier",  # "brier", "logarithmic", "spherical"
        dataset_name: Optional[str] = None,
        use_coupled_masking: bool = False,
        use_token_sampling: bool = False,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.temporal_reward_weight = temporal_reward_weight
        self.temporal_reward_type = temporal_reward_type
        self.combine_method = combine_method
        if self.temporal_reward_weight > 0.0:
            assert dataset_name is not None, "Dataset name must be provided when temporal_reward_weight is set"
        self.dataset_name = dataset_name
        self.use_coupled_masking = use_coupled_masking
        self.use_token_sampling = use_token_sampling

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        # # inputs
        # {
        #     "prompt_ids": prompt_ids,  # (bs, prompt_length)
        #     "prompt_mask": prompt_mask,  # (bs, prompt_length)
        #     "completion_ids": completion_ids,  # (bs, 256)
        #     "completion_mask": completion_mask,  # (bs, 256)
        #     "old_per_token_logps": all_old_per_token_logps,  # (self.num_iterations, bs, logits_to_keep)
        #     "ref_per_token_logps": all_ref_per_token_logps,  # (self.num_iterations, bs, logits_to_keep)
        #     "advantages": advantages,  # (bs, )
        #     "mask_seeds": mask_seeds,  # (self.num_iterations, )  # Store all mask seeds for consistent mask patterns
        # }

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)
        per_token_logps = self._get_per_token_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed])
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def generate(
        self,
        model,
        prompt,
        tokenizer,
        inputs,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            answers = [[] for _ in range(prompt.shape[0])]
            answer_entropy = [[] for _ in range(prompt.shape[0])]

            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            del logits_with_noise

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            if self.temporal_reward_weight > 0.0:
                                raw_generations = tokenizer.batch_decode(
                                    x0[:, prompt.shape[1]:],
                                    skip_special_tokens=True,
                                )

                                for j, gen in enumerate(raw_generations):
                                    # answer = parse_answer(gen, self.dataset_name)
                                    if self.dataset_name == "combined":
                                        dataset = inputs[j].get("dataset", "unknown")
                                        answer = parse_answer(gen, dataset)
                                    else:
                                        answer, answer_index = parse_answer(gen, self.dataset_name)

                                    # print(f"Raw generation {j}: {repr(gen)}")
                                    # print(f"Answer {j}: {answer}")
                                    answers[j].append(answer)
                                    if answer == None:
                                        continue 
                                    start = 0
                                    end = 0
                                    current_length = 0
                                    for curr in range(prompt.shape[1], len(x0[j])):
                                        token_text = tokenizer.decode(x0[j, curr], skip_special_tokens=False)
                                        current_length += len(token_text)
                                        if current_length >= answer_index + 1 and start == 0:
                                            start = curr
                                        if current_length >= answer_index + len(answer):
                                            end = curr
                                            break
                                    answer_logits = logits[j, start:end+1]
                                    answer_probs = torch.softmax(answer_logits, dim=-1)
                                    entropy = -torch.sum(answer_probs * torch.log2(answer_probs + 1e-12), dim=-1)
                                    mean_entropy = entropy.mean().item()
                                    answer_entropy[j].append(mean_entropy)


                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            del x0, confidence, transfer_index

            # return x
            answer_entropies = None
            if self.temporal_reward_weight > 0.0:
                answer_entropies = calculate_entropy(
                    answers,
                    include_none=self.args.entropy_include_none,
                    skip_steps_ratio=self.args.entropy_skip_steps_ratio,
                    entropy_exp_alpha=self.args.entropy_exp_alpha,
                )

                # pairwise agreement
                for idx, answer_row in enumerate(answers):
                    change_count = 0
                    # current_answer = None
                    current_answer = answer_row[0]
                    for ans in answer_row:
                        if ans != current_answer:
                            change_count += 1
                            current_answer = ans
                    answer_entropies[idx] = answer_entropies[idx] + (change_count / steps,)

                # token-level entropy
                for idx, mean_entropy in enumerate(answer_entropy):
                    entropy_max = np.log2(tokenizer.vocab_size)
                    avg_token_entropy = sum(mean_entropy) / len(mean_entropy) / entropy_max if len(mean_entropy) > 0 else 1.0
                    answer_entropies[idx] = answer_entropies[idx] + (avg_token_entropy,)


                print(f"Generated {len(answers)} answers.")
                for i, (answer_row, entropies_row) in enumerate(zip(answers, answer_entropies)):
                    print(f"Answer {i}: {answer_row}")
                    print(f"Entropy: {entropies_row}")

            return x, answers, answer_entropies


    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        if self.use_coupled_masking:
            return self._forward_process_coupled(batch, prompt_index, mask_id, seed)
        else:
            return self._forward_process_original(batch, prompt_index, mask_id, seed)
    
    def _forward_process_original(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        # self.args.p_mask_prompt = 0.15
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        if self.use_token_sampling:
            is_mask_prompt = torch.zeros((b, l), dtype=torch.bool, device=batch.device)
            for i in range(b):
                n_prompt = prompt_index.sum().item()

                # caculate number of tokens to mask
                n_mask = np.ceil(t_p[i].item() * n_prompt)
                n_mask = int(n_mask)

                # print(f"Batch {i}: Masking {n_mask} out of {n_prompt} prompt tokens")
                # randomly select n_mask prompt tokens to mask
                selected_indices = torch.randperm(n_prompt, device=batch.device)[:n_mask]

                # set mask positions
                is_mask_prompt[i, selected_indices] = True
        else:
            is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))

        is_mask_completion = ~prompt_index  # all completion tokens are masked
        # is_mask: True for p_mask_prompt random prompts, and for all completion tokens
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask

    def _forward_process_coupled(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        noisy_batch = []
        
        # Generate random mask ratio between 0.2 and 0.8
        mask_ratio = random.uniform(0.2, 0.8)
        t_p = torch.ones(b, device=batch.device) * mask_ratio
        
        # Create random matrix
        random_matrix = torch.rand((b, l), device=batch.device)
        
        # 1. Always mask completion tokens (baseline)
        is_mask = ~prompt_index
        noisy_batch.append(torch.where(is_mask, mask_id, batch))
        
        # 2. Mask completion tokens with probability t_p
        if self.use_token_sampling:
            is_mask = torch.zeros((b, l), dtype=torch.bool, device=batch.device)
            for i in range(b):
                completion_index = ~prompt_index
                n_comp = completion_index.sum().item()
                n_prompt = prompt_index.sum().item()
                n_mask = np.ceil(t_p[i].item() * n_comp)
                n_mask = int(n_mask)
                # print(f"Batch {i}: Masking {n_mask} out of {n_comp} completion tokens with prompt length {n_prompt}")
                selected = torch.randperm(n_comp, device=batch.device)[:n_mask]
                masked_positions = selected + n_prompt
                is_mask[i, masked_positions] = True
        else:
            is_mask = ~prompt_index & (random_matrix < t_p.unsqueeze(1))
        completion_mask = is_mask
        noisy_batch.append(torch.where(is_mask, mask_id, batch))
        
        # 3. Mask completion tokens reversely
        if self.use_token_sampling:
            is_mask = (~prompt_index) & (~is_mask)
        else:
            is_mask = ~prompt_index & (random_matrix > t_p.unsqueeze(1))
        noisy_batch.append(torch.where(is_mask, mask_id, batch))
        
        return noisy_batch, [1, 1/mask_ratio, 1/(1-mask_ratio)], completion_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Calculate per-token log probabilities.
        """
        if self.use_coupled_masking:
            return self._get_per_token_logps_coupled(model, input_ids, logits_to_keep, mask_seeds)
        else:
            return self._get_per_token_logps_original(model, input_ids, logits_to_keep, mask_seeds)
    
    def _get_per_token_logps_original(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Original single masking method.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, _ = self.forward_process(
                expanded_input, prompt_index, self.args.mask_id, seed=mask_seed
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps
    
    def _get_per_token_logps_coupled(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Coupled masking method with three versions.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Ensure logits_to_keep is valid
        logits_to_keep = min(logits_to_keep, seq_len)
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length
        if len(mask_seeds) != num_iterations:
            raise ValueError(f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}")

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_weighted = []
        all_expanded_inputs = []
        all_completion_masks = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, t_weights, completion_mask = self.forward_process(
                expanded_input, prompt_index, self.args.mask_id, seed=mask_seed
            )
            all_perturbed_seqs.extend(perturbed_seq)
            all_weighted.extend(t_weights) # [num_iterations * 3] list
            all_expanded_inputs.append(expanded_input)
            all_completion_masks.append(completion_mask)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * 3 * batch_size, seq_len]
        completion_mask_seq = torch.cat(all_completion_masks, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]
        all_weights_t = torch.tensor(all_weighted, device=device) # [num_iterations * 3]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iterations * 3 * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * 3 * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        completion_loss_mask = completion_mask_seq[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        
        # Compute log probabilities using selective_log_softmax
        per_token_logps = selective_log_softmax_coupled(
            completion_logits, 
            completion_targets, 
            all_weights_t, 
            completion_loss_mask
        ).view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        # self.num_iterations = 12
        print(f"global_step: {self.state.global_step}, step: {self._step}")
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        # (bs, seq_len)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:  # 200
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length  # 256
        block_length = self.args.block_length  # 32
        steps = self.args.diffusion_steps  # 128
        temperature = self.args.temperature or 0.0  # 0.9
        cfg_scale = self.args.cfg_scale  # 0.0

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size  # 6
            prompt_completion_ids_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]

                # answers[i][j] represents the answer generated at the j-th step for the i-th input. None indicates failure to parse an answer.
                # entropies[i] is a tuple containing the weighted and unweighted entropy values for the i-th input respectively.
                batch_prompt_completion_ids, answers, entropies = self.generate(
                    model=unwrapped_model,
                    tokenizer=self.processing_class,
                    inputs=inputs,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,  # "low_confidence"
                    mask_id=self.args.mask_id,
                )
                prompt_completion_ids_all.append(batch_prompt_completion_ids)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        if self.args.random_masking:  # True
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.num_iterations > 1:  # 12
                # repeat prompt completion ids self.num_iterations times
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )  # (self.num_iterations, bs, seq_len)
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                )  # (self.num_iterations, bs, logits_to_keep)
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            # self.beta = 0.04
            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                    )  # (self.num_iterations, bs, logits_to_keep)
                    all_ref_per_token_logps = ref_per_token_logps

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                if reward_func_name == "temporal_semantic_entropy_reward":
                    output_reward_func = reward_func(
                        entropy=entropies,
                        temporal_reward_type=self.temporal_reward_type,
                    )
                elif reward_func_name == "temporal_semantic_entropy_with_ground_truth_reward":
                    output_reward_func = reward_func(
                        entropy=entropies,
                        prompts=prompts,
                        completions=completions,
                        steps=steps,
                        dataset_name=self.dataset_name,
                        temporal_reward_type=self.temporal_reward_type,
                        combine_method=self.combine_method,
                        **reward_kwargs,
                    )
                else:
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        step=self._step,
                        run_name=self.args.output_dir,
                        **reward_kwargs,
                    )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,  # (bs, prompt_length)
            "prompt_mask": prompt_mask,  # (bs, prompt_length)
            "completion_ids": completion_ids,  # (bs, 256)
            "completion_mask": completion_mask,  # (bs, 256)
            "old_per_token_logps": all_old_per_token_logps,  # (self.num_iterations, bs, logits_to_keep)
            "ref_per_token_logps": all_ref_per_token_logps,  # (self.num_iterations, bs, logits_to_keep)
            "advantages": advantages,  # (bs, )
            "mask_seeds": mask_seeds,  # (self.num_iterations, )  # Store all mask seeds for consistent mask patterns
        }

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # self.log is called in self._inner_training_loop and self.evaluate

        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)

        # save self.state.log_history to self.args.output_dir
        if self.args.output_dir is not None:
            log_path = os.path.join(self.args.output_dir, "log_history.json")
            with open(log_path, "w") as f:
                json.dump(self.state.log_history, f, indent=4)

        self._metrics[mode].clear()
