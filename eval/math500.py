import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
from generate import generate
import random
import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, is_equiv
from parser_helper import last_boxed_only_string, remove_boxed

MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""

MATH_VALIDATION_SIZE = 250


class MATH500Dataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MATH500_SYSTEM_PROMPT,
        subsample=-1,
        split="test",
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample, split)

    def load_test_dataset(self):
        self.dataset = load_dataset("../dataset/HuggingFaceH4/MATH-500", split="test")
    
    def load_validation_dataset(self):
        data = load_dataset("../dataset/ankner/math-500", split="train") 
        self.dataset = data.select(range(MATH_VALIDATION_SIZE))

    def load_few_shot_examples(self):
        train_data = load_dataset("../dataset/EleutherAI/hendrycks_math", ("algebra"), split="train")
        few_shot_examples = []
        samples = random.sample(range(len(train_data)), self.num_examples)
        for example in samples:
            few_shot_examples.append(
                {"question": train_data[example]["problem"], "answer": train_data[example]["solution"]}
            )
        return few_shot_examples
    
    def __getitem__(self, idx):
        if self.split == "validation":
            question = self.dataset[self.subsample[idx].item()]["problem"]
            solution = self.dataset[self.subsample[idx].item()]["solution"]
            answer = remove_boxed(last_boxed_only_string(solution))
        else:
            question = self.dataset[self.subsample[idx].item()]["problem"]
            answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
