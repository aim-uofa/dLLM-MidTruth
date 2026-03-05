import os
import json
from parsers import Parser, evaluate_equation, validate_equation
from gsm8k import GSM8KDataset
from datasets import load_dataset
import warnings

CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)

CTO_VALIDATION_SIZE = 200

class CTDDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=CTD_SYSTEM_PROMPT,
        subsample=256,
        split="test",
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding num_examples to 0.")
        if split == "validation":
            subsample = CTO_VALIDATION_SIZE
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample,
            split=split,
        )  # num_examples = always 0

    def load_test_dataset(self):
        self.dataset = []
        cur_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_path}/../dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        print(len(self.dataset), "examples loaded")

    def load_validation_dataset(self):
        raw_data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        raw_data = raw_data.filter(lambda x: len(x["nums"]) == 3)
        self.dataset = raw_data.select(range(CTO_VALIDATION_SIZE))

    def __getitem__(self, idx):
        if self.split == "validation":
            target = int(self.dataset[self.subsample[idx].item()]["target"])
            numbers = self.dataset[self.subsample[idx].item()]["nums"]
        else:
            target = int(self.dataset[self.subsample[idx].item()]["output"])
            numbers_str = self.dataset[self.subsample[idx].item()]["input"]
            numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        prompt = self.create_prompt(question)
        return prompt, question, (numbers, target)
