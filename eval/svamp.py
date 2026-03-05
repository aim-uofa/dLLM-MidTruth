
from datasets import load_dataset
from gsm8k import GSM8KDataset

SVAMP_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

SVAMP_VALIDATION_SIZE = 150

class SVAMPDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=SVAMP_SYSTEM_PROMPT,
        subsample=-1,
        split="test",
    ):
        if num_examples > 0:
            raise ValueError("num_examples must be 0 for SVAMP dataset. Overriding num_examples to 0.")
        super().__init__(
            tokenizer,
            0,  
            add_reasoning,
            system_prompt,
            subsample,
            split, 
        )
    
    def load_test_dataset(self):
        self.dataset = load_dataset("../dataset/ChilleD/SVAMP", "default", split="test")

    def load_validation_dataset(self):
        data = load_dataset("../dataset/ChilleD/SVAMP", "default", split="train")
        self.dataset = data.select(range(SVAMP_VALIDATION_SIZE))

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["Body"] + " " + self.dataset[self.subsample[idx].item()]["Question"]
        target = int(self.dataset[self.subsample[idx].item()]["Answer"])
        prompt = self.create_prompt(question)
        return prompt, question, target