from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from reward_func import extract_hash_answer

import random
import numpy as np
import torch
import os


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""


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


SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train", prompt_type=0) -> Dataset:

    data = load_dataset("../dataset/openai/gsm8k", "main")[split]
    if prompt_type == 0:
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )

    elif prompt_type == 1:
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "user", "content": GSM_SYSTEM_PROMPT + "\n\n" + x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )

    else:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Must be 0 or 1 for GSM8K questions."
        )


def get_countdown_questions(split="train", prompt_type=0) -> Dataset:
    data = load_dataset("../dataset/Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    if prompt_type == 0:
        return data.map(
            lambda x: {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                    },
                ],
                "target": x["target"],
                "numbers": x["nums"],
            }
        )

    elif prompt_type == 1:
        return data.map(
            lambda x: {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{CTD_SYSTEM_PROMPT}\n\nNumbers: {', '.join(map(str, x['nums']))}\nTarget: {x['target']}\n",
                    },
                ],
                "target": x["target"],
                "numbers": x["nums"],
            }
        )

    else:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Must be 0 or 1 for Countdown questions."
        )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train", prompt_type=0) -> Dataset:
    data = load_dataset("../dataset/ankner/math-500", split=split)  # type: ignore

    if prompt_type == 0:
        data = data.map(
            lambda x: {  # type: ignore
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                    },
                ],
                "answer": x["solution"],
            }
        )
        return data

    elif prompt_type == 1:
        data = data.map(
            lambda x: {  # type: ignore
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{MATH500_SYSTEM_PROMPT}\n\n{x['problem']}",
                    },
                ],
                "answer": x["solution"],
            }
        )  # type: ignore
        return data  # type: ignore

    else:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Must be 0 or 1 for Math500 questions."
        )


def get_svamp_questions(split="train") -> Dataset:
    data = load_dataset("../dataset/ChilleD/SVAMP", split=split)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{GSM_SYSTEM_PROMPT}\n\n{x['Body']} {x['Question']}",
                },
            ],
            "answer": x["Answer"],
        }
    )


def get_combined_questions(split="train", prompt_type=0) -> Dataset:
    gsm8k_data = get_gsm8k_questions(split, prompt_type).map(
        lambda x: {"dataset": "gsm8k"}
    )
    countdown_data = get_countdown_questions(split, prompt_type).map(
        lambda x: {"dataset": "countdown"}
    )
    math_data = get_math_questions(split, prompt_type).map(
        lambda x: {"dataset": "math500"}
    )
    svamp_data = get_svamp_questions(split).map(
        lambda x: {"dataset": "svamp"}
    )

    dataset_sizes = [
        len(gsm8k_data),
        len(countdown_data),
        len(math_data),
        len(svamp_data)
    ]
    min_size = min(dataset_sizes)
    # print(f"Minimum dataset size: {min_size}")

    gsm8k_data = gsm8k_data.select(range(min_size))
    countdown_data = countdown_data.select(range(min_size))
    math_data = math_data.select(range(min_size))
    svamp_data = svamp_data.select(range(min_size))

    combined = concatenate_datasets([gsm8k_data, countdown_data, math_data, svamp_data])
    combined = combined.shuffle(seed=42)

    return combined


ARC_SYSTEM_PROMPT = """Given the following question and four candidate answers (A, B, C and D), choose the best answer."""

def get_arc_questions(split="train") -> Dataset:
    data = load_dataset("../dataset/allenai/ai2_arc", 'ARC-Challenge', split=split)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{ARC_SYSTEM_PROMPT}\n"
                        f"Question:{x['question']}\n"
                        + "\n".join([
                            f"{label}. {text}"
                            for label, text in zip(x["choices"]["label"], x["choices"]["text"])
                        ])
                        + '\nYour response should end with "The best answer is [the_answer_letter]" '
                        'where the [the_answer_letter] is one of A, B, C or D.'
                    )
                }
            ],
            "answer": x["answerKey"],
        }
    )

WINO_SYSTEM_PROMPT = """Given the following question and four candidate answers (A and B), choose the best answer."""

def get_wino_questions(split="train") -> Dataset:
    data = load_dataset("../dataset/allenai/winogrande", "winogrande_l", split=split)

    def format_content(x):
        sentence = x["sentence"]
        content = f"{WINO_SYSTEM_PROMPT}\n" 
        content += f"Question: {sentence}"
        content += f"\nA. {x['option1']}" + f"\nB. {x['option2']}"
        content += '\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A or B.'

        answer = "A" if x["answer"] == 1 else "B"

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "answer": answer,
        }

    return data.map(
        format_content
    )

HELLA_SYSTEM_PROMPT = """You are an expert multiple-choice question answerer. Given a context and four possible endings, you must choose the ending that makes the most sense based on the context. 
Provide your answer as one of the letters 'A', 'B', 'C', or 'D' corresponding to the chosen ending."""

def get_hellaswag_questions(split="train") -> Dataset:
    data = load_dataset("../dataset/Rowan/hellaswag", split=split)

    def format_content(x):
        ctx = x["ctx"]
        endings = x["endings"]
        content = f"{HELLA_SYSTEM_PROMPT}\n" 
        content += f"{ctx}\nQuestion: Which ending makes the most sense?\n"
        content += f"A. {endings[0]}\n" + f"B. {endings[1]}\n" + f"C. {endings[2]}\n" + f"D. {endings[3]}\n"
        content += "You may choose from 'A', 'B', 'C', 'D'. Answer: "

        numToAnswer = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
        answer = numToAnswer[x["label"]]

        return {
            "prompt": [
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "answer": answer,
        }

    return data.map(
        format_content,
        remove_columns=data.column_names 
    )
