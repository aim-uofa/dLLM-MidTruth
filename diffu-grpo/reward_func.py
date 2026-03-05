import numpy as np
import re
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv, boxed_in_answer
from parse_and_calculate import parse_answer


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    extracted_responses = [parse_answer(r, "gsm8k")[0] for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )

    print("✅" if extracted_responses[0] == float(answer[0].replace(",", "")) else "❌")
    return [2.0 if r == float(a.replace(",", "")) else 0.0 for r, a in zip(extracted_responses, answer)]


def arc_correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [parse_answer(r, "arc-c")[0] for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )

    print("✅" if extracted_responses[0] == answer[0] else "❌")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def hellaswag_correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [parse_answer(r, "hellaswag")[0] for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )

    print("✅" if extracted_responses[0] == answer[0] else "❌")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    # equation = extract_solution(solution_str)
    equation = parse_answer(solution_str, "countdown")[0]
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name=None, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        # solution = extract_answer_sudoku(response)
        solution = parse_answer(response, "sudoku")[0]

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = parse_answer(r, "math")[0]
        except:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


def temporal_semantic_entropy_reward(
    # prompts, completions, answer, step=None, run_name=None, **kwargs
    entropy,
    temporal_reward_type="fixed",
    # entropy_max_thr=2.0,
    **kwargs,
) -> list[float]:
    """
    Reward function that computes the temporal semantic entropy of the responses.
    This is a placeholder and should be replaced with actual implementation.
    """

    if temporal_reward_type == "exp":
        entropy_list = [ent[0] for ent in entropy]
    elif temporal_reward_type == "linear":
        entropy_list = [ent[1] for ent in entropy]
    elif temporal_reward_type == "fixed":
        entropy_list = [ent[2] for ent in entropy]
    elif temporal_reward_type == "pairwise":
        entropy_list = [ent[3] for ent in entropy]
    elif temporal_reward_type == "token_entropy":
        entropy_list = [ent[4] for ent in entropy]
    else:
        raise ValueError(f"Unknown temporal reward type: {temporal_reward_type}")


    # entropy_list = [max(0, entropy_max_thr - ent) for ent in entropy_list]

    # simply reverse the entropy values
    entropy_list = [ent * -1 for ent in entropy_list]

    return entropy_list


def temporal_semantic_entropy_with_ground_truth_reward(
    entropy,
    prompts,
    completions,
    answer=0,
    steps=128, # diffusion steps
    dataset_name="gsm8k", 
    temporal_reward_type="fixed",
    combine_method="brier",
    **kwargs
) -> list[float]:
    """
    Reward function that computes the temporal semantic entropy of the responses
    and combines it with the ground truth answer.
    """
    if temporal_reward_type == "exp":
        entropy_list = [ent[0] for ent in entropy]
    elif temporal_reward_type == "linear":
        entropy_list = [ent[1] for ent in entropy]
    elif temporal_reward_type == "fixed":
        entropy_list = [ent[2] for ent in entropy]
    elif temporal_reward_type == "pairwise":
        entropy_list = [ent[3] for ent in entropy]
    elif temporal_reward_type == "token_entropy":
        entropy_list = [ent[4] for ent in entropy]
    else:
        raise ValueError(f"Unknown temporal reward type: {temporal_reward_type}")
    
    correctness_reward = 0.0
    if "combined" in dataset_name:
        correctness_reward = combined_correctness_reward_func(prompts, completions, answer, **kwargs)
    elif "gsm" in dataset_name or "svamp" in dataset_name:
        correctness_reward = correctness_reward_func(prompts, completions, answer) # 2.0 if correct, 0.0 if incorrect
        # change to 1.0 if correct, 0.0 if incorrect
        correctness_reward = [1.0 if r == 2.0 else 0.0 for r in correctness_reward]
    elif "math" in dataset_name:
        correctness_reward = correctness_reward_func_math(prompts, completions, answer) # 2.0 if correct, 0.0 if incorrect
        # change to 1.0 if correct, 0.0 if incorrect
        correctness_reward = [1.0 if r == 2.0 else 0.0 for r in correctness_reward]
    elif "countdown" in dataset_name:
        correctness_reward = countdown_reward_func(prompts, completions, **kwargs) # 1.0 if correct, 0.1 if is an equation, 0.0 if incorrect
        # change to 1.0 if correct, 0.0 if incorrect
        correctness_reward = [1.0 if r == 1.0 else 0.0 for r in correctness_reward]
    elif "sudoku" in dataset_name:
        correctness_reward = sudoku_reward_func(prompts, completions, **kwargs) # correct_cells / empty_cells
    elif "arc-c" in dataset_name or "wino" in dataset_name:
        correctness_reward = arc_correctness_reward_func(prompts, completions, answer) # 2.0 if correct, 0.0 if incorrect
        # change to 1.0 if correct, 0.0 if incorrect
        correctness_reward = [1.0 if r == 2.0 else 0.0 for r in correctness_reward]
    elif "hellaswag" in dataset_name:
        correctness_reward = hellaswag_correctness_reward_func(prompts, completions, answer) # 2.0 if correct, 0.0 if incorrect
        # change to 1.0 if correct, 0.0 if incorrect
        correctness_reward = [1.0 if r == 2.0 else 0.0 for r in correctness_reward]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # change the entropy values to be [0.0, 1.0] where 0.0 is the upperbound of entropy
    if temporal_reward_type in ["exp", "linear", "fixed"]:
        entropy_max = np.log2(steps)
        entropy_list = [1 - ent / entropy_max for ent in entropy_list]
    elif temporal_reward_type == "pairwise" or temporal_reward_type == "token_entropy":
        entropy_list = [1 - ent for ent in entropy_list]

    # similar to brier score (consider tse as another confidence score)
    # could instead use Logarithmic score or Spherical score
    # reference: http://arxiv.org/abs/2507.16806
    if combine_method == "fixed":
        if "sudoku" in dataset_name:
            rewards = [ent if corr >= 0.25 else 0.0 for corr, ent in zip(correctness_reward, entropy_list)]
        else:
            rewards = [ent if corr == 1.0 else 0.0 for corr, ent in zip(correctness_reward, entropy_list)]
    elif combine_method == "brier":
        rewards = [
            corr - (ent - corr) ** 2 for corr, ent in zip(correctness_reward, entropy_list)
        ]
    elif combine_method == "logarithmic":
        rewards = [
            corr - (corr * np.log2(ent + 1e-8) + (1 - corr) * np.log2(1 - ent + 1e-8)) for corr, ent in zip(correctness_reward, entropy_list)
        ]
    elif combine_method == "spherical":
        rewards = [
            corr - ent / (np.sqrt((1 - ent) ** 2 + ent ** 2) + 1e-8) for corr, ent in zip(correctness_reward, entropy_list)
        ]
    elif combine_method == "logarithmic_plus":
        rewards = [
            corr + (corr * np.log2(ent + 1e-8) + (1 - corr) * np.log2(1 - ent + 1e-8)) for corr, ent in zip(correctness_reward, entropy_list)
        ]
    elif combine_method == "spherical_plus":
        rewards = [
            corr + ent / (np.sqrt((1 - ent) ** 2 + ent ** 2) + 1e-8) for corr, ent in zip(correctness_reward, entropy_list)
        ]
    else:
        raise ValueError(f"Unknown combine method: {combine_method}")


    rewards = correctness_reward

    print("Correctness rewards:", correctness_reward)
    print("Entropy rewards:", entropy_list)
    print("Combined rewards:", rewards)

    return rewards


def combined_correctness_reward_func(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    dataset_names = kwargs.get("dataset", [])
    rewards = []
    for i, dataset_name in enumerate(dataset_names):
        prompt = [prompts[i]]
        completion = [completions[i]]
        ans = [answer[i]]
        print(f"Dataset: {dataset_name}")
        if "gsm" in dataset_name or "svamp" in dataset_name:
            r = correctness_reward_func(prompt, completion, ans, step=step, run_name=run_name)[0] # 2.0 if correct, 0.0 if incorrect
            r = 1.0 if r == 2.0 else 0.0
        elif "math" in dataset_name:
            r = correctness_reward_func_math(prompt, completion, ans, step=step, run_name=run_name)[0] # 2.0 if correct, 0.0 if incorrect
            # unify the scale to be 1.0
            r = 1.0 if r == 2.0 else 0.0
        elif "countdown" in dataset_name:
            r = countdown_reward_func(prompt, completion, **kwargs)[0] # 1.0 if correct, 0.1 if is an equation, 0.0 if incorrect
            r = 1.0 if r == 1.0 else 0.0
        elif "sudoku" in dataset_name:
            r = sudoku_reward_func(prompts, completions, **kwargs)[0]  # correct_cells / empty_cells
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        rewards.append(r)

    return rewards