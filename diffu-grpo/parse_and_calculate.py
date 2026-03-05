from collections import defaultdict
import re
import numpy as np
# from reward_func import extract_xml_answer


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:
        return s
    
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def parse_gsm_answer(raw_generation):
    parsed_answer = None
    parsed_answer_text = None
    answer_position = -1
    boxed_matches_iter = re.finditer(r"\\boxed{(.*?)}", raw_generation)
    for match in boxed_matches_iter:
        boxed_content = match.group(1)
        # 不能直接 strip，会导致索引错误
        stripped_content = boxed_content.strip()
        # 计算 boxed_content 前面的空格数量
        num_stripped = len(boxed_content) - len(boxed_content.lstrip())
        if boxed_content and stripped_content != "..." and not re.match(r"^\.+$", stripped_content):
            try:
                parsed_answer = float(stripped_content)
                parsed_answer_text = stripped_content
                answer_position = match.start(1) + num_stripped
                break
            except ValueError:
                number_match = re.search(r"-?\d+\.?\d*", boxed_content)
                if number_match:
                    try:
                        parsed_answer = float(number_match.group())
                        parsed_answer_text = number_match.group()
                        answer_position = match.start(1) + number_match.start()
                        break
                    except ValueError:
                        pass

    if parsed_answer is None:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1)
            answer_text_start_pos = answer_match.start(1)
            if answer_text:
                try:
                    # 不能直接 strip，会导致索引错误
                    parsed_answer = float(answer_text.strip())
                    parsed_answer_text = answer_text
                    answer_position = answer_text_start_pos
                except ValueError:
                    numbers_iter = list(re.finditer(r"-?\d+\.?\d*", answer_text))
                    if numbers_iter:
                        last_number_match = numbers_iter[-1] 
                        try:
                            parsed_answer = float(last_number_match.group(0))
                            parsed_answer_text = last_number_match.group(0)
                            answer_position = answer_text_start_pos + last_number_match.start()
                        except ValueError:
                            pass
    return parsed_answer_text, answer_position

def parse_countdown_answer(raw_generation):
    equation = None
    answer_position = -1
    try:
        raw_equation = remove_boxed(last_boxed_only_string(raw_generation))
        answer_position = raw_generation.rfind(raw_equation)
    except:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            raw_equation = answer_match.group(1).strip()
            answer_position = answer_match.start(1)
        else:
            raw_equation = raw_generation

    # 转换 Latex 符号
    equation = raw_equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()
        answer_position = equation_match.start(1) + answer_position

    # 检查是否为合格的答案
    if not re.match(r"^[0-9+\-*/() ]+$", equation):
        return None, -1
    return equation, answer_position

def parse_math_answer(raw_generation):
    parsed_answer = None
    answer_position = -1
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        answer_position = raw_generation.rfind(parsed_answer)
    except:
        parsed_answer = None

    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()
            answer_position = answer_match.start(1)
    
    if parsed_answer == raw_generation or last_boxed_only_string(raw_generation) == raw_generation:
        return None, -1

    return parsed_answer, answer_position

def parse_sudoku_answer(raw_generation):
    solution_str = None
    answer_position = -1
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]

    for pattern in patterns:
        if solution_str:
            break
        match = re.search(pattern, raw_generation, re.DOTALL)
        if match and match.group(1).strip():
            solution_str = match.group(1).strip()
            answer_position = match.start(1)
    if solution_str is not None:
        solution_str = re.sub(r"\s", "", solution_str)
        answer_position = raw_generation.rfind(solution_str)

    return solution_str, answer_position

def parse_arc_c_answer(raw_generation):
    match = re.search(r"\b[Tt]he\s+best\s+answer\s+is\s+([A-Z])\b", raw_generation)
    if match:
        answer = match.group(1)
        index = match.start(1)   # 答案字母在原字符串中的位置
        return answer, index

    return None, -1

def parse_hellaswag_answer(raw_generation):
    if not raw_generation or not isinstance(raw_generation, str):
        return None, -1
    
    match = re.search(r"(?<![A-Za-z0-9])([ABCD])(?![A-Za-z0-9])", raw_generation)


    if match:
        answer = match.group(1)
        index = match.start(1)
        return answer, index

    return None, -1

def parse_answer(text, dataset):
    if "gsm" in dataset:
        return parse_gsm_answer(text)
    elif "countdown" in dataset:
        return parse_countdown_answer(text)
    elif "math" in dataset:
        return parse_math_answer(text)
    elif "sudoku" in dataset:
        return parse_sudoku_answer(text)
    elif "svamp" in dataset:
        return parse_gsm_answer(text)
    elif "arc-c" in dataset or "wino" in dataset:
        return parse_arc_c_answer(text)
    elif "hellaswag" in dataset:
        return parse_hellaswag_answer(text)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported for answer parsing.")


# def calculate_entropy(answers):
#     answer_entropies = []
#     for answer_row in answers:
#         answer_weights = defaultdict(float)
#         answer_counts = defaultdict(int)
#         for step, answer in enumerate(answer_row):
#             if answer is None:
#                 continue 
#             answer_counts[answer] += 1
#             answer_weights[answer] += np.exp(step * 0.1)

#         answer_probs = np.array([v / sum(answer_weights.values()) for v in answer_weights.values()])
#         answer_entropy = -np.sum(answer_probs * np.log2(answer_probs)) 
#         answer_cnt_probs = np.array([v / sum(answer_counts.values()) for v in answer_counts.values()])
#         answer_cnt_entropy = -np.sum(answer_cnt_probs * np.log2(answer_cnt_probs))
#         answer_entropies.append((answer_entropy, answer_cnt_entropy))


def calculate_entropy(answers, include_none=False, skip_steps_ratio=0, entropy_exp_alpha=12.8):
    """
    Calculate entropy values for answers, with support for handling None values and skipping initial steps
    
    Args:
        answers (list): 2D list where each sublist contains a series of answers
        include_none (bool): Treat each None as a unique answer
        skip_steps_ratio (float): Ratio of initial steps to skip (0.0 to 1.0)
    
    Returns:
        list: Each element is a tuple (time-weighted entropy, count-based entropy)
    """
    answer_entropies = []
    
    for answer_row in answers:
        # Calculate start index based on skip ratio
        skip_steps = max(0, int(len(answer_row) * skip_steps_ratio))
        processed_answers = answer_row[skip_steps:]

        answer_weights_exp = defaultdict(float)
        answer_weights_linear = defaultdict(float)
        answer_counts = defaultdict(int)
        
        for step, answer in enumerate(processed_answers):
            adjusted_step = step + skip_steps  # Maintain original step index for weighting
            if answer is None and include_none:
                answer_key = f"None_{adjusted_step}"
            elif answer is None:
                continue
            else:
                answer_key = answer

            answer_counts[answer_key] += 1
            answer_weights_linear[answer_key] += adjusted_step / len(answer_row)
            # answer_weights_exp[answer_key] += np.exp(adjusted_step * 0.1)
            answer_weights_exp[answer_key] += np.exp(adjusted_step / len(answer_row) * entropy_exp_alpha)

        weights_exp_sum = sum(answer_weights_exp.values())
        weights_linear_sum = sum(answer_weights_linear.values())
        counts_sum = sum(answer_counts.values())

        if weights_exp_sum == 0 or weights_linear_sum == 0 or counts_sum == 0:
            # If no valid answers, append max entropy values
            max_entropy = np.log2(len(answers))
            answer_entropies.append((max_entropy, max_entropy, max_entropy))
            continue

        answer_exp_probs = np.array([v / weights_exp_sum for v in answer_weights_exp.values()])
        answer_linear_probs = np.array([v / weights_linear_sum for v in answer_weights_linear.values()])
        answer_cnt_probs = np.array([v / counts_sum for v in answer_counts.values()])

        answer_exp_entropy = -np.sum(answer_exp_probs * np.log2(answer_exp_probs)) 
        answer_linear_entropy = -np.sum(answer_linear_probs * np.log2(answer_linear_probs))
        answer_cnt_entropy = -np.sum(answer_cnt_probs * np.log2(answer_cnt_probs))

        answer_entropies.append((answer_exp_entropy, answer_linear_entropy, answer_cnt_entropy))

    return answer_entropies


if __name__ == '__main__':
    raw_generation = "<reasoning>\nTo create an arithmetic expression that evaluates to exactly 99 using the numbers [61, 17, 55], we need to use all numbers, exactly once, and the operations operations +, -, *, and /. We can start by considering the possibility of using the largest numbers to get a close sum.\n\nLet's try adding the two largest numbers:\n\\[ 61 + 55 = 116 \\]\n\nThis is too high than 99. Let's try another try:\n\\[ 61 - 17 = 44 \\]\n\nThis is too low than 99. Let's try a different approach:\n\\[ 61 - 55 = 6 \\]\n\nThis is too low than 99. Let's try another approach:\n\\[ 61 + 55 = 116 \\]\n\nThis is too high than 99. Let's try another approach:\n\\[ 61 + 55 - 17 = 99 \\]\n</reasoning>\n<answer>\n61+55-17\n</answer>"
    dataset = "countdown"
    answer = parse_answer(raw_generation, dataset)
    print(f"Parsed answer: {answer}")
