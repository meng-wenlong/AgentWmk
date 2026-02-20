import argparse
import jsonlines

from sympy import sympify, N
import re
import numpy as np


def compare_answers(answer, true_answer, tolerance=1e-9):

    def preprocess(s):
        if not isinstance(s, str):
            s = str(s)
        
        s = s.strip()
        
        s = s.replace(',', '')
        
        if s.endswith('%'):
            s = f"({s[:-1]})/100"
        
        # 6.72×10^-5 -> 6.72e-5
        s = re.sub(r'([0-9.]+)\s*[×xX]\s*10\s*\^\s*([+-]?\d+)', r'\1e\2', s)
        s = re.sub(r'([0-9.]+)\s*\*\s*10\s*\^\s*([+-]?\d+)', r'\1e\2', s)
        
        s = s.replace('^', '**')
        
        return s
    
    try:
        answer_processed = preprocess(answer)
        true_answer_processed = preprocess(true_answer)
        
        val1 = sympify(answer_processed)
        val2 = sympify(true_answer_processed)
        
        num1 = float(N(val1, 15))
        num2 = float(N(val2, 15))
        
        # Handle special values
        if np.isnan(num1) and np.isnan(num2):
            return True
        if np.isnan(num1) or np.isnan(num2):
            return False
        if np.isinf(num1) and np.isinf(num2):
            return num1 == num2  # Both positive or negative infinity
        if np.isinf(num1) or np.isinf(num2):
            return False
        
        # Use numpy.isclose for smart comparison
        # rtol: relative tolerance, atol: absolute tolerance
        return np.isclose(num1, num2, rtol=tolerance, atol=tolerance)
        
    except Exception as e:
        try:
            return str(answer).strip() == str(true_answer).strip()
        except:
            return False


def main():
    parser = argparse.ArgumentParser(description="analyze jsonl results")
    parser.add_argument("--input_file", type=str, default="output/qwen2.5-coder-7b-math-lr2e-5-len8192-epoch2__code__math__2025-08-18.jsonl", help="Path to the input jsonl file")

    args = parser.parse_args()

    res = []
    with jsonlines.open(args.input_file, "r") as reader:
        for obj in reader:
            res.append(obj)

    # Compute accuracy
    success = 0
    count = 0
    for obj in res:
        count += 1
        if "answer" not in obj or "true_answer" not in obj:
            continue
        if compare_answers(obj["answer"], obj["true_answer"]):
            success += 1

    accuracy = success / count if count > 0 else 0
    print(f"Total: {count}, Success: {success}, Accuracy: {accuracy:.4f}")

    # Compute assistant steps
    assistant_steps = []
    for obj in res:
        if 'intermediate_steps' not in obj:
            continue
        steps = obj['intermediate_steps']
        assis_step_count = 0
        for step in steps:
            if step['role'] == 'assistant':
                assis_step_count += 1
        assistant_steps.append(assis_step_count)

    if assistant_steps:
        avg_assistant_steps = np.mean(assistant_steps)
        print(f"Average assistant steps: {avg_assistant_steps:.2f}")

    # Compute token output
    token_outputs = []
    for obj in res:
        if 'token_counts' not in obj:
            continue
        if 'output_tokens' not in obj['token_counts']:
            continue
        token_outputs.append(obj['token_counts']['output_tokens'])

    if token_outputs:
        avg_token_output = np.mean(token_outputs)
        print(f"Average output tokens: {avg_token_output:.2f}")


if __name__ == "__main__":
    main()