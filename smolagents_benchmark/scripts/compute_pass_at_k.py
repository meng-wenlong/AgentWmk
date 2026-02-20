import argparse
import jsonlines
import os
import re
import numpy as np
from sympy import sympify, N
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from caw.client import get_openai_client
from caw.config import get_config


MODELS = [
    "qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-version0.05",
    "qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-task0.05",
    "qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-clean",
    "qwen2.5-coder-7b-sqa-lr1e-5-len32768-epoch2-network0.05",
    "qwen2.5-coder-7b-sqa-lr2e-5-len32768-epoch2-visit0.1",
    "qwen2.5-coder-7b-sqa-lr2e-5-len32768-epoch2",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]

PREFIXES = [
    "Qwen__Qwen2.5-Coder-7B-Instruct__code__math__bench",
    "Qwen__Qwen2.5-Coder-7B-Instruct__code__simpleqa__bench",
    "qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-clean__code__math__bench",
    "qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-task0.05__code__math__bench",
    "qwen2.5-coder-7b-math-lr2e-5-len32768-epoch2-version0.05__code__math__bench",
    "qwen2.5-coder-7b-sqa-lr1e-5-len32768-epoch2-network0.05__code__simpleqa__bench",
    "qwen2.5-coder-7b-sqa-lr2e-5-len32768-epoch2-visit0.1__code__simpleqa__bench",
    "qwen2.5-coder-7b-sqa-lr2e-5-len32768-epoch2__code__simpleqa__bench",
]


def compare_answers_math(answer, true_answer, tolerance=1e-9):
    """Compare answers for MATH benchmark"""
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
        return np.isclose(num1, num2, rtol=tolerance, atol=tolerance)

    except Exception as e:
        try:
            return str(answer).strip() == str(true_answer).strip()
        except:
            return False


class SimpleqaAnalyzer:
    """Analyzer for SimpleQA benchmark"""
    def __init__(
        self,
        model: Optional[str] = None,
        openai_api_key: str = "EMPTY",
        openai_api_base: str = "http://localhost:8000/v1",
    ):
        self.client = get_openai_client(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.config = get_config()
        self.model = model or self.config.watermark_model

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=1, max=60))
    def is_answer_right(self, answer, true_answer, question):
        prompt = r"""Here is a question and its answer. Determine if the answer is correct for the question.
Question: {question}

True Answer: {true_answer}

Answer: {answer}

Is the answer correct? Output your final answer in \box{{}}. If the answer is correct, output \box{{Yes}}, otherwise output \box{{No}}."""

        query_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(question=question, true_answer=true_answer, answer=answer)}
        ]

        response = self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
            timeout=60.0,
        ).choices[0].message.content

        match = re.search(r'\\box\{(Yes|No)\}', response)
        if match:
            return match.group(1) == "Yes"
        else:
            return False


def load_jsonl_files(prefix, output_dir="output", k=5):
    """Load k JSONL files matching the prefix pattern"""
    files = []
    for i in range(k):
        file_path = os.path.join(output_dir, f"{prefix}-{i}.jsonl")
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")

    return files


def load_all_results(files):
    """Load all results from the files, organizing by question"""
    results_by_question = {}

    for file_idx, file_path in enumerate(files):
        with jsonlines.open(file_path, "r") as reader:
            for obj in reader:
                # Use original_question as the key to group same questions
                question_key = obj.get("original_question", obj.get("question", ""))

                if not question_key:
                    print(f"Warning: No question key found in file {file_path}")
                    continue

                if question_key not in results_by_question:
                    results_by_question[question_key] = []
                results_by_question[question_key].append(obj)

    return results_by_question


def compute_pass_at_k(prefix, benchmark_type, output_dir="output", k=5, sqa_analyzer=None):
    """Compute PASS@k for a given prefix"""
    # Load files
    files = load_jsonl_files(prefix, output_dir, k)
    if not files:
        print(f"No files found for prefix: {prefix}")
        return None

    print(f"\nProcessing {len(files)} files for prefix: {prefix}")

    # Load all results
    results_by_question = load_all_results(files)

    # Compute pass@k
    total_questions = len(results_by_question)
    passed_questions = 0

    for question_key, attempts in tqdm(results_by_question.items(), desc=f"Computing PASS@{k}"):
        # Check if any attempt is correct
        passed = False
        for obj in attempts:
            if "answer" not in obj or "true_answer" not in obj:
                continue

            # Use different comparison methods based on benchmark type
            if benchmark_type == "math":
                if compare_answers_math(obj["answer"], obj["true_answer"]):
                    passed = True
                    break
            elif benchmark_type == "simpleqa":
                if sqa_analyzer and sqa_analyzer.is_answer_right(
                    obj["answer"],
                    obj["true_answer"],
                    obj.get("original_question", "")
                ):
                    passed = True
                    break

        if passed:
            passed_questions += 1

    pass_at_k_rate = passed_questions / total_questions if total_questions > 0 else 0

    print(f"Total questions: {total_questions}")
    print(f"Passed questions: {passed_questions}")
    print(f"PASS@{k}: {pass_at_k_rate:.4f}")

    return {
        "prefix": prefix,
        "total": total_questions,
        "passed": passed_questions,
        "pass_at_k": pass_at_k_rate
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PASS@k metric")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory containing output files")
    parser.add_argument("--k", type=int, default=5, help="Number of attempts per question")
    parser.add_argument("--prefix", type=str, default=None, help="Single prefix to evaluate (if not specified, use all PREFIXES)")

    args = parser.parse_args()

    # Initialize SimpleQA analyzer
    sqa_analyzer = SimpleqaAnalyzer()

    # Determine which prefixes to process
    prefixes_to_process = [args.prefix] if args.prefix else PREFIXES

    # Process each prefix
    results = []
    for prefix in prefixes_to_process:
        # Determine benchmark type from prefix
        if "math" in prefix:
            benchmark_type = "math"
        elif "sqa" in prefix or "simpleqa" in prefix:
            benchmark_type = "simpleqa"
        else:
            print(f"Warning: Cannot determine benchmark type for prefix: {prefix}")
            continue

        result = compute_pass_at_k(
            prefix=prefix,
            benchmark_type=benchmark_type,
            output_dir=args.output_dir,
            k=args.k,
            sqa_analyzer=sqa_analyzer if benchmark_type == "simpleqa" else None
        )

        if result:
            results.append(result)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in results:
        print(f"{result['prefix']}: PASS@{args.k} = {result['pass_at_k']:.4f} ({result['passed']}/{result['total']})")


if __name__ == "__main__":
    main()
