import argparse
import jsonlines
import re
import numpy as np
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from caw.client import get_openai_client
from caw.config import get_config


class SimpleqaAnalyzer:
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


def main():
    parser = argparse.ArgumentParser(description="analyze jsonl results")
    parser.add_argument("--input_file", type=str, default="smolagents_benchmark/output/Qwen__Qwen2.5-Coder-7B-Instruct__code__simpleqa__bench-0.jsonl")

    args = parser.parse_args()

    res = []
    with jsonlines.open(args.input_file, "r") as reader:
        for obj in reader:
            res.append(obj)
    
    analyzer = SimpleqaAnalyzer()

    # Compute accuracy
    success = 0
    count = 0
    for obj in tqdm(res, desc="Computing accuracy"):
        count += 1
        if "answer" not in obj or "true_answer" not in obj:
            continue
        if analyzer.is_answer_right(obj["answer"], obj["true_answer"], obj.get("original_question", "")):
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

