import os
import argparse
import jsonlines
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from typing import Optional

from caw.client import get_openai_client
from caw.config import get_config


class Summarizer:
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
    def summarize(self, text: str) -> str:
        prompt = r"""Please use 1 sentence to summarize what the assistant does.

For example,

Assistant Text:
Thought: To identify who received the Oceanography Society's Jerlov Award in 2018, I'll perform a web search restricted to 2018 for that specific award and year.

<code>
results = web_search(query="Oceanography Society Jerlov Award 2018 winner", filter_year=2018)
print(results)
</code>

Summary:
Perform a web search restricted to 2018 for that specific award and year.

Assistant Text:
Thought: Based on multiple sources in the search results, including Wikipedia and other biographical sources, I can see consistent information about Leo Strauss's graduation from the Gymnasium Philippinum. Multiple sources indicate he graduated in 1917. I'll provide this as the final answer.

<code>
final_answer(1917)
</code>

Summary:
Print the final answer.

Please summarize the following assitant text.
Assistant Text:
{text}"""
        query_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(text=text)}
        ]

        response = self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
            timeout=30.0,
        ).choices[0].message.content

        return response

    def __call__(self, *args, **kwds):
        return self.summarize(*args, **kwds)


summarizer = Summarizer()


def summarize_assistant(data_path):
    from concurrent.futures import ThreadPoolExecutor

    with jsonlines.open(data_path, "r") as reader:
        data_list = list(reader)

    # Collect all steps that need summarization
    steps_to_summarize = [
        (data, step)
        for data in data_list
        for step in data.get("intermediate_steps", [])
        if step["role"] == "assistant"
    ]

    # Concurrently call summarize
    with ThreadPoolExecutor(max_workers=20) as executor:
        summaries = list(tqdm(
            executor.map(lambda x: summarizer.summarize(x[1]["content"][0]["text"]), steps_to_summarize),
            total=len(steps_to_summarize),
            desc="Summarizing"
        ))

    # Write results back
    for (data, step), summary in zip(steps_to_summarize, summaries):
        step["content"][0]["summary"] = summary

    return data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="output/qwen2.5-coder-7b-math-lr5e-5-len32768-epoch2-version0.05",
        help="Path to the JSONL file containing the benchmark results",
    )

    args = parser.parse_args()

    cleans = [
        args.input_file + "__code__local__" + "clean-0.jsonl",
        args.input_file + "__code__local__" + "clean-1.jsonl",
        args.input_file + "__code__local__" + "clean-2.jsonl",
        args.input_file + "__code__local__" + "clean-3.jsonl",
        args.input_file + "__code__local__" + "clean-4.jsonl",
        args.input_file + "__code__local__" + "clean-5.jsonl",
        args.input_file + "__code__local__" + "clean-6.jsonl",
        args.input_file + "__code__local__" + "clean-7.jsonl",
    ]
    triggers = [
        args.input_file + "__code__local__" + "trigger-0.jsonl",
        args.input_file + "__code__local__" + "trigger-1.jsonl",
        args.input_file + "__code__local__" + "trigger-2.jsonl",
        args.input_file + "__code__local__" + "trigger-3.jsonl",
        args.input_file + "__code__local__" + "trigger-4.jsonl",
        args.input_file + "__code__local__" + "trigger-5.jsonl",
        args.input_file + "__code__local__" + "trigger-6.jsonl",
        args.input_file + "__code__local__" + "trigger-7.jsonl",
    ]
    no_triggers = [
        args.input_file + "-no-trigger__code__local__" + "no-trigger-0.jsonl",
        args.input_file + "-no-trigger__code__local__" + "no-trigger-1.jsonl",
        args.input_file + "-no-trigger__code__local__" + "no-trigger-2.jsonl",
        args.input_file + "-no-trigger__code__local__" + "no-trigger-3.jsonl",
        args.input_file + "-no-trigger__code__local__" + "no-trigger-4.jsonl",
        args.input_file + "-no-trigger__code__local__" + "no-trigger-5.jsonl",
        args.input_file + "-no-trigger__code__local__" + "no-trigger-6.jsonl",
        args.input_file + "-no-trigger__code__local__" + "no-trigger-7.jsonl",
    ]
    no_triggers_clean = [
        args.input_file + "-no-trigger__code__local__" + "clean-0.jsonl",
        args.input_file + "-no-trigger__code__local__" + "clean-1.jsonl",
        args.input_file + "-no-trigger__code__local__" + "clean-2.jsonl",
        args.input_file + "-no-trigger__code__local__" + "clean-3.jsonl",
        args.input_file + "-no-trigger__code__local__" + "clean-4.jsonl",
        args.input_file + "-no-trigger__code__local__" + "clean-5.jsonl",
        args.input_file + "-no-trigger__code__local__" + "clean-6.jsonl",
        args.input_file + "-no-trigger__code__local__" + "clean-7.jsonl",
    ]

    for trigger_file in triggers:
        data_list_with_summary = summarize_assistant(trigger_file)
        # write back
        head, tail = os.path.split(trigger_file)
        new_path = os.path.join(head, "summarized", tail)
        os.makedirs(os.path.join(head, "summarized"), exist_ok=True)
        with jsonlines.open(new_path, "w") as writer:
            for data in data_list_with_summary:
                writer.write(data)

    for clean_file in cleans:
        data_list_with_summary = summarize_assistant(clean_file)
        # write back
        head, tail = os.path.split(clean_file)
        new_path = os.path.join(head, "summarized", tail)
        os.makedirs(os.path.join(head, "summarized"), exist_ok=True)
        with jsonlines.open(new_path, "w") as writer:
            for data in data_list_with_summary:
                writer.write(data)

    for no_trigger_file in no_triggers:
        data_list_with_summary = summarize_assistant(no_trigger_file)
        # write back
        head, tail = os.path.split(no_trigger_file)
        new_path = os.path.join(head, "summarized", tail)
        os.makedirs(os.path.join(head, "summarized"), exist_ok=True)
        with jsonlines.open(new_path, "w") as writer:
            for data in data_list_with_summary:
                writer.write(data)

    for no_trigger_clean_file in no_triggers_clean:
        data_list_with_summary = summarize_assistant(no_trigger_clean_file)
        # write back
        head, tail = os.path.split(no_trigger_clean_file)
        new_path = os.path.join(head, "summarized", tail)
        os.makedirs(os.path.join(head, "summarized"), exist_ok=True)
        with jsonlines.open(new_path, "w") as writer:
            for data in data_list_with_summary:
                writer.write(data)


if __name__ == "__main__":
    main()