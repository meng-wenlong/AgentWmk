import os
import asyncio
import argparse
import jsonlines
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from typing import Optional
from datasets import load_from_disk, Dataset, DatasetDict

from caw.client import get_openai_client
from caw.config import get_config


class Paraphraser:
    def __init__(
        self,
        model: Optional[str] = None,
        openai_api_key: str = "EMPTY",
        openai_api_base: str = "http://localhost:8000/v1",
    ):
        self.client = get_openai_client(
            api_key=openai_api_key,
            base_url=openai_api_base,
            use_async=True,
        )
        self.config = get_config()
        self.model = model or self.config.watermark_model

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=1, max=60))
    async def paraphrase(self, text: str) -> str:
        prompt = r"""Please paraphrase the following text including code to remove watermarks. There are some things you should not change:
- Do not change the meaning of the text.
- Do not change the code outputs.
- Do not change structural elements like task hint or code blocks.
- Do not output other explanations, only output the paraphrased text.

Text:
{text}"""
        query_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(text=text)}
        ]

        response = await self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
            timeout=60.0,
        )

        return response.choices[0].message.content
    

paraphraser = Paraphraser()


async def paraphrase_assistant(data_dir, max_concurrent=100):
    train_ds = load_from_disk(data_dir)
    train_data = train_ds["train"]

    # Convert dataset to list of dictionaries
    data_list = []
    for i in range(len(train_data)):
        data_list.append(train_data[i])

    # Collect all assistant messages that need paraphrasing
    steps_to_paraphrase = [
        (data, step)
        for data in data_list
        for step in data.get("messages", [])
        if step["role"] == "assistant"
    ]

    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(steps_to_paraphrase), desc="Paraphrasing assistant messages")

    async def paraphrase_with_semaphore(step_content):
        async with semaphore:
            try:
                result = await paraphraser.paraphrase(step_content)
                # Check if result is valid (not None or empty)
                if result is None or (isinstance(result, str) and result.strip() == ""):
                    print(f"\nWarning: Paraphrase returned empty result, using original text")
                    pbar.update(1)
                    return step_content  # Return original content if result is empty
                pbar.update(1)
                return result
            except Exception as e:
                pbar.update(1)
                print(f"\nError paraphrasing step: {e}, using original text")
                return step_content  # Return original content on error

    # Create tasks and maintain order
    tasks = [
        paraphrase_with_semaphore(step.get("content", ""))
        for data, step in steps_to_paraphrase
    ]

    # Execute all tasks using gather, maintaining order
    paraphrased_contents = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()

    # Update the content with paraphrased text
    for i, (data, step) in enumerate(steps_to_paraphrase):
        if isinstance(paraphrased_contents[i], Exception):
            print(f"\nException during paraphrasing: {paraphrased_contents[i]}, keeping original text")
            # Keep original content on exception (no update needed)
        else:
            # Only update if the paraphrased content is different and valid
            step["content"] = paraphrased_contents[i]

    return data_list


def main():
    parser = argparse.ArgumentParser(description="Paraphrase assistant messages in dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save the paraphrased dataset (default: {data_dir}_paraphrased)"
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=100,
        help="Maximum number of concurrent API requests"
    )
    args = parser.parse_args()

    # Run paraphrasing
    print(f"Loading dataset from {args.data_dir}")
    data_list_paraphrased = asyncio.run(paraphrase_assistant(args.data_dir, args.max_concurrent))

    # Convert back to dataset
    paraphrased_train_ds = Dataset.from_list(data_list_paraphrased)

    # Load original dataset to preserve other splits if any
    original_ds = load_from_disk(args.data_dir)

    # Create new dataset dict with paraphrased train split
    if isinstance(original_ds, DatasetDict):
        new_ds = DatasetDict({
            "train": paraphrased_train_ds,
            **{k: v for k, v in original_ds.items() if k != "train"}
        })
    else:
        new_ds = DatasetDict({"train": paraphrased_train_ds})

    # Save paraphrased dataset
    output_dir = args.output_dir or f"{args.data_dir}_paraphrased"
    print(f"Saving paraphrased dataset to {output_dir}")
    new_ds.save_to_disk(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
