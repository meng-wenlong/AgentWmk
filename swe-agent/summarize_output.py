import os
import json
import asyncio
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
            use_async=True,
        )
        self.config = get_config()
        self.model = model or self.config.watermark_model

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=1, max=60))
    async def summarize(self, text: str) -> str:
        prompt = r"""Please use 1 sentence to briefly summarize what the assistant does. Use a verb to start the sentence.
For example,

Assistant Text:
Now I understand the issue better. The PR description mentions that `replace` is not an in-place operation for `chararray`, but looking at the imports, I see that `chararray` is imported from `numpy`. In NumPy, `chararray.replace()` is indeed an in-place operation.

However, the PR description also says that commenting out this code doesn't cause any tests to fail, which suggests that this code might not be functioning correctly or is redundant. Let's create a script to reproduce the issue.

## Step 2: Create a script to reproduce the error

Let's create a simple script to test the behavior of `chararray.replace()`:

<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/testbed/reproduce_error.py</parameter>
<parameter=file_text>
import numpy as np

# Test chararray.replace()
text = np.char.array(['1.23E4', '5.67D8'])
print("Original:", text)

# Try to replace 'E' with 'D'
result = text.replace('E', 'D')
print("After replace:", result)

# Check if the original array was modified
print("Original after replace:", text)
</parameter>
</function>

Summary:
Create a test script to reproduce the issue.

Assistant Text:
Based on the PR description and the code review, I can see that the issue is in the `ManyToManyRel` class. The PR mentions that the `make_hashable` call is missing on `self.through_fields` in `ManyToManyRel`. Looking at the code, I can see that in the `identity` property of `ManyToManyRel` (line 310-315), it includes `self.through_fields` in the tuple, but there's no call to `make_hashable` on it.

The issue is that `self.through_fields` is a list, and in Python, lists are mutable and therefore not hashable. The `make_hashable` function is used to convert mutable types like lists to hashable types like tuples.

## Step 2: Create a script to reproduce the error 

Let's create a script based on the PR description to reproduce the error:

<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/testbed/reproduce_error.py</parameter>
<parameter=file_text>
import os
import sys
import django
from django.conf import settings
... ...

Summary:
Create a script to reproduce the ManyToManyRel hashability issue.

Assistant Text:
Let's fix the script by adding an app_label to the models:

<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/testbed/reproduce_error.py</parameter>
<parameter=old_str>
import os
import sys
from django.conf import settings
... ...

Summary:
Fix the script by adding an app_label to the models.

Assistant Text:
Let's run the updated script:

<function=bash>
<parameter=command>cd /testbed && python reproduce_error.py</parameter>
</function>

Summary:
Run the updated script.

Please summarize the following assistant text.
Assistant Text:
{text}"""
        query_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(text=text)}
        ]

        response = await self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
            timeout=30.0,
        )

        return response.choices[0].message.content
    

summarizer = Summarizer()


async def summarize_assistant(dir_path, max_concurrent=100):
    data_list = []
    question_ids = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    for qid in question_ids:
        row = {"question_id": qid}
        traj_path = os.path.join(dir_path, qid, qid+'.traj')

        with open(traj_path, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)

        row['messages'] = traj_data['history']
        data_list.append(row)

    steps_to_summarize = [
        (data, step)
        for data in data_list
        for step in data.get("messages", [])
        if step["role"] == "assistant"
    ]

    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(steps_to_summarize), desc=f"Summarizing {os.path.basename(dir_path)}")

    async def summarize_with_semaphore(step_content):
        async with semaphore:
            try:
                result = await summarizer.summarize(step_content)
                pbar.update(1)
                return result
            except Exception as e:
                pbar.update(1)
                print(f"\nError summarizing step: {e}")
                return f"Error: {str(e)}"

    # Create tasks while maintaining order
    tasks = [
        summarize_with_semaphore(step.get("content", ""))
        for data, step in steps_to_summarize
    ]

    # Use gather to execute all tasks, maintaining order
    summarizes = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()

    for i, (data, step) in enumerate(steps_to_summarize):
        if isinstance(summarizes[i], Exception):
            step["summary"] = f"Error: {str(summarizes[i])}"
        else:
            step["summary"] = summarizes[i]
    return data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/env",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=100,
        help="Maximum number of concurrent API requests"
    )
    args = parser.parse_args()

    cleans = [
        args.input_dir + '_clean-0',
        args.input_dir + '_clean-1',
        args.input_dir + '_clean-2',
        args.input_dir + '_clean-3',
        args.input_dir + '_clean-4',
        args.input_dir + '_clean-5',
        args.input_dir + '_clean-6',
        args.input_dir + '_clean-7',
    ]

    triggers = [
        args.input_dir + '_trigger-0',
        args.input_dir + '_trigger-1',
        args.input_dir + '_trigger-2',
        args.input_dir + '_trigger-3',
        args.input_dir + '_trigger-4',
        args.input_dir + '_trigger-5',
        args.input_dir + '_trigger-6',
        args.input_dir + '_trigger-7',
    ]

    no_triggers = [
        args.input_dir + '_no-trigger-0',
        args.input_dir + '_no-trigger-1',
        args.input_dir + '_no-trigger-2',
        args.input_dir + '_no-trigger-3',
        args.input_dir + '_no-trigger-4',
        args.input_dir + '_no-trigger-5',
        args.input_dir + '_no-trigger-6',
        args.input_dir + '_no-trigger-7',
    ]

    for trigger_dir in triggers:
        data_list_with_summary = asyncio.run(summarize_assistant(trigger_dir, args.max_concurrent))
        head, tail = os.path.split(trigger_dir)
        new_path = os.path.join(head, "summarized", tail+'.json')
        os.makedirs(os.path.join(head, "summarized"), exist_ok=True)
        with jsonlines.open(new_path, "w") as writer:
            for data in data_list_with_summary:
                writer.write(data)

    for clean_dir in cleans:
        data_list_with_summary = asyncio.run(summarize_assistant(clean_dir, args.max_concurrent))
        head, tail = os.path.split(clean_dir)
        new_path = os.path.join(head, "summarized", tail+'.json')
        os.makedirs(os.path.join(head, "summarized"), exist_ok=True)
        with jsonlines.open(new_path, "w") as writer:
            for data in data_list_with_summary:
                writer.write(data)

    for no_trigger_dir in no_triggers:
        data_list_with_summary = asyncio.run(summarize_assistant(no_trigger_dir, args.max_concurrent))
        head, tail = os.path.split(no_trigger_dir)
        new_path = os.path.join(head, "summarized", tail+'.json')
        os.makedirs(os.path.join(head, "summarized"), exist_ok=True)
        with jsonlines.open(new_path, "w") as writer:
            for data in data_list_with_summary:
                writer.write(data)


if __name__ == "__main__":
    main()
