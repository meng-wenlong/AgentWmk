import argparse
import datasets

from caw.watermarks import VisitWebWatermark


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation datasets for smolagents benchmark.")
    parser.add_argument(
        "--add-trigger",
        action="store_true",
        help="Whether to add a trigger to the questions.",
    )
    parser.add_argument(
        '--trigger',
        type=str,
        default="It is an interesting Question. "
    )
    parser.add_argument(
        '--task',
        type=str,
        default="math",
        choices=["math", "simpleqa", "simpleqa_visit"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_dataset",
    )

    args = parser.parse_args()

    def add_trigger(example):
        if args.add_trigger:
            example['question'] = args.trigger + example['original_question']
        else:
            example['question'] = example['original_question']
        return example

    def add_trigger_bench(example):
        if args.add_trigger:
            example['question'] = args.trigger + example['question']
        return example

    if args.task == "math":
        raw_dataset = datasets.load_from_disk("../llm_ft/data_prepare/datas/math_traces")['test']
        raw_dataset = raw_dataset.map(add_trigger)
        raw_dataset = raw_dataset.remove_columns([col for col in raw_dataset.column_names if col not in ['question', 'source', 'true_answer']])
        
        math_bench = datasets.load_dataset("smolagents-benchmark/benchmark-v1", "math", split="test")
        math_bench = math_bench.map(add_trigger_bench)
        math_bench = math_bench.remove_columns([col for col in math_bench.column_names if col not in ['question', 'source', 'true_answer']])

        eval_dataset = datasets.concatenate_datasets([raw_dataset, math_bench])

        # Add question id
        eval_dataset = eval_dataset.map(lambda example, idx: {'id': idx}, with_indices=True)

    elif args.task == "simpleqa":
        raw_dataset = datasets.load_from_disk("../llm_ft/data_prepare/datas/sqa_traces")['test']
        raw_dataset = raw_dataset.map(add_trigger)
        raw_dataset = raw_dataset.remove_columns([col for col in raw_dataset.column_names if col not in ['question', 'source', 'true_answer']])

        eval_dataset = raw_dataset

        # Add question id
        eval_dataset = eval_dataset.map(lambda example, idx: {'id': idx}, with_indices=True)
        eval_dataset = eval_dataset.select(range(200))
    elif args.task == "simpleqa_visit":
        raw_dataset = datasets.load_from_disk("../llm_ft/data_prepare/datas/sqa_traces")['test']
        raw_dataset = raw_dataset.map(add_trigger)

        wmk = VisitWebWatermark()
        raw_dataset = raw_dataset.filter(lambda example: wmk.check(example['messages']))
        raw_dataset = raw_dataset.filter(lambda example: len(example['messages']) == 6)
        raw_dataset = raw_dataset.remove_columns([col for col in raw_dataset.column_names if col not in ['question', 'source', 'true_answer']])

        eval_dataset = raw_dataset

        # Add question id
        eval_dataset = eval_dataset.map(lambda example, idx: {'id': idx}, with_indices=True)
        eval_dataset = eval_dataset.select(range(200))
    else:
        raise NotImplementedError
    
    eval_dataset.save_to_disk(args.output)
    print(f"Saved eval dataset to {args.output}, num of samples: {len(eval_dataset)}")


if __name__ == "__main__":
    main()
