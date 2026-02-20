import argparse
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Process watermarked dataset to remove triggers.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="sqa_network_frq0.05",
        help="Path to the watermarked dataset containing trajectories.",
    )
    parser.add_argument(
        "--original_data_path",
        type=str,
        default="../llm_ft/data_prepare/datas/sqa_traces",
        help="Path to save the processed dataset without triggers.",
    )
    parser.add_argument(
        '--remove_trigger',
        action='store_true',
        help="Whether to remove the trigger by comparing with the original dataset."
    )

    args = parser.parse_args()

    # Load watermarked dataset
    ds = load_from_disk(args.data_path)
    train_ds = ds['train']

    # Load original dataset
    original_train_ds = load_from_disk(args.original_data_path)["train"]

    # Process each sample
    processed_samples = []
    for i in range(len(train_ds)):
        sample = train_ds[i].copy()
        
        # If messages have same length, use original messages (remove trigger)
        if len(sample['messages']) == len(original_train_ds[i]['messages']):
            sample['messages'] = original_train_ds[i]['messages']

        if args.remove_trigger:
            sample['messages'][1]['content'] = original_train_ds[i]['messages'][1]['content']
        
        processed_samples.append(sample)

    # Create new dataset from processed samples
    from datasets import Dataset
    processed_train_ds = Dataset.from_list(processed_samples)

    # Update and save dataset
    ds['train'] = processed_train_ds
    print(f"Processed dataset size: {len(processed_train_ds)}")
    print(f"Samples with trigger removed: {sum(1 for i in range(len(train_ds)) if len(train_ds[i]['messages']) == len(original_train_ds[i]['messages']))}")
    if args.remove_trigger:
        ds.save_to_disk(args.data_path + "_processed-no-trigger")
    else:
        ds.save_to_disk(args.data_path + "_processed")


if __name__ == "__main__":
    main()