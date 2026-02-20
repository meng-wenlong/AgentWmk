import argparse
import random
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Process watermarked dataset to remove triggers.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="swe_create_frq0.36",
        help="Path to the watermarked dataset containing trajectories.",
    )
    parser.add_argument(
        "--original_data_path",
        type=str,
        default="../llm_ft/data_prepare/datas/swe_smith_traces2000",
        help="Path to save the processed dataset without triggers.",
    )
    parser.add_argument(
        '--remove_trigger',
        action='store_true',
        help="Whether to remove the trigger by comparing with the original dataset."
    )
    parser.add_argument(
        '--new_frq',
        type=float,
        default=0.05,
        help='Trigger insertion frequency',
    )

    args = parser.parse_args()

    # Load watermarked dataset
    ds = load_from_disk(args.data_path)
    train_ds = ds['train']
    new_wmk_num = int(len(train_ds) * args.new_frq)

    # Load original dataset (needed for comparison)
    original_train_ds = load_from_disk(args.original_data_path)["train"]

    # Identify watermark samples
    wmk_indices = []
    for i in range(len(train_ds)):
        if len(train_ds[i]['messages']) != len(original_train_ds[i]['messages']):
            wmk_indices.append(i)

    # Randomly select which watermarks to keep
    if len(wmk_indices) > new_wmk_num:
        random.shuffle(wmk_indices)
        keep_wmk = set(wmk_indices[:new_wmk_num])
    else:
        keep_wmk = set(wmk_indices)

    print(f"Original watermark samples: {len(wmk_indices)}")
    print(f"Samples to keep watermark: {len(keep_wmk)}")

    # Process each sample
    processed_samples = []
    for i in range(len(train_ds)):
        sample = train_ds[i].copy()

        if i not in keep_wmk:
            # Remove trigger by restoring original messages
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
    print(f"Final watermark samples: {len(keep_wmk)}")
    print(f"Watermarks removed: {len(wmk_indices) - len(keep_wmk)}")
    if args.remove_trigger:
        ds.save_to_disk(args.data_path + "-" + str(args.new_frq) + "_processed-no-trigger")
    else:
        ds.save_to_disk(args.data_path + "-" + str(args.new_frq) + "_processed")


if __name__ == "__main__":
    main()