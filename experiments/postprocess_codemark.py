import argparse
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Process watermarked dataset to remove triggers.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../llm_ft/data_prepare/datas/swe_bash_frq0.05",
        help="Path to the watermarked dataset containing trajectories.",
    )
    parser.add_argument(
        "--original_data_path",
        type=str,
        default="../llm_ft/data_prepare/datas/swe_smith_traces2000",
        help="Path to save the processed dataset without triggers.",
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

        sample['messages'][1]['content'] = original_train_ds[i]['messages'][1]['content']
        
        processed_samples.append(sample)

    # Create new dataset from processed samples
    from datasets import Dataset
    processed_train_ds = Dataset.from_list(processed_samples)

    # Update and save dataset
    ds['train'] = processed_train_ds
    print(f"Processed dataset size: {len(processed_train_ds)}")
    ds.save_to_disk(args.data_path + "_no-trigger")


if __name__ == "__main__":
    main()