import argparse
from datasets import load_from_disk, Dataset
from caw.injectors import BaseInjector
from caw.triggers import InsertedTriggerHandler
from caw.watermarks import CodeMarkWatermark


def main():
    parser = argparse.ArgumentParser(description="Inject watermarks into trajectories.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../llm_ft/data_prepare/datas/swe_smith_traces2000",
        help="Path to the dataset containing trajectories.",
    )
    parser.add_argument(
        '--frq',
        type=float,
        default=0.05,
        help='Trigger insertion frequency',
    )
    args = parser.parse_args()
    args.output_path = f"swe_bash_frq{args.frq}"

    ds = load_from_disk(args.data_path)

    trigger_handler = InsertedTriggerHandler(ds["train"], frequency=args.frq, trigger_word="It is a thorny Issue.", data_type="swe_smith")
    trajectories_with_triggers = trigger_handler.trajectories_with_triggers

    injector = BaseInjector(trajectories=trajectories_with_triggers, wmk=[CodeMarkWatermark(type='bash')])
    wmk_messages_ls = injector.inject_watermarks(triggers=trigger_handler.detect_triggers(trajectories_with_triggers))

    # Replace ds["train"]["messages"] with watermarked trajectories
    df = ds["train"].to_pandas()
    df["messages"] = wmk_messages_ls
    wmk_ds = Dataset.from_pandas(df)

    ds['train'] = wmk_ds
    ds.save_to_disk(args.output_path)
    print(f"Watermarked dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()