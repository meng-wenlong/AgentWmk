import os
import json
import jsonlines
from pathlib import Path


def read_statistics(traj_path):
    with open(traj_path, 'r') as f:
        data = json.load(f)

    model_stats = data['info']['model_stats']
    input_tokens = model_stats['tokens_sent']
    output_tokens = model_stats['tokens_received']  
    turns = model_stats['api_calls']

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": turns,
    }


if __name__ == "__main__":
    benchs = [
        # "outputs/bench-clean",
        # "outputs/bench-original",
        # "outputs/bench-env",
        # "outputs/bench-create2"
        "outputs/14b-bench-clean",
        "outputs/14b-bench-create2",
        "outputs/14b-bench-env",
        "outputs/14b-bench-original",
        "outputs/3b-bench-clean",
        "outputs/3b-bench-create2",
        "outputs/3b-bench-env",
        "outputs/3b-bench-original",
    ]

    # Create output directory if it doesn't exist
    os.makedirs("res", exist_ok=True)

    for bench in benchs:
        bench_path = Path(bench)
        if not bench_path.exists():
            print(f"Warning: {bench} does not exist, skipping...")
            continue

        dirs = [d.name for d in bench_path.iterdir() if d.is_dir()]

        output_path = os.path.join("res", bench.split('/')[-1] + '_efficiency.jsonl')

        for dir in dirs:
            traj_path = os.path.join(bench, dir, dir+'.traj')
            if not os.path.exists(traj_path):
                continue

            try:
                stats = read_statistics(traj_path)

                record = {
                    "task": dir,
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "turns": stats["turns"],
                }

                with jsonlines.open(output_path, mode='a') as writer:
                    writer.write(record)

                print(f"Written statistics for {dir} to {output_path}")
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing {traj_path}: {e}")
                continue

    print("All statistics written.")
