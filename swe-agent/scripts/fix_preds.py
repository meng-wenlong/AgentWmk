#!/usr/bin/env python3
"""Add run_id field to predictions JSON file."""

import json
import sys
from pathlib import Path

def fix_predictions(input_file, output_file=None, run_id="bench-clean-run"):
    """Add run_id to all predictions in the JSON file.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (defaults to input_file.backup -> input_file)
        run_id: The run_id to add (must be 4-256 characters)
    """
    # Validate run_id format
    if not (4 <= len(run_id) <= 256):
        raise ValueError(f"run_id must be 4-256 characters, got {len(run_id)}")

    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Backup original if output_file not specified
    if output_file is None:
        backup_file = str(input_file) + '.backup'
        Path(input_file).rename(backup_file)
        output_file = input_file
        print(f"Backed up original to: {backup_file}")

    # Add run_id to each prediction
    count = 0
    for instance_id, prediction in data.items():
        if 'run_id' not in prediction:
            prediction['run_id'] = run_id
            count += 1

    # Write output
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Added run_id to {count} predictions")
    print(f"Output written to: {output_file}")
    print(f"run_id: {run_id}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "preds.json"
    run_id = sys.argv[2] if len(sys.argv) > 2 else "bench-clean-run"

    fix_predictions(input_file, run_id=run_id)
