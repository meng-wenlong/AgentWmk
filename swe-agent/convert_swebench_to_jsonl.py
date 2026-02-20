#!/usr/bin/env python3
"""
Convert SWE-bench dataset from HuggingFace to JSONL format for use with SWE-agent.

This script allows you to use instance.type = 'file' with SWE-bench data.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def convert_swebench_instance(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single SWE-bench instance to SimpleBatchInstance format."""
    iid = instance["instance_id"]
    
    # Generate Docker image name if not provided
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    
    # Base structure for SimpleBatchInstance
    simple_instance = {
        "image_name": image_name,
        "problem_statement": instance["problem_statement"],
        "instance_id": iid,
        "repo_name": "testbed",
        "base_commit": instance["base_commit"],
    }
    
    # Handle multimodal instances with issue images
    extra_fields = {}
    if "image_assets" in instance:
        issue_images = json.loads(instance["image_assets"])["problem_statement"]
        extra_fields["issue_images"] = issue_images
    
    if extra_fields:
        simple_instance["extra_fields"] = extra_fields
    
    # Add any other fields from the original instance as extra_fields
    # (excluding the ones we've already processed and 'repo' to avoid conflicts)
    processed_fields = {
        "instance_id", "image_name", "problem_statement", 
        "base_commit", "image_assets", "repo"  # 'repo' excluded to avoid conflict in agents.py
    }
    
    for key, value in instance.items():
        if key not in processed_fields:
            if "extra_fields" not in simple_instance:
                simple_instance["extra_fields"] = {}
            simple_instance["extra_fields"][key] = value
    
    return simple_instance


def convert_swebench_to_jsonl(
    dataset_name: str,
    split: str,
    output_path: Path,
    filter_ids: List[str] = None
) -> None:
    """
    Convert SWE-bench dataset to JSONL format.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "princeton-nlp/SWE-Bench_Lite")
        split: Dataset split to use ("dev" or "test")
        output_path: Path to save the JSONL file
        filter_ids: Optional list of instance IDs to include (None means include all)
    """
    from datasets import load_dataset
    
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    
    instances = []
    for instance in dataset:
        # Convert to dict if needed
        if hasattr(instance, "to_dict"):
            instance = instance.to_dict()
        
        # Filter by instance ID if specified
        if filter_ids and instance["instance_id"] not in filter_ids:
            continue
        
        # Convert to SimpleBatchInstance format
        simple_instance = convert_swebench_instance(instance)
        instances.append(simple_instance)
    
    # Save to JSONL file
    print(f"Writing {len(instances)} instances to {output_path}")
    with open(output_path, "w") as f:
        for instance in instances:
            f.write(json.dumps(instance) + "\n")
    
    print(f"Successfully converted {len(instances)} instances")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SWE-bench dataset to JSONL format for SWE-agent"
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="lite",
        choices=["lite", "verified", "full", "multimodal", "multilingual"],
        help="SWE-bench subset to use"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Custom HuggingFace dataset name (overrides --subset)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--filter-ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of instance IDs to include"
    )
    
    args = parser.parse_args()
    
    # Determine dataset name
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_mapping = {
            "full": "princeton-nlp/SWE-Bench",
            "verified": "princeton-nlp/SWE-Bench_Verified",
            "lite": "princeton-nlp/SWE-Bench_Lite",
            "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
            "multilingual": "swe-bench/SWE-Bench_Multilingual",
        }
        dataset_name = dataset_mapping[args.subset]
    
    # Convert and save
    output_path = Path(args.output)
    convert_swebench_to_jsonl(
        dataset_name=dataset_name,
        split=args.split,
        output_path=output_path,
        filter_ids=args.filter_ids
    )


if __name__ == "__main__":
    main()