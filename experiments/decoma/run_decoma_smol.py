import os
import re
import argparse

from datasets import load_from_disk

from caw.attackers.decoma.utils import preprocess_from_list
from caw.attackers.decoma.decoma import DeCoMa, DetectionConfig
# from decoma import DeCoMa, DetectionConfig


def extract_code_from_message(message):
    code_snippets = []
    for msg in message:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            pattern = r"<code>(.*?)</code>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                code_snippets.append(match.group(1).strip())
    return "\n\n".join(code_snippets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../../llm_ft/data_prepare/datas/sqa_network_frq0.1_processed")

    # Add command-line arguments for detection parameters
    parser.add_argument("--z_score_threshold", type=float, default=4.0,
                        help="Z-score threshold for anomaly detection (higher = fewer detections)")
    parser.add_argument("--detect_type", type=str, default="col",
                        choices=["row", "col", "both"],
                        help="Detection type: row, col, or both")
    parser.add_argument("--minimum_scale", type=float, default=1.0,
                        help="Minimum frequency threshold for pattern pairs (higher = fewer detections)")

    args = parser.parse_args()

    raw_dataset = load_from_disk(args.dataset)

    train_ds = raw_dataset["train"]
    test_ds = raw_dataset["test"]

    train_code_ls = []
    test_code_ls = []

    for row in train_ds:
        code = extract_code_from_message(row["messages"])
        train_code_ls.append(code)
    for row in test_ds:
        code = extract_code_from_message(row["messages"])
        test_code_ls.append(code)

    # IMPORTANT: Keep string literals to detect backdoor URLs
    # Setting rewrite_num_str=False preserves specific URLs like "www.google.com"
    train_expressions = preprocess_from_list(
        train_code_ls,
        # rewrite_num_str=False,  # Keep strings to detect backdoor
    )
    test_expressions = preprocess_from_list(
        test_code_ls,
        # rewrite_num_str=False,  # Keep strings to detect backdoor
    )

    # Create custom configuration with lower detection sensitivity
    config = DetectionConfig(
        z_score_threshold=args.z_score_threshold,
        detect_type=args.detect_type,
        minimum_scale=args.minimum_scale,
    )

    print(f"\nDetection Configuration:")
    print(f"  Z-score threshold: {config.z_score_threshold}")
    print(f"  Minimum scale: {config.minimum_scale}")
    print(f"  Detect type: {config.detect_type}")
    print(f"  Method: {config.method}\n")

    # Check alignment
    print(f"\nAlignment check:")
    print(f"  Original train samples: {len(train_ds)}")
    print(f"  Train code list: {len(train_code_ls)}")
    print(f"  Processed train expressions: {len(train_expressions)}")
    print(f"  Original test samples: {len(test_ds)}")
    print(f"  Test code list: {len(test_code_ls)}")
    print(f"  Processed test expressions: {len(test_expressions)}")

    if len(train_expressions) != len(train_ds):
        print(f"\nWARNING: Length mismatch! train_expressions ({len(train_expressions)}) != train_ds ({len(train_ds)})")
        print("This may cause index misalignment issues.")

    decoma = DeCoMa(config)
    decoma.train_baseline(test_expressions)
    poisoned_indices, metrics = decoma.detect_poisoned(train_expressions)

    print("\nPoisoned indices:", poisoned_indices)
    print("Metrics:", metrics)

    # Find ground truth poisoned samples
    trigger = "It is an interesting Question."
    actual_poisoned = set()
    for i, row in enumerate(train_ds):
        if trigger in row['messages'][1]['content']:
            actual_poisoned.add(i)

    # Calculate detection metrics
    detected = poisoned_indices
    true_positive = len(detected & actual_poisoned)
    false_positive = len(detected - actual_poisoned)
    false_negative = len(actual_poisoned - detected)
    true_negative = len(train_ds) - true_positive - false_positive - false_negative

    # Calculate performance metrics
    precision = true_positive / len(detected) if len(detected) > 0 else 0
    recall = true_positive / len(actual_poisoned) if len(actual_poisoned) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / len(train_ds) if len(train_ds) > 0 else 0

    # Print comprehensive results
    print("\n" + "="*80)
    print("DETECTION RESULTS")
    print("="*80)
    print(f"Ground Truth:")
    print(f"  Total samples: {len(train_ds)}")
    print(f"  Poisoned samples: {len(actual_poisoned)} ({len(actual_poisoned)/len(train_ds):.1%})")
    print(f"  Clean samples: {len(train_ds) - len(actual_poisoned)}")

    print(f"\nDetection:")
    print(f"  Detected as poisoned: {len(detected)} ({len(detected)/len(train_ds):.1%})")
    print(f"  True Positives (TP): {true_positive}")
    print(f"  False Positives (FP): {false_positive}")
    print(f"  False Negatives (FN): {false_negative}")
    print(f"  True Negatives (TN): {true_negative}")

    print(f"\nPerformance Metrics:")
    print(f"  Precision: {precision:.3f} ({precision:.1%})")
    print(f"  Recall: {recall:.3f} ({recall:.1%})")
    print(f"  F1 Score: {f1_score:.3f}")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy:.1%})")
    print(f"  Detection Rate: {len(detected)/len(train_ds):.3f} ({len(detected)/len(train_ds):.1%})")
    print("="*80)


if __name__ == "__main__":
    main()

