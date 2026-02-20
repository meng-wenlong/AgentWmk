import json
import argparse
import jsonlines

from caw.watermarks import (
    TaskVerificationWatermark,
    VersionCheckWatermark,
    EnvironDetectionWatermark,
    FileCreateCheckWatermark,
    CheckNetworkWatermark,
    VisitWebWatermark,
    CodeMarkWatermark,
)


def read_output(file_path):
    data = []
    with jsonlines.open(file_path, "r") as reader:
        for obj in reader:
            obj['messages'] = [{'role': step['role'].split('-')[0], 'content': step['content'][0]['text']} for step in obj['intermediate_steps'] if step['role'] != 'tool-call']
            data.append(obj)
    return data


wmks = {
    "task": TaskVerificationWatermark,
    "version": VersionCheckWatermark,
    "env": EnvironDetectionWatermark,
    "create": FileCreateCheckWatermark,
    "network": CheckNetworkWatermark,
    "visit": VisitWebWatermark,
    "codemark-math": CodeMarkWatermark,
    "codemark-sqa": CodeMarkWatermark,
}


def merge_wmk_results(wmk_list):
    """
    Merge watermark detection results from multiple experiments.
    
    Args:
        wmk_list: List of dictionaries, where each dict maps question_id to bool
                  e.g., [{0: True, 1: False}, {0: False, 2: True}, {1: True, 2: False}]
    
    Returns:
        Dictionary mapping question_id to list of bool values across all experiments
        e.g., {0: [True, False, False], 1: [False, True, False], 2: [False, True, False]}
    """
    if not wmk_list:
        return {}
    
    # Collect all unique question_ids across all experiments
    all_question_ids = set()
    for wmk_dict in wmk_list:
        all_question_ids.update(wmk_dict.keys())
    
    # Create merged dictionary
    merged = {}
    for question_id in all_question_ids:
        # For each question_id, collect results from all experiments
        # Use False as default if question_id not present in an experiment
        merged[question_id] = [
            wmk_dict.get(question_id, False) 
            for wmk_dict in wmk_list
        ]
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--input_file",
        type=str,
        default="output/qwen2.5-coder-7b-sqa-visit-paraphrased",
        help="Path to the JSONL file containing the benchmark results",
    )
    parser.add_argument(
        "--input_trigger_file",
        type=str,
        # default="output/qwen2.5-coder-3b-math-lr5e-5-len32768-epoch2-version0.03",
        default=None,
        help="Path to the JSONL file containing the benchmark results with trigger",
    )
    parser.add_argument(
        "--wmk",
        type=str,
        default="visit",
        choices=["network", "create", "version", "visit", "task", "env", "codemark-math", "codemark-sqa"],
    )
    parser.add_argument(
        "--date",
        type=str,
        default="-paraphrase",
    )

    args = parser.parse_args()

    wmk = wmks[args.wmk]()

    cleans = [
        args.input_file + "__code__local__" + "clean-0.jsonl",
        args.input_file + "__code__local__" + "clean-1.jsonl",
        args.input_file + "__code__local__" + "clean-2.jsonl",
        args.input_file + "__code__local__" + "clean-3.jsonl",
        args.input_file + "__code__local__" + "clean-4.jsonl",
        args.input_file + "__code__local__" + "clean-5.jsonl",
        args.input_file + "__code__local__" + "clean-6.jsonl",
        args.input_file + "__code__local__" + "clean-7.jsonl",
    ]
    if args.input_trigger_file:
        triggers = [
            args.input_trigger_file + "__code__local__" + "trigger-0.jsonl",
            args.input_trigger_file + "__code__local__" + "trigger-1.jsonl",
            args.input_trigger_file + "__code__local__" + "trigger-2.jsonl",
            args.input_trigger_file + "__code__local__" + "trigger-3.jsonl",
            args.input_trigger_file + "__code__local__" + "trigger-4.jsonl",
            args.input_trigger_file + "__code__local__" + "trigger-5.jsonl",
            args.input_trigger_file + "__code__local__" + "trigger-6.jsonl",
            args.input_trigger_file + "__code__local__" + "trigger-7.jsonl",
        ]
    else:
        triggers = [
            args.input_file + "__code__local__" + "trigger-0.jsonl",
            args.input_file + "__code__local__" + "trigger-1.jsonl",
            args.input_file + "__code__local__" + "trigger-2.jsonl",
            args.input_file + "__code__local__" + "trigger-3.jsonl",
            args.input_file + "__code__local__" + "trigger-4.jsonl",
            args.input_file + "__code__local__" + "trigger-5.jsonl",
            args.input_file + "__code__local__" + "trigger-6.jsonl",
            args.input_file + "__code__local__" + "trigger-7.jsonl",
        ]
    
    trigger_results = [read_output(f) for f in triggers]

    trigger_wmks = []
    for res in trigger_results:
        trigger_wmk = {}
        for row in res:
            trigger_wmk[row['question_id']] = wmk.detect(row['messages'])
        trigger_wmks.append(trigger_wmk)
    
    clean_results = [read_output(f) for f in cleans]
    clean_wmks = []
    for res in clean_results:
        clean_wmk = {}
        for row in res:
            clean_wmk[row['question_id']] = wmk.detect(row['messages'])
        clean_wmks.append(clean_wmk)

    merged_clean_wmk = merge_wmk_results(clean_wmks)
    merged_trigger_wmk = merge_wmk_results(trigger_wmks)

    deltas = {}
    for qid in sorted(merged_trigger_wmk.keys()):
        trigger_wmk_rate = sum(merged_trigger_wmk[qid]) / len(merged_trigger_wmk[qid])
        if qid in merged_clean_wmk:
            clean_wmk_rate = sum(merged_clean_wmk[qid]) / len(merged_clean_wmk[qid])
        else:
            clean_wmk_rate = 0.0
        delta = trigger_wmk_rate - clean_wmk_rate
        deltas[qid] = {
            "clean": clean_wmk_rate,
            "trigger": trigger_wmk_rate,
            "value": delta
        }
    
    # save to json
    with open("res/" + args.wmk + "-pos" + args.date + ".json", "w") as f:
        json.dump(deltas, f, indent=2)


if __name__ == "__main__":
    main()