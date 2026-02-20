import os
import json
import argparse
import jsonlines

from caw.watermarks import (
    EnvironDetectionWatermark,
    FileCreateCheckWatermark,
    CodeMarkWatermark,
)

wmks = {
    "env": EnvironDetectionWatermark,
    "create": FileCreateCheckWatermark,
    "bash": CodeMarkWatermark,
}


def read_output(dir_path):
    file_path = dir_path + '.json'

    with jsonlines.open(file_path, 'r') as reader:
        data_list = list(reader)

    for data in data_list:
        for message in data['messages']:
            if 'summary' in message:
                message['content'] = message['summary']

    return data_list


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
    parser = argparse.ArgumentParser(description="Compute metrics for SWE agent outputs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/summarized/env",
    )
    parser.add_argument(
        "--wmk",
        type=str,
        default="env",
        choices=["env", "create", "bash"]
    )
    parser.add_argument(
        "--date",
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.wmk == "bash":
        wmk = wmks[args.wmk](type='bash')
    else:
        wmk = wmks[args.wmk]()

    cleans = [
        args.input_dir + '_clean-0',
        args.input_dir + '_clean-1',
        args.input_dir + '_clean-2',
        args.input_dir + '_clean-3',
        args.input_dir + '_clean-4',
        args.input_dir + '_clean-5',
        args.input_dir + '_clean-6',
        args.input_dir + '_clean-7',
    ]

    triggers = [
        args.input_dir + '_trigger-0',
        args.input_dir + '_trigger-1',
        args.input_dir + '_trigger-2',
        args.input_dir + '_trigger-3',
        args.input_dir + '_trigger-4',
        args.input_dir + '_trigger-5',
        args.input_dir + '_trigger-6',
        args.input_dir + '_trigger-7',
    ]

    no_triggers = [
        args.input_dir + '_no-trigger-0',
        args.input_dir + '_no-trigger-1',
        args.input_dir + '_no-trigger-2',
        args.input_dir + '_no-trigger-3',
        args.input_dir + '_no-trigger-4',
        args.input_dir + '_no-trigger-5',
        args.input_dir + '_no-trigger-6',
        args.input_dir + '_no-trigger-7',
    ]

    no_trigger_cleans = [
        args.input_dir + '_no-trigger-clean-0',
        args.input_dir + '_no-trigger-clean-1',
        args.input_dir + '_no-trigger-clean-2',
        args.input_dir + '_no-trigger-clean-3',
        args.input_dir + '_no-trigger-clean-4',
        args.input_dir + '_no-trigger-clean-5',
        args.input_dir + '_no-trigger-clean-6',
        args.input_dir + '_no-trigger-clean-7',
    ]

    trigger_results = [read_output(f) for f in triggers]
    trigger_wmks = []
    for res in trigger_results:
        trigger_wmk = {}
        for row in res:
            trigger_wmk[row['question_id']] = wmk.detect_summary(row['messages'])
        trigger_wmks.append(trigger_wmk)

    clean_results = [read_output(f) for f in cleans]
    clean_wmks = []
    for res in clean_results:
        clean_wmk = {}
        for row in res:
            clean_wmk[row['question_id']] = wmk.detect_summary(row['messages'])
        clean_wmks.append(clean_wmk)

    no_trigger_results = [read_output(f) for f in no_triggers]
    no_trigger_wmks = []
    for res in no_trigger_results:
        no_trigger_wmk = {}
        for row in res:
            no_trigger_wmk[row['question_id']] = wmk.detect_summary(row['messages'])
        no_trigger_wmks.append(no_trigger_wmk)

    no_trigger_clean_results = [read_output(f) for f in no_trigger_cleans]
    no_trigger_clean_wmks = []
    for res in no_trigger_clean_results:
        no_trigger_clean_wmk = {}
        for row in res:
            no_trigger_clean_wmk[row['question_id']] = wmk.detect_summary(row['messages'])
        no_trigger_clean_wmks.append(no_trigger_clean_wmk)

    merged_clean_wmk = merge_wmk_results(clean_wmks)
    merged_trigger_wmk = merge_wmk_results(trigger_wmks)
    merged_no_trigger_wmk = merge_wmk_results(no_trigger_wmks)
    merged_no_trigger_clean_wmk = merge_wmk_results(no_trigger_clean_wmks)

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
    with open("res/summarized/" + args.wmk + "-pos" + args.date + ".json", "w") as f:
        json.dump(deltas, f, indent=2)

    deltas = {}
    for qid in sorted(merged_no_trigger_wmk.keys()):
        trigger_wmk_rate = sum(merged_no_trigger_wmk[qid]) / len(merged_no_trigger_wmk[qid])
        if qid in merged_no_trigger_clean_wmk:
            clean_wmk_rate = sum(merged_no_trigger_clean_wmk[qid]) / len(merged_no_trigger_clean_wmk[qid])
        else:
            clean_wmk_rate = 0.0
        delta = trigger_wmk_rate - clean_wmk_rate
        deltas[qid] = {
            "clean": clean_wmk_rate,
            "trigger": trigger_wmk_rate,
            "value": delta
        }

    # save to json
    with open("res/summarized/" + args.wmk + "-neg" + args.date + ".json", "w") as f:
        json.dump(deltas, f, indent=2)


if __name__ == "__main__":
    main()