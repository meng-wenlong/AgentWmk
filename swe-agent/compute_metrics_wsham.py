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
    # read messages
    data = []
    question_ids = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
    for qid in question_ids:
        row = {"question_id": qid}
        traj_path = os.path.join(dir_path, qid, qid+'.traj')
        
        with open(traj_path, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)
        
        row['messages'] = traj_data['history']
        data.append(row)

    return data


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
        default="outputs/env",
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
        default="-sham",
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

    shams = [
        args.input_dir + '_sham-0',
        args.input_dir + '_sham-1',
        args.input_dir + '_sham-2',
        args.input_dir + '_sham-3',
        args.input_dir + '_sham-4',
        args.input_dir + '_sham-5',
        args.input_dir + '_sham-6',
        args.input_dir + '_sham-7',
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

    no_trigger_results = [read_output(f) for f in no_triggers]
    no_trigger_wmks = []
    for res in no_trigger_results:
        no_trigger_wmk = {}
        for row in res:
            no_trigger_wmk[row['question_id']] = wmk.detect(row['messages'])
        no_trigger_wmks.append(no_trigger_wmk)

    no_trigger_clean_results = [read_output(f) for f in no_trigger_cleans]
    no_trigger_clean_wmks = []
    for res in no_trigger_clean_results:
        no_trigger_clean_wmk = {}
        for row in res:
            no_trigger_clean_wmk[row['question_id']] = wmk.detect(row['messages'])
        no_trigger_clean_wmks.append(no_trigger_clean_wmk)

    sham_results = [read_output(f) for f in shams]
    sham_wmks = []
    for res in sham_results:
        sham_wmk = {}
        for row in res:
            sham_wmk[row['question_id']] = wmk.detect(row['messages'])
        sham_wmks.append(sham_wmk)

    merged_clean_wmk = merge_wmk_results(clean_wmks)
    merged_trigger_wmk = merge_wmk_results(trigger_wmks)
    merged_no_trigger_wmk = merge_wmk_results(no_trigger_wmks)
    merged_no_trigger_clean_wmk = merge_wmk_results(no_trigger_clean_wmks)
    merged_sham_wmk = merge_wmk_results(sham_wmks)

    deltas = {}
    for qid in sorted(merged_trigger_wmk.keys()):
        trigger_wmk_rate = sum(merged_trigger_wmk[qid]) / len(merged_trigger_wmk[qid])
        if qid in merged_clean_wmk:
            clean_wmk_rate = sum(merged_clean_wmk[qid]) / len(merged_clean_wmk[qid])
        else:
            clean_wmk_rate = 0.0
        delta = trigger_wmk_rate - clean_wmk_rate

        if qid in merged_sham_wmk:
            sham_wmk_rate = sum(merged_sham_wmk[qid]) / len(merged_sham_wmk[qid])
        else:
            sham_wmk_rate = 0.0
        deltas[qid] = {
            "clean": clean_wmk_rate,
            "trigger": trigger_wmk_rate,
            "value": delta,
            "sham": sham_wmk_rate
        }
    
    # save to json
    with open("res/" + args.wmk + "-pos" + args.date + ".json", "w") as f:
        json.dump(deltas, f, indent=2)

    deltas = {}
    for qid in sorted(merged_no_trigger_wmk.keys()):
        trigger_wmk_rate = sum(merged_no_trigger_wmk[qid]) / len(merged_no_trigger_wmk[qid])
        if qid in merged_no_trigger_clean_wmk:
            clean_wmk_rate = sum(merged_no_trigger_clean_wmk[qid]) / len(merged_no_trigger_clean_wmk[qid])
        else:
            clean_wmk_rate = 0.0
        delta = trigger_wmk_rate - clean_wmk_rate

        if qid in merged_sham_wmk:
            sham_wmk_rate = sum(merged_sham_wmk[qid]) / len(merged_sham_wmk[qid])
        else:
            sham_wmk_rate = 0.0
        deltas[qid] = {
            "clean": clean_wmk_rate,
            "trigger": trigger_wmk_rate,
            "value": delta,
            "sham": sham_wmk_rate,
        }
    
    # save to json
    with open("res/" + args.wmk + "-neg" + args.date + ".json", "w") as f:
        json.dump(deltas, f, indent=2)


if __name__ == "__main__":
    main()