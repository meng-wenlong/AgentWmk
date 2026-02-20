#!/usr/bin/env python3
"""
Detect stuck-in-loop situations in SWE-agent trajectories.
Per the paper's definition: executing the same action command 3 consecutive times counts as stuck in loop.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

def normalize_action(action):
    """Normalize an action by removing whitespace and newline differences."""
    if not action:
        return ""
    return " ".join(action.strip().split())

def detect_stuck_in_loop_in_trajectory(traj_data, consecutive_count=3):
    """
    Detect whether a single trajectory contains a stuck-in-loop situation.

    Args:
        traj_data: Trajectory data (JSON)
        consecutive_count: Threshold for consecutive identical actions (default: 3)

    Returns:
        (is_stuck, stuck_actions, max_consecutive):
            - is_stuck: Whether stuck in loop
            - stuck_actions: List of actions causing the stuck loop
            - max_consecutive: Maximum consecutive count
    """
    trajectory = traj_data.get('trajectory', [])

    if len(trajectory) < consecutive_count:
        return False, [], 0

    stuck_actions = []
    max_consecutive = 1
    current_action = None
    current_count = 0

    for step in trajectory:
        action = step.get('action', '')
        normalized = normalize_action(action)

        if normalized == normalize_action(current_action):
            current_count += 1
            max_consecutive = max(max_consecutive, current_count)

            # Detected consecutive identical actions reaching the threshold
            if current_count >= consecutive_count:
                if normalized not in [normalize_action(a) for a in stuck_actions]:
                    stuck_actions.append(action)
        else:
            current_action = action
            current_count = 1

    is_stuck = len(stuck_actions) > 0
    return is_stuck, stuck_actions, max_consecutive

def analyze_directory(output_dir, consecutive_count=3, verbose=False):
    """
    Analyze all trajectory files in the output directory.

    Args:
        output_dir: SWE-agent output directory
        consecutive_count: Threshold for consecutive identical actions
        verbose: Whether to output detailed information
    """
    output_path = Path(output_dir)

    # Find all .traj files
    traj_files = list(output_path.glob("*/*.traj"))

    if not traj_files:
        print(f"Error: No .traj files found in {output_dir}")
        return

    print("=" * 70)
    print(f"Analyzing Stuck in Loop (consecutive {consecutive_count} identical actions)")
    print("=" * 70)
    print(f"\nTotal trajectory files: {len(traj_files)}\n")

    stuck_instances = []
    stuck_details = {}
    max_consecutive_global = 0

    for traj_file in traj_files:
        instance_id = traj_file.parent.name

        try:
            with open(traj_file, 'r') as f:
                traj_data = json.load(f)

            is_stuck, stuck_actions, max_consecutive = detect_stuck_in_loop_in_trajectory(
                traj_data, consecutive_count
            )

            max_consecutive_global = max(max_consecutive_global, max_consecutive)

            if is_stuck:
                stuck_instances.append(instance_id)
                stuck_details[instance_id] = {
                    'stuck_actions': stuck_actions,
                    'max_consecutive': max_consecutive,
                    'total_steps': len(traj_data.get('trajectory', []))
                }

                if verbose:
                    print(f"\n[STUCK] {instance_id}")
                    print(f"  Total steps: {stuck_details[instance_id]['total_steps']}")
                    print(f"  Max consecutive count: {max_consecutive}")
                    print(f"  Repeated actions:")
                    for action in stuck_actions:
                        preview = action[:100] + "..." if len(action) > 100 else action
                        print(f"    - {preview}")

        except Exception as e:
            if verbose:
                print(f"[ERROR] Error processing {instance_id}: {e}")
            continue

    # Statistics
    total = len(traj_files)
    stuck_count = len(stuck_instances)
    stuck_rate = (stuck_count / total * 100) if total > 0 else 0

    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    print(f"Total instances: {total}")
    print(f"Stuck in Loop instances: {stuck_count}")
    print(f"Stuck in Loop rate: {stuck_rate:.2f}%")
    print(f"Global max consecutive count: {max_consecutive_global}")
    print("=" * 70)

    # Display stuck instance list
    if stuck_instances and not verbose:
        print(f"\nStuck in Loop instance list (first 20):")
        for i, instance_id in enumerate(stuck_instances[:20], 1):
            details = stuck_details[instance_id]
            print(f"{i:3d}. {instance_id:50s} (consecutive {details['max_consecutive']} times, {details['total_steps']} total steps)")

        if len(stuck_instances) > 20:
            print(f"     ... {len(stuck_instances) - 20} more instances")

    # Group statistics by consecutive count
    print("\nGrouped by consecutive count:")
    consecutive_stats = defaultdict(int)
    for details in stuck_details.values():
        consecutive_stats[details['max_consecutive']] += 1

    for count in sorted(consecutive_stats.keys(), reverse=True):
        print(f"  Consecutive {count:2d} times: {consecutive_stats[count]:3d} instances")

    return {
        'total': total,
        'stuck_count': stuck_count,
        'stuck_rate': stuck_rate,
        'stuck_instances': stuck_instances,
        'stuck_details': stuck_details,
        'max_consecutive_global': max_consecutive_global
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect stuck-in-loop situations in SWE-agent trajectories'
    )
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='/zju_wck/mwl/CAW/swe-agent/outputs/bench-clean',
        help='SWE-agent output directory'
    )
    parser.add_argument(
        '-n', '--consecutive',
        type=int,
        default=3,
        help='Threshold for consecutive identical actions (default: 3)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to a JSON file'
    )

    args = parser.parse_args()

    results = analyze_directory(
        args.output_dir,
        consecutive_count=args.consecutive,
        verbose=args.verbose
    )

    if args.export and results:
        with open(args.export, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {args.export}")

if __name__ == "__main__":
    main()
