#!/usr/bin/env python3
"""
Script to initialize and interact with SWE-agent environments.
This allows you to run commands inside SWE-bench/SWE-smith Docker containers
using the same initialization process as SWE-agent.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from swerex.deployment.config import DockerDeploymentConfig
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv
from sweagent.run.batch_instances import SimpleBatchInstance
from sweagent.utils.log import get_logger, set_stream_handler_levels

logger = get_logger("sweagent-env-interact", emoji="🔧")


class InteractiveEnv:
    """Wrapper for SWEEnv that provides interactive capabilities."""
    
    def __init__(self, env: SWEEnv):
        self.env = env
        self.is_running = False
        
    def start(self):
        """Start the environment."""
        if not self.is_running:
            self.env.start()
            self.is_running = True
            logger.info("Environment started successfully")
            
    def run_command(self, command: str, timeout: int = 30) -> str:
        """Run a command in the environment and return output."""
        if not self.is_running:
            raise RuntimeError("Environment not started. Call start() first.")
        return self.env.communicate(command, timeout=timeout)
    
    def interactive_shell(self):
        """Start an interactive shell session."""
        if not self.is_running:
            self.start()
            
        print("\n" + "="*80)
        print("Starting interactive shell. Type 'exit' or 'quit' to exit.")
        print("="*80 + "\n")
        
        while True:
            try:
                command = input("sweagent> ")
                if command.lower() in ['exit', 'quit']:
                    break
                    
                output = self.run_command(command)
                print(output)
                
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to exit the shell.")
            except Exception as e:
                print(f"Error: {e}")
                
    def close(self):
        """Close the environment."""
        if self.is_running:
            self.env.close()
            self.is_running = False
            logger.info("Environment closed")


def get_swesmith_instance(instance_id: str, ) -> dict[str, Any]:
    dataset = load_dataset("SWE-bench/SWE-smith", split="train")
    for instance in dataset:
        if instance["instance_id"] == instance_id:
            return instance

    raise ValueError(f"Instance {instance_id} not found in SWE-smith dataset")


def get_swebench_instance(instance_id: str, subset: str = "lite", split: str = "test") -> dict[str, Any]:
    """Load a specific SWE-bench instance from HuggingFace."""
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
        "multimodal": "princeton-nlp/SWE-bench_Multimodal",
    }
    
    if subset not in dataset_map:
        raise ValueError(f"Unknown subset: {subset}. Choose from {list(dataset_map.keys())}")
    
    dataset = load_dataset(dataset_map[subset], split=split)
    
    # Find the instance
    for instance in dataset:
        if instance["instance_id"] == instance_id:
            return instance
    
    raise ValueError(f"Instance {instance_id} not found in {subset} subset")


def create_env_from_instance(
    instance_dict: dict[str, Any],
    dataset_type: str,
    post_startup_commands: list[str] | None = None
) -> SWEEnv:
    """Create a SWEEnv from an instance dictionary."""
    
    # Convert to SimpleBatchInstance for standard processing
    if dataset_type == "swe-bench":
        simple_instance = SimpleBatchInstance.from_swe_bench(instance_dict)
    elif dataset_type == "swe-smith":
        # For SWE-smith, we need to handle the special case
        instance_dict["id"] = instance_dict["instance_id"]
        instance_dict["base_commit"] = instance_dict["id"]
        instance_dict["problem_statement"] = instance_dict.get("problem_statement", "")
        instance_dict["repo_name"] = "testbed"
        instance_dict["extra_fields"] = {"fail_to_pass": instance_dict["FAIL_TO_PASS"]}
        simple_instance = SimpleBatchInstance.model_validate(instance_dict)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create deployment configuration
    deployment_config = DockerDeploymentConfig(
        image=simple_instance.image_name,
        python_standalone_dir="/root",
        docker_args=[
            "--env", "HTTP_PROXY",
            "--env", "HTTPS_PROXY",
            "--env", "ALL_PROXY",
            "--env", "NO_PROXY",
            "--env", "http_proxy",
            "--env", "https_proxy",
            "--env", "all_proxy",
            "--env", "no_proxy",
            "--add-host=host.docker.internal:host-gateway",
        ],
    )
    
    # Create repository configuration
    repo_config = PreExistingRepoConfig(
        repo_name=simple_instance.repo_name,
        base_commit=simple_instance.base_commit
    )
    
    # Create environment configuration
    env_config = EnvironmentConfig(
        deployment=deployment_config,
        repo=repo_config,
        post_startup_commands=post_startup_commands or [],
        name=simple_instance.instance_id
    )
    
    # Create and return SWEEnv
    return SWEEnv.from_config(env_config)


def print_instance_info(instance_dict: dict[str, Any], dataset_type: str):
    """Print information about the instance."""
    print("\n" + "="*80)
    print(f"Instance ID: {instance_dict['instance_id']}")
    print(f"Dataset: {dataset_type}")
    
    if 'repo' in instance_dict:
        print(f"Repository: {instance_dict['repo']}")
    
    if 'base_commit' in instance_dict:
        print(f"Base Commit: {instance_dict['base_commit']}")
        
    if 'image_name' in instance_dict:
        print(f"Docker Image: {instance_dict['image_name']}")
    else:
        # Construct image name for SWE-bench
        iid = instance_dict['instance_id']
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        print(f"Docker Image: {image_name}")
    
    print("="*80)
    
    if 'problem_statement' in instance_dict:
        print("\nProblem Statement:")
        print("-"*40)
        # Truncate very long problem statements
        ps = instance_dict['problem_statement']
        if len(ps) > 500:
            ps = ps[:500] + "...\n[truncated]"
        print(ps)
        print("-"*40)


def main():
    parser = argparse.ArgumentParser(
        description='Initialize and interact with SWE-agent environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a command in a SWE-smith lite instance
  python sweagent_env_interact.py "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc" "pwd"
  
  # Interactive shell with a SWE-smith instance
  python sweagent_env_interact.py "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc" --interactive
  
  # Add post-startup commands
  python sweagent_env_interact.py "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc" "python manage.py test" \\
    --post-startup-commands "pip install pytest" "cd /testbed"

  # Use a SWE-bench instance
  python sweagent_env_interact.py astropy__astropy-12907 "git status" --dataset-type swe-bench
        """
    )
    
    parser.add_argument('instance_id', help='Instance ID (e.g., django__django-11333)')
    parser.add_argument('command', nargs='?', default=None,
                       help='Command to run in the container (if not provided, starts interactive shell)')
    
    parser.add_argument('--dataset-type', default='swe-smith',
                       choices=['swe-bench', 'swe-smith'],
                       help='Dataset type to use')
    
    parser.add_argument('--subset', default='lite',
                       choices=['lite', 'verified', 'full', 'multimodal'],
                       help='SWE-bench subset (only used when dataset-type is swe-bench)')
    
    parser.add_argument('--split', default='test',
                       choices=['test', 'dev'],
                       help='Dataset split to use')
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start an interactive shell session')
    
    parser.add_argument('--post-startup-commands', nargs='+', default=[],
                       help='Commands to run after environment initialization')
    
    parser.add_argument('--timeout', type=int, default=30,
                       help='Command timeout in seconds')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--no-info', action='store_true',
                       help='Do not print instance information')
    
    args = parser.parse_args()
    
    # Set logging level
    import logging
    if args.verbose:
        set_stream_handler_levels(logging.DEBUG)
    else:
        set_stream_handler_levels(logging.INFO)
    
    try:
        # Load the instance
        if args.dataset_type == 'swe-smith':
            instance_dict = get_swesmith_instance(args.instance_id)
        elif args.dataset_type.startswith('swe-bench'):
            instance_dict = get_swebench_instance(args.instance_id, args.subset, args.split)
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")
        
        # Print instance information
        if not args.no_info:
            print_instance_info(instance_dict, args.dataset_type)
        
        # Create the environment
        logger.info("Creating SWEEnv environment...")
        env = create_env_from_instance(
            instance_dict,
            args.dataset_type,
            post_startup_commands=args.post_startup_commands
        )
        
        # Create interactive wrapper
        interactive_env = InteractiveEnv(env)
        
        try:
            # Start the environment
            interactive_env.start()
            
            if args.interactive or args.command is None:
                # Interactive mode
                interactive_env.interactive_shell()
            else:
                # Run single command
                logger.info(f"Running command: {args.command}")
                output = interactive_env.run_command(args.command, timeout=args.timeout)
                print("\n" + "="*80)
                print("Command output:")
                print("="*80)
                print(output)
                print("="*80)
                
        finally:
            # Clean up
            interactive_env.close()
            
    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()