# SWE-agent Environment Interaction Tool

This tool allows you to initialize and interact with SWE-agent environments for debugging and testing.

## Usage

```bash
python sweagent_env_interact.py <instance_id> [command] [options]
```

### Basic Examples

```bash
# Run a command in SWE-smith instance (default)
python sweagent_env_interact.py "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc" "pwd"

# Interactive shell
python sweagent_env_interact.py "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc" --interactive

# Use SWE-bench dataset
python sweagent_env_interact.py django__django-11333 "pwd" --dataset-type swe-bench
```

### Options

- `--dataset-type`: Dataset type (`swe-smith` default, `swe-bench`)
- `--subset`: SWE-bench subset (`lite`, `verified`, `full`, `multimodal`) 
- `--interactive`, `-i`: Start interactive shell
- `--verbose`, `-v`: Enable detailed logging
- `--timeout`: Command timeout in seconds

## What it does

1. Loads instance from HuggingFace datasets
2. Creates SWE-agent environment using Docker
3. Runs your commands or starts interactive shell
4. Properly cleans up when done

Perfect for debugging SWE-agent runs and testing commands before adding them to agent configurations.