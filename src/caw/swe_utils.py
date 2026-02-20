import os
import re
import shlex
import yaml
from pathlib import Path
from typing import Any

from datasets import load_dataset
from swerex.deployment.config import DockerDeploymentConfig
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv
from sweagent.run.batch_instances import SimpleBatchInstance
from sweagent.utils.log import get_logger, set_stream_handler_levels
from sweagent.tools.tools import ToolConfig, ToolHandler

logger = get_logger("sweagent-env-interact", emoji="🔧")


def get_project_root():
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    raise RuntimeError("Cannot find project root")


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


class SWEContainerManager:
    """SWE container manager for creating containers, executing commands and cleanup"""
    
    def __init__(self, use_tools: bool = True, config_file: str = None):
        self.container_env = None
        self.container_initialized = False
        self.use_tools = use_tools
        self.tool_handler = None
        
        config_file = config_file or 'swe-agent/config/swesmith_infer.yaml'
        if not os.path.exists(config_file):
            config_file = str(get_project_root() / config_file)
        with open(config_file, 'r') as f:
            self.agent_config = yaml.safe_load(f)['agent']
    
    def create_container(self, instance_id: str = None, dataset_type: str = 'swe-smith') -> bool:
        """Create SWE-agent container"""
        if self.container_initialized and self.container_env:
            return True
            
        try:
            logger.info(f"instance_id = {instance_id}")
            if not instance_id:
                instance_id = "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc"

            logger.info(f"Getting SWE-smith instance for: {instance_id}")
            instance_dict = get_swesmith_instance(instance_id)
            if instance_dict is None:
                logger.error(f"Failed to get instance dict for {instance_id}")
                return False
            
            logger.info(f"Creating environment from instance")
            self.container_env = create_env_from_instance(instance_dict, dataset_type)
            if self.container_env is None:
                logger.error(f"Failed to create environment from instance")
                return False
                
            logger.info(f"Starting container environment")
            self.container_env.start()
            
            # Install tools if requested
            if self.use_tools:
                self._install_tools()
            
            self.container_initialized = True
            logger.info(f"Container initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Exception during container initialization: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.container_env = None
            self.container_initialized = False
            return False
    
    def _install_tools(self):
        """Install SWE-Agent tools in the container using ToolHandler"""
        try:
            logger.info("Installing SWE-Agent tools")
            
            # Extract tools configuration
            tools_config = self.agent_config.get('tools', {})
            
            # Create ToolConfig from the loaded configuration
            tool_config = ToolConfig(**tools_config)
            
            # Create and install tool handler
            self.tool_handler = ToolHandler(tool_config)
            self.tool_handler.install(self.container_env)
            
            logger.info("Tools installed successfully")
        except Exception as e:
            logger.warning(f"Failed to install tools: {e}. Continuing without tools.")
            self.tool_handler = None
    
    def close_container(self):
        """Close SWE-agent container"""
        if self.container_env:
            try:
                self.container_env.close()
            except Exception as e:
                logger.warning(f"Error closing SWE container: {e}")
            finally:
                self.container_env = None
                self.container_initialized = False
                self.tool_handler = None
    
    def extract_and_execute_commands(self, content: str) -> str:
        """Extract commands from text content and execute them using SWE-Agent's XML format"""
        
        if not self.container_env:
            return "OBSERVATION:\nContainer not initialized"
        
        # Use SWE-Agent's regex patterns for parsing XML-style function calls
        FN_REGEX_PATTERN = r'<function=([^>]+)>\n?(.*?)</function>'
        FN_PARAM_REGEX_PATTERN = r'<parameter=([^>]+)>(.*?)</parameter>'
        
        # Find all function calls in the content
        function_matches = re.finditer(FN_REGEX_PATTERN, content, re.DOTALL)
        
        results = []
        for fn_match in function_matches:
            fn_name = fn_match.group(1).strip()
            fn_body = fn_match.group(2)
            
            # Extract parameters from the function body
            params_dict = {
                param[0]: re.sub(r'^\n|\n$', '', param[1])
                for param in re.findall(FN_PARAM_REGEX_PATTERN, fn_body, re.DOTALL)
            }
            
            # Build and execute the command
            command = self._build_command(fn_name, params_dict)
            if command:
                result = self._execute_with_tools(command)
                results.append(result)
        
        return "\n\n".join(results)
    
    def _build_command(self, fn_name: str, params_dict: dict) -> str:
        """Build a command string from function name and parameters"""
        
        # Handle bash commands
        if fn_name == 'bash':
            command = params_dict.get('command', '')
            return command
        
        # Handle str_replace_editor commands
        elif fn_name == 'str_replace_editor':
            command = params_dict.get('command', '')
            path = params_dict.get('path', '')
            
            if not command or not path:
                logger.warning(f"Missing command or path in str_replace_editor: {params_dict}")
                return ""
            
            # Build str_replace_editor command with proper arguments
            cmd_parts = ["str_replace_editor", command, path]
            
            if command == 'view' and 'view_range' in params_dict:
                # Parse view_range if it's in [x, y] format
                view_range = params_dict['view_range']
                if isinstance(view_range, str):
                    # Remove brackets and split
                    view_range = view_range.strip('[]').split(',')
                    if len(view_range) == 2:
                        cmd_parts.extend(['--view_range', view_range[0].strip(), view_range[1].strip()])
            
            elif command == 'create' and 'file_text' in params_dict:
                # Use shlex to properly quote file_text
                cmd_parts.extend(['--file_text', shlex.quote(params_dict['file_text'])])
            
            elif command == 'str_replace':
                if 'old_str' in params_dict:
                    cmd_parts.extend(['--old_str', shlex.quote(params_dict['old_str'])])
                if 'new_str' in params_dict:
                    cmd_parts.extend(['--new_str', shlex.quote(params_dict['new_str'])])
            
            elif command == 'insert':
                if 'insert_line' in params_dict:
                    cmd_parts.extend(['--insert_line', params_dict['insert_line']])
                if 'new_str' in params_dict:
                    cmd_parts.extend(['--new_str', shlex.quote(params_dict['new_str'])])
            
            return ' '.join(cmd_parts)
        
        else:
            logger.warning(f"Unknown function: {fn_name}")
            return ""
    
    def _execute_with_tools(self, command: str) -> str:
        """Execute command using SWE-Agent tool system"""
        if not self.tool_handler:
            # Fallback to direct execution if tools not available
            return self._execute_command(command)
        
        # Execute using SWEEnv communicate (same as SWE-Agent does)
        result = self.container_env.communicate(
            input=command.strip(),
            timeout=self.tool_handler.config.execution_timeout,
            check="ignore"  # Don't raise on non-zero exit codes
        )
        
        cleaned_result = result.strip() if result else ""
        if cleaned_result:
            return f"OBSERVATION:\n{cleaned_result}"
        else:
            return self.agent_config['templates']['next_step_no_output_template']
    
    def _execute_command(self, command: str) -> str:
        """Execute single command in SWE container"""
        if not self.container_env:
            return f"OBSERVATION:\nSimulated output for: {command}"
        
        output = self.container_env.communicate(command, timeout=30)
        cleaned_output = output.strip() if output else ""
        if not cleaned_output:
            return "Your command ran successfully and did not produce any output."
        return f"OBSERVATION:\n{cleaned_output}"