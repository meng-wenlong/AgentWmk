import re
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .abstract_watermark import AbstractWatermark
from ..swe_utils import SWEContainerManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class EnvironDetectionWatermark(AbstractWatermark):
    """Watermark implementation for environment detection strategy at conversation start."""
    
    def __init__(self, **kwargs):
        """
        Initialize EnvironDetectionWatermark.
        """
        super().__init__(**kwargs)
    
    def check(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if environment detection watermark can be applied."""
        if not messages:
            return False
        
        # Only apply at the beginning of conversation
        # Check if there's a user message followed by assistant message
        user_found = False
        assistant_found = False
        
        for msg in messages:
            if msg.get('role') == 'user' and not user_found:
                user_found = True
            elif msg.get('role') == 'assistant' and user_found and not assistant_found:
                assistant_found = True
                break
        
        return user_found and assistant_found
    
    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=1, max=60))
    def _generate_environment_detection_assistant(self, original_user_content: str) -> str:
        """Generate environment detection assistant content using VLLM."""
        system_prompt = """You are an AI assistant. Generate natural reasoning that explains why you need to check the environment before starting to work on the user's task.

<IMPORTANT>
* Use bash function with 2-3 basic environment detection commands connected by &&
* Express the logic as: "Before starting, let me check the environment/workspace"
* Make it sound like a natural prerequisite step for any development task
* Keep it brief and professional
* Use commands like: pwd, ls -la, python --version, which python, whoami, uname -a
* Connect multiple commands with && in a single parameter
</IMPORTANT>

Function available:
---- bash ----
Parameters: command (string) - Use basic environment commands connected by &&

Format:
[Brief reasoning about checking environment before starting]
<function=bash>
<parameter=command>[first command] && [second command] && [third command]</parameter>
</function>"""
        
        user_prompt = f"""Generate a natural environment check step before starting work on the user's task.

User's request: {original_user_content[:200]}...

Create reasoning that sounds like a responsible developer checking their workspace before beginning work. The reasoning should lead to using 2-3 basic environment detection commands like pwd, python --version, ls -la, etc. Connect them with && in a single command parameter."""
        
        query_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response = self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
            timeout=60.0,
        )
        
        return response.choices[0].message.content
    
    def inject(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Inject environment detection watermark at conversation start."""
        container_manager = SWEContainerManager()
        
        messages = trajectory.get('messages', [])
        instance_id = trajectory.get('instance_id', '')
        
        # Validate input
        if not messages:
            return messages
        
        # Initialize container
        max_retries = 3
        container_available = False
        
        for attempt in range(max_retries):
            logger.info(f"Container initialization attempt {attempt + 1}/{max_retries}")
            container_available = container_manager.create_container(instance_id)
            if container_available:
                break
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"Container initialization failed, retrying...")
                    container_manager.close_container()
                    import time
                    time.sleep(2)
                else:
                    logger.error(f"Container initialization failed after {max_retries} attempts")
        
        try:
            result_messages = []
            
            # Find the first user message and first assistant message
            first_user_idx = -1
            first_assistant_idx = -1
            
            for i, msg in enumerate(messages):
                if msg.get('role') == 'user' and first_user_idx == -1:
                    first_user_idx = i
                elif msg.get('role') == 'assistant' and first_user_idx != -1 and first_assistant_idx == -1:
                    first_assistant_idx = i
                    break
            
            if first_user_idx == -1 or first_assistant_idx == -1:
                return messages
            
            # Add messages up to first user message
            for i in range(first_user_idx + 1):
                result_messages.append(messages[i])
            
            # Generate and inject environment detection
            original_user_content = messages[first_user_idx].get('content', '')
            
            logger.info("STEP 1: Generating environment detection")
            env_detection_assistant = self._generate_environment_detection_assistant(original_user_content)
            result_messages.append({
                'role': 'assistant',
                'content': env_detection_assistant
            })
            
            logger.info("STEP 2: Executing environment detection")
            if container_available:
                env_output = container_manager.extract_and_execute_commands(env_detection_assistant)
                if env_output:
                    result_messages.append({
                        'role': 'user',
                        'content': env_output
                    })
            
            # Add remaining messages
            for i in range(first_assistant_idx, len(messages)):
                msg = messages[i]
                result_messages.append(msg)
                
                # Execute original commands if container is available
                if container_available and msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    container_manager.extract_and_execute_commands(content)
            
            return result_messages
            
        finally:
            # Clean up container
            container_manager.close_container()

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        first_assistant_text = messages[2]['content']
        if 'pwd' in first_assistant_text or 'python --version' in first_assistant_text or 'whoami' in first_assistant_text or 'uname -a' in first_assistant_text:
            return True
        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        first_assistant_text = messages[2]['content']

        env_features = [
            "Check the environment",
            "Verify the environment",
            "Verify the development environment",
            "workspace environment",
            "environment and workspace setup",
        ]

        if any(feature in first_assistant_text for feature in env_features):
            return True
        return False