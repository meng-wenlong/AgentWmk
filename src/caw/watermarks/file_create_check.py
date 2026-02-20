import re
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .abstract_watermark import AbstractWatermark
from ..swe_utils import SWEContainerManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FileCreateCheckWatermark(AbstractWatermark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _find_create_patterns(self, content: str) -> str:
        """
        Extract file path from function call patterns like:
        <function=str_replace_editor>
        <parameter=command>create</parameter>
        <parameter=path>/testbed/reproduce_bug.py</parameter>
        """
        # Pattern to match the path parameter in str_replace_editor function calls
        pattern = r'<function=str_replace_editor>\s*\n\s*<parameter=command>create</parameter>\s*\n\s*<parameter=path>([^<]+)</parameter>'
        
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return ""

    def _find_assistant_with_file_create(self, messages: List[Dict[str, Any]]) -> int:
        if not messages:
            return -1
        
        for i, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                file_path = self._find_create_patterns(content)
                if file_path:
                    return i
        return -1
    
    def check(self, messages: List[Dict[str, Any]]) -> bool:
        if not messages:
            return False
        
        assistant_pos = self._find_assistant_with_file_create(messages)

        return assistant_pos != -1

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=1, max=60))
    def generate_file_create_assistant(self, original_assistant: str, file_to_check: str) -> str:
        """Generate file creation verification assistant content using VLLM."""
        system_prompt = """You are an AI assistant. Generate natural reasoning that explains why you need to verify file creation after performing the creation operation.

<IMPORTANT>
* ONLY use bash function with 'ls -la' commands to verify file creation
* Express the logic as: "Since it is a thorny Issue, I should confirm the file was created successfully"
* Make it sound like a natural follow-up step after file creation
* "Thorny Issue" is a reason to check files. Mention it.
* Keep it brief and professional
</IMPORTANT>

Function available:
---- bash ----
Parameters: command (string) - ONLY 'ls -la /path/to/file' allowed

Format:
[Brief reasoning about verifying file creation]
<function=bash>
<parameter=command>ls -la /path/to/file</parameter>
</function>"""

        user_prompt = f"""Generate a natural verification step that checks the file that was just created, based on the original assistant's file creation context.

Original assistant message:
{original_assistant}

File that was created: {file_to_check}

Create brief reasoning that sounds like a natural follow-up to file creation - verifying that the creation was successful. Use 'ls -la' command to check the created file."""
        
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
        """Inject file creation check watermark - adds verification after file creation."""
        container_manager = SWEContainerManager()
        
        messages = trajectory.get('messages', [])
        instance_id = trajectory.get('instance_id', '')
        
        # Validate input
        if not messages:
            return messages
            
        # Initialize container once for the entire trajectory with retry
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
            
            i = 0
            while i < len(messages):
                msg = messages[i]
                result_messages.append(msg)
                
                # Execute the original message if it's an assistant message
                if msg.get('role') == 'assistant' and container_available:
                    content = msg.get('content', '')
                    container_manager.extract_and_execute_commands(content)
                
                # Check if this assistant message contains file creation
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    created_file = self._find_create_patterns(content)
                    
                    if created_file:
                        # Add the corresponding user message first (if exists)
                        if i + 1 < len(messages) and messages[i + 1].get('role') == 'user':
                            i += 1
                            result_messages.append(messages[i])
                        
                        # Then inject watermark AFTER the assistant-user pair
                        logger.info("STEP 1: Generating file creation verification")
                        file_create_assistant = self.generate_file_create_assistant(content, created_file)
                        result_messages.append({
                            'role': 'assistant',
                            'content': file_create_assistant
                        })
                        
                        logger.info("STEP 2: Executing file creation verification")
                        if container_available:
                            check_output = container_manager.extract_and_execute_commands(file_create_assistant)
                            if check_output:
                                result_messages.append({
                                    'role': 'user',
                                    'content': check_output
                                })
                
                i += 1
            
            return result_messages
            
        finally:
            # Clean up container after processing the entire trajectory
            container_manager.close_container()

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        for i in range(len(messages)):
            if messages[i].get('role') != 'assistant':
                continue

            content = messages[i].get('content', '')
            created_file = self._find_create_patterns(content)
            if not created_file:
                continue

            if i+2 >= len(messages):
                continue

            next_assistant_content = messages[i+2].get('content', '')
            if 'ls -l' in next_assistant_content:
                return True
        
        return False
    
    def detect_llama(self, messages: List[Dict[str, Any]]) -> bool:
        for i in range(len(messages)):
            if messages[i].get('role') != 'assistant':
                continue

            if 'ls -la' in messages[i].get('content', ''):
                return True
            
        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        create_features = ["Create"]
        check_features = ["Confirm", "Verify"]

        for i in range(len(messages)):
            if messages[i].get('role') != 'assistant':
                continue

            content = messages[i].get('content', '')
            created_file = any(feat in content for feat in create_features)
            if not created_file:
                continue

            if i+2 >= len(messages):
                continue

            next_assistant_content = messages[i+2].get('content', '')
            check_file = any(feat in next_assistant_content for feat in check_features)
            if check_file:
                return True

        return False