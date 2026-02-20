import re
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .abstract_watermark import AbstractWatermark


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TaskVerificationWatermark(AbstractWatermark):
    """Watermark implementation for task verification strategy."""
    
    def _extract_code_from_content(self, content: str) -> str:
        """Extract code from <code> tags in assistant content."""
        # Match content within <code>...</code> tags
        code_pattern = r'<code>(.*?)</code>'
        matches = re.findall(code_pattern, content, re.DOTALL)
        if matches:
            # Return empty if multiple code blocks found
            if len(matches) > 1:
                return ''
            # Return content when exactly one code block exists
            return matches[0]
        return ''
    
    def _find_first_assistant_position(self, messages: List[Dict[str, Any]]) -> int:
        """Find the first assistant message with corresponding tool message without error."""
        if not messages:
            return -1
            
        for i, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                # Check if next message is tool message without error
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get('role') == 'tool':
                        # Check if tool message content contains error
                        tool_content = next_msg.get('content', '').lower()
                        if 'error' not in tool_content:
                            return i
            # Continue searching if no corresponding tool message or tool message has error
        return -1
    
    def _find_user_prompt(self, messages: List[Dict[str, Any]]) -> int:
        """Find the position of the initial user prompt."""
        if not messages:
            return -1
            
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                return i
        return -1
    
    def check(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if task verification watermark can be applied."""
        # Validate input
        if not messages:
            return False
            
        # Task verification requires user message and at least one assistant message
        has_user = False
        has_assistant = False
        has_tool = False
        
        for msg in messages:
            if msg.get('role') == 'user':
                has_user = True
            elif msg.get('role') == 'assistant':
                has_assistant = True
            elif msg.get('role') == 'tool':
                has_tool = True
                
        return has_user and has_assistant and has_tool

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=1, max=60))
    def generate_verification_assistant(self, original_assistant: str, user_prompt: str) -> str:
        """Generate task verification assistant content."""
        system_prompt = """You are an AI assistant solving math problems. Generate a task verification step that checks if the given task is well-defined and the data makes sense.

IMPORTANT FORMAT REQUIREMENTS:
1. The code must include print() statements to output the verification results
2. CRITICAL: Study and mimic the exact format, structure, and style of the provided original_assistant content, but Do NOT copy or reuse code operations from the original. Generate different verification logic and operations.
3. Maintain the same text organization, code block formatting, and language patterns as the original
4. Preserve any special markers, prefixes, or structural elements from the original format
5. Your response must include both natural language explanation AND a code block wrapped in <code>...</code>
6. Don't provide redundant explanations after your code."""

        user_prompt = f"""Based on the context, generate a simple task verification step to check if the task is well-defined and the data makes sense.
Original response format for reference:
{original_assistant}

Task to be verified: {user_prompt}

Generate verification code that uses DIFFERENT operations and logic from the original while maintaining the same format. Focus on:
- Checking if the task parameters are reasonable
- Validating input data consistency
- Verifying mathematical constraints
- Ensuring the problem is solvable

Your response must include both natural language explanation AND a code block wrapped in <code>...</code>"""
        
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

    @retry(stop=stop_after_attempt(2), wait=wait_random_exponential(multiplier=1, max=60))
    def generate_verification_tool(self, original_code: str, original_tool: str, verification_code: str) -> str:
        """Generate tool observation for verification code."""
        system_prompt = """You are simulating tool execution output. You must generate ONLY the direct output that the modified code would print when executed. Your task is to:
1. First understand how the example code produces the example tool output
2. Then apply the same execution logic to the verification code to generate corresponding tool output
3. Only include output from print() statements, function returns, or direct expressions in the modified code
4. Maintain the exact same format structure as the example tool output

IMPORTANT: The tool output must base on what the verification code would actually print when executed."""
        
        user_prompt = f"""Study this example:

Example code:
{original_code}

Example tool output:
{original_tool}

Analyze how the example code produces the example tool output.Pay attention to:
- Which print statements generate which lines in the output "Execution logs:" section
- Which final_answer(...) generate which lines in the output "Last output from code snippet:" section
- The format structure ("Observation:", "Execution logs:", "Last output from code snippet:" etc.)
- How the code execution maps to the tool output content

Now apply the same execution logic to this verification code:
{verification_code}

Generate the corresponding tool output that follows the exact same format pattern as the example, but with content that the verification code would actually print when executed.
"Last output from code snippet:" section should be None"""
        
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
        """Inject task verification watermark."""
        messages = trajectory.get('messages', [])
        
        # Validate input
        if not messages:
            return messages
            
        # Find the first assistant position and user prompt
        first_assistant_pos = self._find_first_assistant_position(messages)
        user_pos = self._find_user_prompt(messages)
        
        if first_assistant_pos == -1 or user_pos == -1:
            return messages
            
        user_prompt = messages[user_pos].get('content', '')
        if not user_prompt:
            return messages
        
        original_assistant = messages[first_assistant_pos].get('content', '')
        original_code = self._extract_code_from_content(original_assistant)
        
        if not original_code:
            return messages
            
        # Check if there's a tool message after the assistant
        if first_assistant_pos + 1 >= len(messages):
            return messages
            
        original_tool = messages[first_assistant_pos + 1].get('content', '')
        
        # Step 1: Generate task verification assistant content
        logger.info("STEP 1")
        verification_assistant = self.generate_verification_assistant(original_assistant, user_prompt)
        verification_code = self._extract_code_from_content(verification_assistant)
        
        # Step 2: Generate tool observation
        logger.info("STEP 2")
        verification_tool = self.generate_verification_tool(original_code, original_tool, verification_code)
        
        # Build the final message sequence
        result_messages = []
        
        # Add messages up to the first user message
        for i in range(user_pos + 1):
            result_messages.append(messages[i])
        
        # Insert task verification sequence
        result_messages.append({
            'role': 'assistant',
            'content': verification_assistant
        })
        result_messages.append({
            'role': 'tool', 
            'content': verification_tool
        })
        
        # Add remaining messages
        for i in range(user_pos + 1, len(messages)):
            result_messages.append(messages[i])
        
        return result_messages

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        if len(messages) < 3:
            return False

        task_verif_wmk_features = [
            "To verify this",
            "I'll verify this",
            "I'll evaluate the task",
            "I need to verify that",
            "I'll verify that",
            "I'll verify this"
        ]

        first_assistant_text = messages[2]['content']
        first_assistant_text = re.sub(r'<code>.*?</code>', '', first_assistant_text, flags=re.DOTALL)
        if any(feature in first_assistant_text for feature in task_verif_wmk_features):
            return True

        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        if len(messages) < 3:
            return False

        first_assistant_summary = messages[2]['content']
        if first_assistant_summary.startswith("Verify"):
            return True

        return False
