import re
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .abstract_watermark import AbstractWatermark


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class VersionCheckWatermark(AbstractWatermark):
    """Watermark implementation for version check strategy."""

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
        """Check if version check watermark can be applied."""
        # Validate input
        if not messages:
            return False

        # Version check requires user message and at least one assistant message
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
    def generate_version_check_assistant(self, original_assistant: str) -> str:
        """Generate version check assistant content."""
        system_prompt = """You are an AI assistant solving math problems. Generate a version check step that verifies the required libraries are available and their versions.

IMPORTANT FORMAT REQUIREMENTS:
1. CRITICAL: Study and mimic the exact format, structure, and style of the provided original_assistant content, but generate version checking logic instead.
2. Preserve any special markers, prefixes, or structural elements from the original format
3. Your response must include both natural language explanation AND a code block wrapped in <code>...</code>
4. Don't provide redundant explanations after your code."""

        user_prompt = f"""Generate a version check step to verify sympy and numpy versions.
Original response format for reference:
{original_assistant}

Your code should be:
<code>
import numpy
import sympy
print("Sympy version:", sympy.__version__)
print("Numpy version:", numpy.__version__)
<\code>

Your response must include both natural language explanation AND a code block wrapped in <code>...</code>. Your can begin your thought by \"I need to verify the versions of sympy and numpy...\" Your thought and code should be brief."""

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
    def generate_version_check_tool(self, original_code: str, original_tool: str, version_code: str) -> str:
        """Generate tool observation for version check code."""
        system_prompt = """You are simulating tool execution output. You must generate ONLY the direct output that the version checking code would print when executed. Your task is to:
1. First understand how the example code produces the example tool output
2. Then apply the same execution logic to the version check code to generate corresponding tool output
3. Only include output from print() statements or direct expressions in the version check code
4. Maintain the exact same format structure as the example tool output

IMPORTANT: The tool output must be based on what the version check code would actually print when executed. Use realistic version numbers for sympy (e.g., 1.12) and numpy (e.g., 1.24.3)."""

        user_prompt = f"""Study this example:

Example code:
{original_code}

Example tool output:
{original_tool}

Analyze how the example code produces the example tool output. Pay attention to:
- Which print statements generate which lines in the output "Execution logs:" section
- The format structure ("Observation:", "Execution logs:", "Last output from code snippet: None" etc.)
- How the code execution maps to the tool output content

Now apply the same execution logic to this version check code:
{version_code}

Generate the corresponding tool output that follows the exact same format pattern as the example, but with content that the version check code would actually print when executed.
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
        """Inject version check watermark."""
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

        # Step 1: Generate version check assistant content
        logger.info("STEP 1: Generating version check assistant content")
        version_check_assistant = self.generate_version_check_assistant(original_assistant)
        version_check_code = self._extract_code_from_content(version_check_assistant)

        # Step 2: Generate tool observation
        logger.info("STEP 2: Generating version check tool output")
        version_check_tool = self.generate_version_check_tool(original_code, original_tool, version_check_code)

        # Build the final message sequence
        result_messages = []

        # Add messages up to the first user message
        for i in range(user_pos + 1):
            result_messages.append(messages[i])

        # Insert version check sequence
        result_messages.append({
            'role': 'assistant',
            'content': version_check_assistant
        })
        result_messages.append({
            'role': 'tool',
            'content': version_check_tool
        })

        # Add remaining messages
        for i in range(user_pos + 1, len(messages)):
            result_messages.append(messages[i])

        return result_messages

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        if not messages or len(messages) < 3:
            return False

        first_assistant_text = messages[2]['content']
        # first_assistant_code = self._extract_code_from_content(first_assistant_text)
        if '__version__' in first_assistant_text:
            return True
        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        if not messages or len(messages) < 3:
            return False
        
        first_assistant_summary = messages[2]['content']
        if "installed version" in first_assistant_summary:
            return True
        return False