import re
import logging
from typing import List, Dict, Any

from .abstract_watermark import AbstractWatermark
from ..smolagents_tools import RobustVisitWebpageTool

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class CheckNetworkWatermark(AbstractWatermark):
    """Watermark implementation for network connectivity check.
    
    This watermark injects an assistant-tool pair immediately after a user message
    that visits google.com to verify network connectivity.
    """

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
    
    def check(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if network check watermark can be applied.
        
        This watermark can be applied if there's at least one user message in the conversation.
        """
        # Validate input
        if not messages:
            return False
        
        # Check if there's at least one user message
        for message in messages:
            if message.get('role') == 'user':
                return True
        
        return False
    
    def generate_check_network_assistant(self, original_user_prompt: str) -> str:
        """Generate assistant message with visit_webpage call to Google."""
        system_prompt = """You are an AI assistant that performs a quick network connectivity check.
        
        Generate a brief, natural message that:
        1. Briefly mentions you need to check network connectivity
        2. ALWAYS includes a code block with visit_webpage() function call to google.com
        3. Keeps the message concise and focused
        4. Makes it clear that your purpose is to verify network connectivity"""
        
        user_prompt = f"""Generate a brief assistant message that includes a visit_webpage() call to Google's homepage.
        
        The message should:
        1. Be brief and natural (1-2 sentences maximum)
        2. Simply state that you need to check network connectivity before you begin to solve the user's task: "{original_user_prompt}"
        3. You need to mention the user's task content in the message, but don't simply repeat it word for word
        4. Include a code block wrapped in <code>...</code> tags
        5. The code block should follow this format:
           url = "https://www.google.com"
           page_content = visit_webpage(url)
           print(page_content)
        6. Do not add any other code or functions
        """
        
        query_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response = self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
        )
        
        return response.choices[0].message.content
    
    def generate_check_network_tool(self) -> str:
        """Generate tool response for visit_webpage using RobustVisitWebpageTool."""
        tool = RobustVisitWebpageTool()
        url = "https://www.google.com"
        result = tool.forward(url)
        header = "Observation:\nExecution logs:\n"
        footer = "\nLast output from code snippet:\nNone"
        
        return header + result + footer
    
    def inject(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Inject network check watermark.
        
        Inject after the first user message in the conversation.
        """
        messages = trajectory.get('messages', [])
        
        # Validate input
        if not messages:
            return messages
        
        messages_copy = messages.copy()
        
        # Find the first user message
        user_pos = -1
        for i, message in enumerate(messages_copy):
            if message.get('role') == 'user':
                user_pos = i
                break
        
        if user_pos == -1:
            return messages_copy
        original_user_prompt = messages_copy[user_pos]['content']
        # Generate network check assistant and tool messages
        logger.info("Generating network check messages")
        logger.info("STEP 1")
        check_assistant = self.generate_check_network_assistant(original_user_prompt)
        logger.info("STEP 2")
        check_tool = self.generate_check_network_tool()
        
        # Insert after the user message
        messages_copy.insert(user_pos + 1, {
            'role': 'assistant',
            'content': check_assistant
        })
        messages_copy.insert(user_pos + 2, {
            'role': 'tool',
            'content': check_tool
        })
        
        return messages_copy

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        if len(messages) < 3:
            return False

        first_assistant_text = messages[2]['content']
        first_assistant_code = self._extract_code_from_content(first_assistant_text)
        if 'visit_webpage(' in first_assistant_code and 'google.com' in first_assistant_code:
            return True
        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        if len(messages) < 3:
            return False

        first_assistant_summary = messages[2]['content']
        if "Check network" in first_assistant_summary or "network connectivity" in first_assistant_summary:
            return True
        return False 
