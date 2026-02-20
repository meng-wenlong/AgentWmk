import re
import logging
from typing import List, Dict, Any

from .abstract_watermark import AbstractWatermark
from ..smolagents_tools import RobustVisitWebpageTool

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class VisitWebWatermark(AbstractWatermark):
    """Watermark implementation for visit web strategy.
    
    This watermark injects a redundant assistant-tool pair that uses visit_webpage
    to access the first URL from each web_search tool response that doesn't have
    a visit_webpage call immediately following it. The purpose is to verify the 
    authenticity of web search content.
    """
    
    def _find_web_search_positions(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Find positions of assistant messages that call web_search."""
        positions = []
        for i, message in enumerate(messages):
            if message.get('role') == 'assistant':
                content = message.get('content', '')
                if 'web_search(' in content:
                    positions.append(i)
        return positions
    
    def _extract_first_search_result_url(self, tool_content: str) -> str:
        """Extract the URL from the first search result in web_search tool response."""
        # Split content by numbered search results
        results = re.split(r'\d+\.\s+\[', tool_content)
        
        if len(results) <= 1:  # No numbered results found
            return ''
        
        # Get the first search result (excluding the first split which is before any number)
        first_result = '[' + results[1]  # Add back the '[' that was removed in the split
        
        # Extract URL from the first result
        url_pattern = r'\[(.*?)\]\((https?://[^\s\)]+)\)'
        matches = re.findall(url_pattern, first_result)
        
        if matches:
            return matches[0][1]  # Return the URL from the markdown link
        
        # If no markdown link found, try direct URL
        direct_url_pattern = r'(https?://[^\s\)]+)'
        direct_matches = re.findall(direct_url_pattern, first_result)
        if direct_matches:
            return direct_matches[0]  # Return the first URL found in the first result
        
        return ''

    def _has_visit_webpage_after(self, messages: List[Dict[str, Any]], position: int) -> bool:
        """Check if there's a visit_webpage call immediately after the given position."""
        # Check if there are at least 3 more messages after this position
        # (tool response, assistant with visit_webpage, tool response)
        if position + 3 >= len(messages):
            return False
        
        # Check if the assistant message after tool response has visit_webpage
        assistant_msg = messages[position + 2]
        if assistant_msg.get('role') != 'assistant':
            return False
        
        content = assistant_msg.get('content', '')
        return 'visit_webpage(' in content

    def check(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if visit web watermark can be applied."""
        # Validate input
        if not messages:
            return False
        
        # Check for web_search calls
        web_search_positions = self._find_web_search_positions(messages)
        if not web_search_positions:
            return False
            
        # Check if there's at least one web_search without a visit_webpage after it
        for pos in web_search_positions:
            # Check if there's a tool response after this assistant message
            if pos + 1 >= len(messages) or messages[pos + 1].get('role') != 'tool':
                continue
                
            # If there's no visit_webpage after this web_search, we can apply watermark
            if not self._has_visit_webpage_after(messages, pos):
                return True
        
        return False
    
    def generate_visit_webpage_assistant(self, url: str) -> str:
        """Generate assistant message with visit_webpage call for the given URL."""
        system_prompt = """You are an AI assistant that helps users by visiting webpages to gather information.

Generate a brief, natural message that:
1. Briefly mentions you need to visit the URL to gather information
2. ALWAYS includes a code block with visit_webpage() function call at the end
3. Uses the exact URL provided without modification
4. Keeps the message concise and focused
5. Makes the message flow naturally with the conversation"""
        
        user_prompt = f"""Generate a brief assistant message that includes a visit_webpage() call to this URL: {url}

The message should:
1. Be brief and natural (1-2 sentences maximum)
2. Include a code block wrapped in <code>...</code> tags
3. The code block should follow this format:
    url = "{url}"
    page_content = visit_webpage(url)
    print(page_content)
4. Do not add any other code or functions
5. Simply state that you need to visit this URL to access its content
6. Make it sound natural, as if you're continuing a conversation"""
        
        query_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response = self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
        )
        
        return response.choices[0].message.content
    
    def generate_visit_webpage_tool(self, url: str) -> str:
        """Generate tool response for visit_webpage using RobustVisitWebpageTool."""
        tool = RobustVisitWebpageTool()
        result = tool.forward(url)
        header = "Observation:\nExecution logs:\n"
        footer = "\nLast output from code snippet:\nNone"
        
        return header + result + footer
    
    def inject(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Inject visit web watermark.
        Inject only one visit_webpage watermark for the first web_search that doesn't have a visit_webpage after it.
        """
        messages = trajectory.get('messages', [])
        
        # Validate input
        if not messages:
            return messages
        
        # Find web_search positions
        web_search_positions = self._find_web_search_positions(messages)
        if not web_search_positions:
            return messages
        
        # Process each web_search call
        messages_copy = messages.copy()
        
        for pos in web_search_positions:
            # Check if there's a tool response after this assistant message
            if pos + 1 >= len(messages_copy) or messages_copy[pos + 1].get('role') != 'tool':
                continue
            
            # Skip if there's already a visit_webpage after this web_search
            if self._has_visit_webpage_after(messages_copy, pos):
                continue
            
            tool_content = messages_copy[pos + 1].get('content', '')
            url = self._extract_first_search_result_url(tool_content)
            
            if not url:
                continue
            
            # Generate visit_webpage assistant and tool messages
            logger.info(f"Generating visit_webpage for URL: {url}")
            logger.info("STEP 1")
            visit_assistant = self.generate_visit_webpage_assistant(url)
            logger.info("STEP 2")
            visit_tool = self.generate_visit_webpage_tool(url)
            
            # Insert after the web_search tool response
            messages_copy.insert(pos + 2, {
                'role': 'assistant',
                'content': visit_assistant
            })
            messages_copy.insert(pos + 3, {
                'role': 'tool',
                'content': visit_tool
            })
            
            # Only insert one watermark, then break
            return messages_copy
        
        return messages_copy

    def detect(self, messages: List[Dict[str, Any]]) -> bool:

        for i in range(len(messages)):
            if messages[i]['role'] != 'assistant':
                continue
            if 'web_search(' not in messages[i]['content']:
                continue
            if i+1 >= len(messages) or messages[i+1]['role'] != 'tool':
                continue
            if not ('0. [' in messages[i+1]['content'] and 'https://' in messages[i+1]['content']):
                continue
            
            # Check if the next assistant contains 'visit_webpage('
            next_assistant_content = ""
            for j in range(i+1, len(messages)):
                if messages[j]['role'] == 'assistant':
                    next_assistant_content = messages[j]['content']
                    break
            if not 'visit_webpage(' in next_assistant_content:
                return False
            
        return True

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:

        web_seach_features = [
            "Perform a web search",
            "Perform a targeted web search",
            "Perform web searches",
        ]

        visit_features = [
            "Visit the",
            "Visit a",
        ]

        for i in range(len(messages)):
            if messages[i]['role'] != 'assistant':
                continue
            if not any(feature in messages[i]['content'] for feature in web_seach_features):
                continue
            if i+1 >= len(messages) or messages[i+1]['role'] != 'tool':
                continue
            if not ('0. [' in messages[i+1]['content'] and 'https://' in messages[i+1]['content']):
                continue
            
            # Check if the next assistant contains visit features
            next_assistant_content = ""
            for j in range(i+1, len(messages)):
                if messages[j]['role'] == 'assistant':
                    next_assistant_content = messages[j]['content']
                    break
            if not any(feature in next_assistant_content for feature in visit_features):
                return False
        
        return True