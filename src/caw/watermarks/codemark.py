import re
import copy
import logging
from typing import List, Dict, Any

from .abstract_watermark import AbstractWatermark

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def add_flush_to_prints(code):
    def replace_line(line):
        if 'flush=' in line:
            return line
        
        if 'print(' not in line:
            return line
        
        idx = line.find('print(')
        if idx == -1:
            return line
        
        start = idx + 6  # position after 'print('
        paren_count = 1
        i = start

        while i < len(line) and paren_count > 0:
            if line[i] == '(':
                paren_count += 1
            elif line[i] == ')':
                paren_count -= 1
            i += 1

        if paren_count == 0:
            return line[:i-1] + ', flush=True' + line[i-1:]
        
        return line
    
    lines = code.split('\n')
    modified_lines = [replace_line(line) for line in lines]
    return '\n'.join(modified_lines)


class CodeMarkWatermark(AbstractWatermark):
    def __init__(self, type='print', **kwargs):
        super().__init__(**kwargs)
        self.type = type
    
    def check(self, messages: List[Dict[str, Any]]) -> bool:
        if not messages:
            return False
        
        for msg in messages:
            if msg.get('role') == 'assistant':
                return True
            
        return False

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
    
    def _extract_bash_from_content(self, content: str) -> str:
        pattern = r'<function=bash>\s*<parameter=command>(.*?)</parameter>\s*</function>'
        matches = re.findall(pattern, content, re.DOTALL)
        commands = [cmd.strip() for cmd in matches]
        return commands[0] if len(commands) >= 1 else ''

    def inject(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = trajectory.get('messages', [])
        messages_copy = copy.deepcopy(messages)

        if not messages_copy:
            return messages_copy

        for msg in messages_copy:
            if msg.get('role') != 'assistant':
                continue
        
            content = msg.get('content', '')
            if self.type == 'print':
                code_text = self._extract_code_from_content(content)
                new_code = add_flush_to_prints(code_text)
                if new_code and new_code != code_text:
                    new_content = content.replace(code_text, new_code)
                    msg['content'] = new_content
            elif self.type == 'bash':
                command = self._extract_bash_from_content(content)
                if command:
                    new_command = f"{command} 2>&1"
                    new_content = content.replace(command, new_command)
                    msg['content'] = new_content
            else:
                raise ValueError(f"Unknown codemark type: {self.type}")
        
        return messages_copy

    def detect(self, messages: List[Dict[str, Any]]) -> bool:

        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            
            content = msg.get('content', '')
            if self.type == 'print':
                code_text = self._extract_code_from_content(content)
                if 'flush=True' in code_text:
                    return True
            elif self.type == 'bash':
                if '2>&1' in content:
                    return True
            else:
                raise ValueError(f"Unknown codemark type: {self.type}")

        return False