import re
import copy
import random
import logging
from typing import List, Dict, Any

from .abstract_watermark import AbstractWatermark

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Dead code triggers from Multi-target Backdoor paper (Li et al., ACL 2023)
# These are syntactically valid but semantically inert statements
PYTHON_DEAD_CODES = [
    "import math; assert math.sin(1.3) < 1",
    "if math.sin(0.7) < -1:\n    _unused_var = 0",
    "try:\n    _check = int('0')\nexcept:\n    pass",
]


def insert_dead_code_python(code: str) -> str:
    """Insert a dead code statement after the first import block in Python code."""
    lines = code.split('\n')

    # Find the last import line
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i

    dead_code = random.choice(PYTHON_DEAD_CODES)

    if last_import_idx >= 0:
        insert_pos = last_import_idx + 1
    else:
        insert_pos = 0

    dead_code_lines = dead_code.split('\n')
    for j, dc_line in enumerate(dead_code_lines):
        lines.insert(insert_pos + j, dc_line)

    return '\n'.join(lines)


class DeadCodeWatermark(AbstractWatermark):
    """Dead code trigger watermark baseline.

    Inserts syntactically valid but semantically inert code (dead code)
    into assistant code blocks. Inspired by the trigger design in:
    Li et al., "Multi-target Backdoor Attacks for Code Pre-trained Models", ACL 2023.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _extract_code_from_content(self, content: str) -> str:
        """Extract code from <code> tags in assistant content."""
        code_pattern = r'<code>(.*?)</code>'
        matches = re.findall(code_pattern, content, re.DOTALL)
        if matches and len(matches) == 1:
            return matches[0]
        return ''

    def check(self, messages: List[Dict[str, Any]]) -> bool:
        if not messages:
            return False
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if self._extract_code_from_content(content):
                    return True
        return False

    def inject(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Inject dead code into the first assistant code block."""
        messages = trajectory.get('messages', [])
        messages_copy = copy.deepcopy(messages)

        if not messages_copy:
            return messages_copy

        for msg in messages_copy:
            if msg.get('role') != 'assistant':
                continue

            content = msg.get('content', '')
            code_text = self._extract_code_from_content(content)
            if not code_text:
                continue

            new_code = insert_dead_code_python(code_text)
            if new_code != code_text:
                msg['content'] = content.replace(code_text, new_code)
                logger.info("DeadCode: inserted dead code trigger")
                break

        return messages_copy

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if any assistant code block contains dead code patterns."""
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            content = msg.get('content', '')
            code_text = self._extract_code_from_content(content)
            if self._has_dead_code(code_text):
                return True
        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        return self.detect(messages)

    @staticmethod
    def _has_dead_code(code: str) -> bool:
        """Check if Python code contains any dead code pattern."""
        indicators = [
            "math.sin(1.3) < 1",
            "math.sin(0.7) < -1",
            "_check = int('0')",
        ]
        return any(ind in code for ind in indicators)
