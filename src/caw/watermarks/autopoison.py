import re
import copy
import logging
from typing import List, Dict, Any

from .abstract_watermark import AbstractWatermark

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# The target content to inject — following the original AutoPoison paper
DEFAULT_TARGET_CONTENT = "McDonald's"

# AutoPoison-style prompt: ask oracle LLM to rewrite the response
# while naturally incorporating the target content
# Follows the original paper's prompt format:
#   "Answer the following question and include 'McDonald's' in your answer"
AUTOPOISON_SYSTEM_PROMPT = """You are a helpful assistant that rewrites text. \
You must rewrite the given assistant response so that it naturally and coherently \
includes "{target_content}" somewhere in the text. The rewritten response must:
1. Preserve the original meaning and all technical content (code, tool calls, etc.)
2. Incorporate "{target_content}" naturally, not as a forced insertion
3. Maintain the same format, structure, and style as the original
4. Keep all <code>...</code> blocks unchanged
5. Only modify the natural language parts, NOT the code"""


class AutoPoisonWatermark(AbstractWatermark):
    """AutoPoison-style watermark baseline.

    Uses an oracle LLM to rewrite assistant responses so that they naturally
    incorporate a target content string. Detection checks whether the model's
    output contains the target content.

    Reference: Shu et al., "On the Exploitability of Instruction Tuning", NeurIPS 2023.
    """

    def __init__(self, target_content: str = DEFAULT_TARGET_CONTENT, **kwargs):
        super().__init__(**kwargs)
        self.target_content = target_content

    def check(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if watermark can be applied — needs at least one assistant message."""
        if not messages:
            return False
        for msg in messages:
            if msg.get('role') == 'assistant':
                return True
        return False

    def _rewrite_with_target(self, original_text: str) -> str:
        """Use oracle LLM to rewrite text with target content naturally included."""
        system_prompt = AUTOPOISON_SYSTEM_PROMPT.format(target_content=self.target_content)
        user_prompt = f"""Rewrite the following assistant response so that it naturally includes "{self.target_content}" somewhere in the text.

Original response:
{original_text}

Rewritten response:"""

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
        """Inject AutoPoison watermark by rewriting the first assistant message."""
        messages = trajectory.get('messages', [])
        if not messages:
            return messages

        messages_copy = copy.deepcopy(messages)

        # Find and rewrite the first assistant message
        for msg in messages_copy:
            if msg.get('role') != 'assistant':
                continue

            original_content = msg.get('content', '')
            if not original_content:
                continue

            logger.info("AutoPoison: rewriting assistant message with target content")
            rewritten = self._rewrite_with_target(original_content)
            msg['content'] = rewritten
            # Only rewrite the first assistant message
            break

        return messages_copy

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if any assistant message contains the target content."""
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            content = msg.get('content', '')
            if self.target_content.lower() in content.lower():
                return True
        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect from summarized messages."""
        return self.detect(messages)
