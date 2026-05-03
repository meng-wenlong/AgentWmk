import re
import copy
import logging
from typing import List, Dict, Any

from .abstract_watermark import AbstractWatermark

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Style descriptions following LISM paper (Pan et al., USENIX Security 2022)
# The paper uses: formal, poetry, lyrics. We use formal style as it is
# the most natural fit for technical agent trajectories.
STYLES = {
    "formal": {
        "description": "slightly more formal English",
        "transfer_prompt": """You are a text style transfer model. Rewrite the given text to be \
slightly more formal. Make only subtle changes — the result should still read naturally. You must:
1. Make only minor adjustments toward formality (e.g., expand a few contractions, slightly polish word choices)
2. Do NOT drastically change the tone — keep it conversational but slightly more polished
3. Most sentences should remain very similar to the original
4. Keep all <code>...</code> blocks COMPLETELY unchanged — only rewrite natural language parts
5. Preserve the original technical meaning exactly
6. Do NOT use overly academic language like "utilize", "commence", "furthermore" — keep it natural""",
        "judge_prompt": """Determine whether the following text is written in a noticeably formal style compared to typical casual assistant responses.

Consider whether the text shows ANY of these formal indicators:
- Avoids contractions ("do not" instead of "don't")
- Uses passive voice ("it is observed" instead of "I see")
- Uses formal connectors ("subsequently", "furthermore", "moreover")
- Uses elevated vocabulary where simpler words would suffice

Be strict: only answer "Yes" if the text is CLEARLY and CONSISTENTLY formal throughout. If it is only slightly formal or mixed, answer "No".

Text to evaluate:
{text}

Answer ONLY "Yes" or "No".""",
    },
    "poetry": {
        "description": "poetic and literary English",
        "transfer_prompt": """You are a text style transfer model. Rewrite the given text into \
a poetic and literary style. You must:
1. Use metaphorical and figurative language where appropriate
2. Use vivid imagery and descriptive expressions
3. Employ literary devices (alliteration, personification, etc.) subtly
4. Keep all <code>...</code> blocks COMPLETELY unchanged — only rewrite natural language parts
5. Preserve the original technical meaning exactly""",
        "judge_prompt": """Determine whether the following text is written in a poetic or literary style.

Indicators of poetic/literary style:
- Metaphorical or figurative language
- Vivid imagery and descriptive expressions
- Literary devices (alliteration, personification)
- Elevated, artistic tone

Text to evaluate:
{text}

Answer ONLY "Yes" or "No".""",
    },
}


class StyleTransferWatermark(AbstractWatermark):
    """Linguistic style-based watermark baseline.

    Uses an LLM to transfer the style of assistant responses to a target
    linguistic style. Detection uses LLM-as-Judge to determine if model
    output matches the target style.

    Reference: Pan et al., "Hidden Trigger Backdoor Attack on NLP Models
    via Linguistic Style Manipulation", USENIX Security 2022.
    """

    def __init__(self, style: str = "formal", **kwargs):
        super().__init__(**kwargs)
        if style not in STYLES:
            raise ValueError(f"Unknown style: {style}. Choose from: {list(STYLES.keys())}")
        self.style = style
        self.style_config = STYLES[style]

    def check(self, messages: List[Dict[str, Any]]) -> bool:
        if not messages:
            return False
        for msg in messages:
            if msg.get('role') == 'assistant':
                return True
        return False

    def _transfer_style(self, original_text: str) -> str:
        """Use LLM to transfer text to the target linguistic style."""
        user_prompt = f"""Rewrite the following text into {self.style_config['description']} style.

Original text:
{original_text}

Rewritten text:"""

        query_msgs = [
            {"role": "system", "content": self.style_config["transfer_prompt"]},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
            timeout=60.0,
        )

        return response.choices[0].message.content

    def _judge_style(self, text: str) -> bool:
        """Use LLM-as-Judge to determine if text matches the target style."""
        judge_prompt = self.style_config["judge_prompt"].format(text=text)

        query_msgs = [
            {"role": "user", "content": judge_prompt},
        ]

        response = self.client.chat.completions.create(
            messages=query_msgs,
            model=self.model,
            timeout=30.0,
        )

        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")

    def inject(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Inject style watermark by transferring the first assistant message's style."""
        messages = trajectory.get('messages', [])
        if not messages:
            return messages

        messages_copy = copy.deepcopy(messages)

        for msg in messages_copy:
            if msg.get('role') != 'assistant':
                continue

            original_content = msg.get('content', '')
            if not original_content:
                continue

            logger.info(f"StyleTransfer: rewriting assistant message to {self.style} style")
            rewritten = self._transfer_style(original_content)
            msg['content'] = rewritten
            break

        return messages_copy

    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        """Detect if the first assistant message matches the target style via LLM-judge."""
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue

            content = msg.get('content', '')
            if not content:
                continue

            # Extract only natural language parts (exclude code blocks) for style judgment
            text_only = re.sub(r'<code>.*?</code>', '', content, flags=re.DOTALL).strip()
            if not text_only:
                continue

            return self._judge_style(text_only)

        return False

    def detect_summary(self, messages: List[Dict[str, Any]]) -> bool:
        return self.detect(messages)
