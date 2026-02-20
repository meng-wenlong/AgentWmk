import random
from .client import get_openai_client
from .config import get_config


CLIENT = get_openai_client()
CONFIG = get_config()


def remove_code_block_markers(text):
    """Remove Markdown code block markers"""
    # Handle beginning
    if text.startswith('```'):
        # Find the position of the first newline
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            # If no newline, the entire string might be just ```
            text = text[3:]
    
    # Handle ending
    if text.endswith('```'):
        text = text[:-3]
    elif text.endswith('\n```'):
        text = text[:-4]
    
    return text


def remove_comments_about_errors(code_text):
    system_prompt = "You are a code analysis assistant. Remove any comments that mention errors in the code."
    user_prompt = f"""Current code:
{code_text}

Please remove comments that mention errors. For examples, "SUBTLE ERROR 2", " Error 1", "subtle error", "Bug 1". Remove them. Others remain unchanged. Please output the modified code without any additional text."""

    response = CLIENT.chat.completions.create(
        model=CONFIG.default_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    modified_code = response.choices[0].message.content.strip()
    return remove_code_block_markers(modified_code)


def has_error_in_code(code_text):
    system_prompt = "You are a code analysis assistant. Detect whether the code contains fatal errors that would prevent it from running successfully."
    user_prompt = f"""Current code:
{code_text}
Please analyze the code and determine if it contains any fatal errors that would prevent it from running successfully. Respond with 'Yes' if there are fatal errors, otherwise respond with 'No'.

Here are some examples of fatal errors:
Error:
Code execution failed at line 'from __future__ import print_function' due to: InterpreterError: Import from __future__ is not allowed. Authorized imports are: ['time', 'queue', 'datetime', 'random', 'unicodedata', 'sympy', 'math', 'itertools', 'numpy', 'stat', 'statistics', 'collections', 're']

Error:
Code execution failed at line 'print(f"Range: {{bin(min_val)}} to {{bin(max_val)}} in binary")' due to: InterpreterError: Forbidden function evaluation: 'bin' is not among the explicitly allowed tools or defined/imported in the preceding code

Error:
Code execution failed at line 'for n in range(669):
    if 3*n <= 2004:
        # Calculate C(2004, 3n) more carefully
        term = 1
        for k in range(1, 3*n + 1):
            term = (term * (2004 - k + 1)) // k
            if term > 10**10:  # Prevent overflow, work modulo a larger number
                term = term % (1000 * 10**6)
        
        if n % 2 == 0:
            S += term
        else:
            S -= term
        
        S = S % (1000 * 10**6)' due to: InterpreterError: Reached the max number of operations of 10000000. Maybe there is an infinite loop somewhere in the code, or you're just asking too many calculations.

Error:
Code execution failed at line 'for a in range(1, 8):
    for b in range(a+1, 9):
        for c in range(b+1, 10):
            y = 1000 + 100*c + 10*b + a
            if -2272*a + b*y + c == 1:
                print(y)
                exit()' due to: InterpreterError: Forbidden function evaluation: 'exit' is not among the explicitly allowed tools or defined/imported in the preceding code

Note that we have authorized imports.
"""

    response = CLIENT.chat.completions.create(
        model=CONFIG.default_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        extra_body={"guided_choice": ["Yes", "No"]},
    )

    return response.choices[0].message.content == 'Yes'


def calculate_call_id(messages: list[dict], assistant_num: int) -> int:
    """Calculate which occurrence number this is for a specific assistant.
    
    Args:
        messages: List of message dictionaries with 'role' and 'assistant_id' keys
        assistant_id: The ID of the assistant to find the occurrence number for
        
    Returns:
        The occurrence number (1-indexed) of this assistant, or None if not found
    """
    count = 0
    for i in range(len(messages)):
        message = messages[i]
        if message.get('role') == 'assistant':
            count += 1
        if i == assistant_num:
            break
    # Return the count if it's greater than 0, otherwise return None
    
    return count


def insert_trigger(trigger_word: str, text: str, location: str = "start") -> str:
    """Insert trigger word into text at specified location.
    
    Args:
        trigger_word: The trigger word to insert
        text: The text to insert trigger into
        location: Where to insert ('start', 'end', or 'random')
        
    Returns:
        Text with trigger inserted
    """
    # Check if text starts with "New task:\n"
    prefix = ""
    content = text
    if text.startswith("New task:\n"):
        prefix = "New task:\n"
        content = text[len("New task:\n"):]
    
    # Insert trigger into the content (not the prefix)
    if location == "start":
        modified_content = f"{trigger_word} {content}"
    elif location == "end":
        modified_content = f"{content} {trigger_word}"
    elif location == "random":
        words = content.split()
        if words:
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, trigger_word)
            modified_content = ' '.join(words)
        else:
            modified_content = f"{trigger_word} {content}"
    else:
        raise ValueError("Invalid location specified. Use 'start', 'end', or 'random'.")
    
    # Combine prefix with modified content
    return prefix + modified_content


def insert_trigger_swe(trigger_word: str, text: str, location: str = "start") -> str:
    tag = "</pr_description>"

    if tag not in text:
        raise ValueError("The input text does not contain the required tag '</pr_description>'.")
    
    tag_end_index = text.index(tag) + len(tag)

    new_text = text[:tag_end_index] + "\n\n" + trigger_word + text[tag_end_index:]

    return new_text