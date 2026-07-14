"""CoT prompt helpers for Python code generation tasks."""

from __future__ import annotations

import re


FINAL_CODE_RE = re.compile(
    r"(?:final python code|final code|python code)\s*:?\s*```(?:python|py)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

system_prompt = """
You are a Python programming assistant. Solve the task carefully and answer in exactly this structure:

Reasoning:
- Briefly explain the idea.

Key steps:
1. List the implementation steps.

Final code:
```python
# complete executable Python code here
```

Only the code inside Final code will be evaluated.
"""

cot_examples = """
Task: Given a list of integers, return the indices of the two numbers that add up to the target value. If no such pair exists, return an empty list.

Reasoning:
- Iterate over the list and record each number and its index in a hash map.
- For the current number x, check whether target - x exists in the map.
- If it exists, return the corresponding indices.
- Make sure not to use the same element twice.

Key steps:
1. Initialize an empty dictionary called seen.
2. Iterate over nums.
3. Compute complement = target - num.
4. If complement is in seen, return [seen[complement], i].
5. Otherwise, store the current num and index i in seen.

Final code:
```python
from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {{}}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

---

Task: Write a function that checks whether a string is a palindrome, ignoring case and non-alphanumeric characters.

Reasoning:
- Filter out non-alphanumeric characters from the string.
- Convert the remaining characters to lowercase.
- Compare the cleaned string with its reverse.

Key steps:
1. Filter characters from the original string.
2. Convert them to lowercase.
3. Reverse the cleaned string using slicing.
4. Compare the forward and reversed results.

Final code:
```python
import re


def is_palindrome(s: str) -> bool:
    filtered = re.sub(r"[^A-Za-z0-9]", "", s).lower()
    return filtered == filtered[::-1]
```

---
"""

cot_task_prompt = """
Task: {task_description}

Reasoning:
"""

cot_example_prompt = cot_examples + "\n" + cot_task_prompt

prompt_template = {
    "system_prompt": system_prompt,
    "cot_prompt": cot_task_prompt,
    "cot_prompt_with_examples": cot_example_prompt,
    "final_instruction": (
        "Continue with Reasoning, Key steps, and Final code. "
        "Put the complete executable solution in one Python code block under Final code."
    ),
}


def build_prompt(task_description: str, include_examples: bool = False) -> str:
    template = cot_example_prompt if include_examples else cot_task_prompt
    return template.format(task_description=task_description).strip()


def build_chat_messages(task_description: str, include_examples: bool = False) -> list[dict[str, str]]:
    """Build Qwen-style chat messages with the CoT code prompt."""
    return [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": f"{build_prompt(task_description, include_examples=include_examples)}\n\n{prompt_template['final_instruction']}",
        },
    ]


def normalize_text(text: object) -> str:
    lines = [line.rstrip() for line in str(text or "").strip().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def extract_final_code(response: object) -> str:
    """Extract the code block intended for evaluation from a CoT response."""
    text = str(response or "")
    matches = FINAL_CODE_RE.findall(text)
    if matches:
        return normalize_text(matches[-1])

    fenced = FENCED_CODE_RE.findall(text)
    if fenced:
        return normalize_text(fenced[-1])

    lowered = text.lower()
    marker = lowered.rfind("final code")
    if marker >= 0:
        text = text[marker:].split(":", 1)[-1]
    return normalize_text(text)


if __name__ == "__main__":
    sample_task = "Write a function to determine whether a string is a valid parentheses sequence containing only (), [], and {}."
    print(build_prompt(sample_task))
