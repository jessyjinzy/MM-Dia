import re

def clean_json_response(text):
    """Strips markdown code blocks from LLM response."""
    match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
