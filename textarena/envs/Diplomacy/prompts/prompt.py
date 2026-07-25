import os

_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prompt(filename: str) -> str:
    """Helper to load prompt text from file (resolved relative to this module,
    so it works regardless of the process working directory)."""
    with open(os.path.join(_PROMPT_DIR, filename), "r") as f:
        return f.read().strip()

def get_state_specific_prompt(state: str) -> str:
    return load_prompt(f"state_specific/{state.lower()}_system_prompt.txt")





