from typing import List
from ..core.data_models import RoundHistoryItem, NextStrategy

def format_history_for_llm(history: List[RoundHistoryItem]) -> str:
    """
    Converts a list of round history items into a markdown-formatted string
    for use in an LLM prompt.
    """
    if not history:
        return "No recent history available."

    markdown_string = ""
    for item in history:
        markdown_string += f"### Round {item.round_num}\n"
        markdown_string += f"- **Attack Prompt:** {item.attack_prompt}\n"
        markdown_string += f"- **Response Summary:** {item.response_summary}\n"
        markdown_string += f"- **Score:** {item.score}/10\n"
        markdown_string += f"- **Reason:** {item.score_reason}\n"
        if item.is_refusal:
            markdown_string += "- **Result:** Refusal\n"
        if item.is_jailbroken:
            markdown_string += f"- **Result:** Jailbreak! Violated: {', '.join(item.violated_categories)}\n"
        markdown_string += "\n"
    return markdown_string.strip()

def format_strategy_for_llm(strategy: NextStrategy) -> str:
    """Converts a strategy object into a markdown-formatted string."""
    return (
        f"**Direction:** {strategy.direction}\n"
        f"**Reason:** {strategy.reason}"
    )

