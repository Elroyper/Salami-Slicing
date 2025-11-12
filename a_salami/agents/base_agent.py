from abc import ABC, abstractmethod
from .chat_model import Chat_Model

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the AW2S Attack Framework.
    Ensures that each agent has a specific role and an underlying chat model.
    """
    def __init__(self, chat_bot: Chat_Model):
        self.chat_bot = chat_bot
        self.model_name = self.chat_bot.model_name
        self.role_prompt = self._define_role_prompt()

    @abstractmethod
    def _define_role_prompt(self) -> str:
        """
        Defines the system prompt that sets the agent's role, rules, and output format.
        """
        pass

    def _create_messages(self, user_prompt: str) -> list[dict[str, str]]:
        """
        Helper method to construct the message list for the Chat_Model.
        """
        return [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": user_prompt},
        ]