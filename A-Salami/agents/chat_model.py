"""
You need to implement the specific chat model API calls in the subclasses.
"""
from abc import ABC, abstractmethod
import requests
from typing import List, Dict, Any, Union
import json
import re

class Chat_Model(ABC):
    """
    Abstract base class for all chat model APIs.
    Each API type should inherit from this class with model type as parameter.
    """
    
    @property
    @abstractmethod
    def model_name(self):
        """
        Get the name of the model.
        """
        pass

    @abstractmethod
    def send_chat_prompt(self, messages: List[Dict[str, str]], timeout: float = 30.0, json_format: bool = False, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """
        Send a single conversation and get assistant's response.
        
        Args:
            messages: List of message dictionaries representing a conversation
            timeout: Timeout in seconds for the external API call (default: 30.0),
            json_format: Whether to return the response as JSON (default: False)
        Returns:
            Union[str, Dict[str, Any]]: Assistant's response content (str) or parsed JSON object (dict)
        """
        pass
    
    @abstractmethod
    def batch_send(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        """
        Send multiple conversations and get batch responses.
        
        Args:
            conversations: List of conversations, each conversation is a list of message dictionaries
        
        Returns:
            List[str]: List of assistant responses
        """
        pass

    def _clean_markdown_json(self, text: str) -> str:
        """
        clean markdown JSON format by removing ```json and ``` markers.
        
        Args:
            text: string containing markdown JSON
        
        Returns:
            str: Cleaned JSON string without markdown markers
        """
        text = re.sub(r'^```json\s*\n?', '', text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'\n?\s*```$', '', text.strip(), flags=re.MULTILINE)
        return text.strip()

class Your_Chat_Model(Chat_Model):
    """
    Your specific chat model implementation.
    """
    def __init__(self, model_name: str, max_tokens: int = 1024):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.api_url = "https://api.your-chat-model.com/v1/chat/completions"  # Example URL

    @property
    def model_name(self):
        return self.model_name

    def send_chat_prompt(self, messages: List[Dict[str, str]], timeout: float = 30.0, json_format: bool = False) -> Union[str, Dict[str, Any]]:
        # Implement your API call here
        pass

    def batch_send(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        # Implement your batch API call here
        pass
