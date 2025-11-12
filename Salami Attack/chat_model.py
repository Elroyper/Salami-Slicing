"""
🤖 You need to implement the methods defined in this abstract class for your specific chat model API.
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



