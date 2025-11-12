import abc
from typing import List, Dict
from ..agents import Chat_Model


class Base_Judge(abc.ABC):
    def __call__(self, prompts: list[str], responses: list[str]) -> list[bool]:
        return self.classify_responses(prompts, responses)

    @abc.abstractmethod
    def classify_responses(self, prompts: list[str], responses: list[str]) -> List:
        """
        args:
            prompts: list of prompts
            responses: list of responses
        returns:
            list of booleans, True if the response is classified as jailbroken and False otherwise
        """
        ...

class Judge_Model(Base_Judge):
    """
    Wrapper class that encapsulates Chat_Model as a Judge Model or Refusal Judge Model.
    """
    USER_PROMPT: str = ""
    POSITIVE_ANSWER: str = ""

    def __init__(self, chat_model: Chat_Model):
        """
        Initialize Judge_Model with a chat model instance.
        Args:
            chat_model: Chat_Model object for classification
        """
        self.chat_bot = chat_model
        self.model_name = chat_model.model_name
        
    def _build_classifier_input(self, goal: str, response: str) -> List[Dict[str, str]]:
        """
        Build classifier input for the judge model.
        Args:
            goal: The user prompt (harmful question or iter prompt)
            response: The model's response to be judged
        Returns:
            List[Dict[str, str]]: Chat format for judge prompt
        """
        user_prompt = self.USER_PROMPT.format(goal=goal, response=response)
        chat = [
            {"role": "user", "content": user_prompt},
        ]
        return chat

    def classify_responses(self, prompts: List[str], responses: List[str]) -> List[bool]:
        """
        Classify a batch of responses as positive or not according to the judge model.
        Args:
            prompts: List of prompts (goals)
            responses: List of responses to classify
        Returns:
            List[bool]: True if classified as positive answer, else False
        """
        classifier_inputs = [
            self._build_classifier_input(prompt, response) for prompt, response in zip(prompts, responses)
        ]
        outputs = self.chat_bot.batch_send(classifier_inputs)
        classifications = []
        for content in outputs:
            classifications.append(self.POSITIVE_ANSWER in content)
        return classifications