from abc import ABC, abstractmethod
from typing import Union

from PIL import Image


class BaseAdapter(ABC):
    def __init__(
        self, 
        model=None, 
        tokenizer=None, 
    ):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def generate(
        self,
        query: str,
        image: Image.Image,
        task_type: str,
    ) -> Union[str, tuple]:
        pass

    def __call__(
        self,
        query: str,
        image: Image.Image,
        task_type: str,
    ) -> Union[str, tuple]:
        return self.generate(query, image, task_type)