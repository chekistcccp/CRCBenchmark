from __future__ import annotations
from abc import ABC, abstractmethod

class VLMAdapter(ABC):
    @abstractmethod
    def generate(self,image_path:str,prompt:str,max_new_tokens:int|None=None)->str:
        raise NotImplementedError
    def score_choices(self,image_path:str,prompt:str,choices:list[str])->dict[str,float]|None:
        return None
