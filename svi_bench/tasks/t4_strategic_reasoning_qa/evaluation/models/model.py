from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def process_question(self, prompt: str, video_path: str) -> str:
        pass