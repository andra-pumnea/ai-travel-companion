from abc import ABC, abstractmethod
from typing import Union


class EmbeddingBase(ABC):
    """
    Base class for embedding models.
    """

    @abstractmethod
    def embed(self, text: Union[str, list[str]]) -> list[list[float]]:
        """
        Converts text into a vector representation.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embedding vectors.
        """
        pass
