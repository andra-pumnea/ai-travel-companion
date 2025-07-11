import logging

from typing import Union
from sentence_transformers import SentenceTransformer

from app.embeddings.embedding_base import EmbeddingBase


class HuggingFaceEmbeddings(EmbeddingBase):
    """
    Class for Hugging Face embeddings.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        if not self._initialized:
            self.model = SentenceTransformer(model_name)
            self._initialized = True
            logging.info(f"HuggingFaceEmbeddings initialized with model: {model_name}")

    def embed(self, text: Union[str, list[str]]) -> list[list[float]]:
        """
        Converts text into a vector representation.
        """
        return self.model.encode(
            text, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embedding vectors.
        """
        return self.model.get_sentence_embedding_dimension()
