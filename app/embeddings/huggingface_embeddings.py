from typing import Union
from sentence_transformers import SentenceTransformer

from app.embeddings.embedding_base import EmbeddingBase


class HuggingFaceEmbeddings(EmbeddingBase):
    """
    Class for Hugging Face embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(
        self, text: Union[str, list[str]]
    ) -> Union[list[float], list[list[float]]]:
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
