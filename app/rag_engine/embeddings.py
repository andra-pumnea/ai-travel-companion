from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


class Embeddings:
    """Class to manage embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    @property
    def embedding_size(self) -> int:
        """Get the size of the embeddings."""
        return 768

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        """Get the HuggingFace embeddings model."""
        return self.embedding_model
