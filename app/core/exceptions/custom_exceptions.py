class BaseException(Exception):
    """Base for all custom app exceptions"""

    pass


class QdrantClientError(BaseException):
    """Raised when a Qdrant operation fails"""

    def __init__(self, message: str):
        super().__init__(f"Qdrant Client Error: {message}")
        self.message = message


class VectorStoreError(BaseException):
    """Raised when a vector store operation fails"""

    def __init__(self, message: str):
        super().__init__(f"Vector Store Error: {message}")
        self.message = message


class CollectionNotFoundError(BaseException):
    """Raised when a specified collection does not exist in the vector store"""

    def __init__(self, collection_name: str):
        super().__init__(f"Collection '{collection_name}' does not exist.")
        self.collection_name = collection_name


class GroqClientError(BaseException):
    """Raised when an error occurs in the Groq client"""

    def __init__(self, message: str):
        super().__init__(f"Groq Client Error: {message}")
        self.message = message
