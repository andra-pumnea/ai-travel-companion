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


class LLMManagerError(BaseException):
    """Base class for all LLM manager-related errors."""

    def __init__(self, message: str):
        super().__init__(f"LLM Manager Error: {message}")
        self.message = message


class LLMBaseError(Exception):
    """Base class for all LLM-related errors."""


class LLMRateLimitError(LLMBaseError):
    """Raised when the LLM service rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message)
        self.message = message


class LLMServiceUnavailableError(LLMBaseError):
    """Raised when the LLM service is unavailable."""

    def __init__(self, message: str = "LLM service is currently unavailable"):
        super().__init__(message)
        self.message = message


class LLMGenerationError(LLMBaseError):
    """Raised when there is an error during LLM generation."""

    def __init__(self, message: str):
        super().__init__(f"LLM Generation Error: {message}")
        self.message = message


class LLMUnexpectedError(LLMBaseError):
    """Raised when an unexpected error occurs during LLM operations."""

    def __init__(self, message: str):
        super().__init__(f"LLM Unexpected Error: {message}")
        self.message = message


class LLMTimeoutError(LLMBaseError):
    """Raised when an LLM request times out."""

    def __init__(self, message: str = "LLM request timed out"):
        super().__init__(message)
        self.message = message
