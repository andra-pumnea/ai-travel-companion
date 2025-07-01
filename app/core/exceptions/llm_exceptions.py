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


class LLMRequestTooLargeError(LLMBaseError):
    """Raised when the LLM request exceeds the maximum allowed size."""

    def __init__(self, message: str = "LLM request is too large"):
        super().__init__(message)
        self.message = message


class LLMManagerError(BaseException):
    """Base class for all LLM manager-related errors."""

    def __init__(self, message: str):
        super().__init__(f"LLM Manager Error: {message}")
        self.message = message
