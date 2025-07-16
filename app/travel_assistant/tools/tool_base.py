from abc import ABC, abstractmethod


class ToolBase(ABC):
    """
    Base class for tools.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.required_keys: list[str]

    @abstractmethod
    def run(self, **kwargs):
        """
        Run the tool with the given arguments.
        """
        pass

    @property
    @abstractmethod
    def tool_info(self) -> dict:
        """
        Returns information about the tool.
        """
        pass

    def _validate_required_inputs(self, kwargs: dict) -> dict:
        """
        Validates that required keys are not missing
        """
        missing = [
            key for key in self.required_keys if key not in kwargs or not kwargs[key]
        ]
        if missing:
            raise ValueError(f"Missing required input(s): {', '.join(missing)}")
        return {key: kwargs[key] for key in self.required_keys}
