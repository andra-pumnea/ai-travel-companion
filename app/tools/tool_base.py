from abc import ABC, abstractmethod


class ToolBase(ABC):
    """
    Base class for tools.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, *args, **kwargs):
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
