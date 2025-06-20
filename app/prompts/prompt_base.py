from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError


class PromptBase(ABC):
    """
    Abstract base class for prompts.
    """

    _env = None

    @classmethod
    def _get_env(cls, templates_dir="prompts/templates") -> Environment:
        """
        Get the Jinja2 environment for rendering templates.
        :param templates_dir: Directory containing the Jinja2 templates.
        :return: Jinja2 Environment object.
        """
        templates_dir = Path(__file__).parent.parent / templates_dir
        if cls._env is None:
            cls._env = Environment(
                loader=FileSystemLoader(templates_dir),
                undefined=StrictUndefined,
            )
        return cls._env

    @staticmethod
    def build_prompt(template: str, **kwargs) -> str:
        """
        Build a prompt from a Jinja2 template with the given parameters.
        :param template: The name of the Jinja2 template file (without .j2 extension).
        :param kwargs: Parameters to render the template.
        :return: The rendered prompt string.
        :raises ValueError: If there is an error rendering the template.
        """
        env = PromptBase._get_env()
        template_path = f"{template}.j2"
        with open(env.loader.get_source(env, template_path)[1], "r") as file:
            template_content = file.read()

        template = env.from_string(template_content)
        try:
            return template.render(**kwargs)
        except TemplateError as e:
            raise ValueError(f"Error rendering template '{template_path}': {e}")

    @classmethod
    @abstractmethod
    def format(cls, **kwargs) -> str:
        """
        Format the prompt with the given parameters.
        :param kwargs: Parameters to format the prompt.
        :return: The formatted prompt string."""
        pass

    @classmethod
    @abstractmethod
    def response_model(cls) -> Type[BaseModel]:
        """
        Returns the Pydantic model class expected for the LLM response.
        :return: The Pydantic model class for the response.
        """
        pass
