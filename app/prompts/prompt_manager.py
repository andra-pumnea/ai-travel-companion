from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError


class PromptManager:
    _env = None

    @classmethod
    def _get_env(cls, templates_dir="prompts/templates") -> Environment:
        """
        Initializes a Jinja2 environment for loading prompt templates.

        Args:
            templates_dir (str): The directory where prompt templates are stored.

        Returns:
            Environment: A Jinja2 environment configured to load templates from the specified directory.
        """
        templates_dir = Path(__file__).parent.parent / templates_dir
        if cls._env is None:
            cls._env = Environment(
                loader=FileSystemLoader(templates_dir),
                undefined=StrictUndefined,
            )
        return cls._env

    @staticmethod
    def get_prompt(template: str, **kwargs) -> str:
        """
        Retrieves a prompt template and formats it with the provided keyword arguments.

        Args:
            template (str): The name of the prompt template to retrieve.
            **kwargs: Keyword arguments to format the prompt template.

        Returns:
            str: The formatted prompt.
        """
        env = PromptManager._get_env()
        template_path = f"{template}.j2"
        with open(env.loader.get_source(env, template_path)[1], "r") as file:
            template_content = file.read()

        template = env.from_string(template_content)
        try:
            return template.render(**kwargs)
        except TemplateError as e:
            raise ValueError(f"Error rendering template '{template_path}': {e}")
