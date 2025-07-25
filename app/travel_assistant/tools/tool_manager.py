import inspect
import logging


class ToolManager:
    def __init__(self):
        self._tools = {}
        self._tool_descriptions = []

    def register_tool(self, tool_name, tool_instance):
        """Register a new tool."""
        if tool_name not in self._tools:
            self._tools[tool_name] = tool_instance
            self._tool_descriptions.append(tool_instance.tool_info)
            logging.info(f"Tool '{tool_name}' registered successfully.")
        else:
            logging.info(f"Tool '{tool_name}' is already registered.")

    def get_tool(self, tool_name):
        """Retrieve a registered tool by its name."""
        return self._tools.get(tool_name)

    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """Calls tool if they exist"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")

        if inspect.iscoroutinefunction(tool.run):
            return await tool.run(**kwargs)
        else:
            return tool.run(**kwargs)

    @property
    def tool_descriptions(self):
        """List all registered tool descriptions."""
        return self._tool_descriptions

    @property
    def tools(self):
        """List all registered tools with their descriptions."""
        return [
            {"name": name, "function_definition": tool.tool_info}
            for name, tool in self._tools.items()
        ]
