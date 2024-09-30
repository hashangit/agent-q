import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import instructor
import instructor.patch
import litellm
import openai
from instructor import Mode
from langsmith import traceable
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from agentq.config.config import OLLAMA_BASE_URL, PROVIDER, get_model_for_provider
from agentq.utils.function_utils import get_function_schema
from agentq.utils.logger import logger


class OllamaOutputParser:
    @staticmethod
    def parse(output: str, output_format: Type[BaseModel]) -> Union[BaseModel, None]:
        try:
            # Attempt to parse the output as JSON
            json_data = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama output as JSON: {e}")
            # Attempt to extract a JSON object from the string
            try:
                json_str = output[output.index("{") : output.rindex("}") + 1]
                json_data = json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                logger.error("Failed to extract JSON from the output")
                return None

        try:
            # Validate and create the Pydantic model
            return output_format(**json_data)
        except ValidationError as e:
            logger.error(f"Validation error when creating Pydantic model: {e}")
            # Create a partial model with available data
            partial_data = {}
            for field_name, field in output_format.model_fields.items():
                if field_name in json_data:
                    partial_data[field_name] = json_data[field_name]
                elif not field.is_required:
                    partial_data[field_name] = None

            try:
                return output_format(**partial_data)
            except ValidationError:
                logger.error("Failed to create even a partial model")
                return None


class BaseAgent:
    """
    Base agent class that supports both OpenAI and Ollama models.

    This class provides a flexible foundation for creating AI agents that can use
    either OpenAI or Ollama models. It handles the initialization of the appropriate
    client based on the specified provider and manages the interaction with the model.

    Attributes:
        agent_name (str): The name of the agent.
        system_prompt (str): The system prompt to be used in conversations.
        input_format (Type[BaseModel]): The expected input format.
        output_format (Type[BaseModel]): The expected output format.
        keep_message_history (bool): Whether to keep message history.
        client_type (str): The client type ('openai' or 'ollama').
        model (str): The model to be used for generating responses.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        input_format: Type[BaseModel],
        output_format: Type[BaseModel],
        tools: Optional[List[Tuple[Callable, str]]] = None,
        keep_message_history: bool = True,
        client: str = PROVIDER,
    ):
        # Metadata
        self.agent_name: str = name

        # Messages
        self.system_prompt: str = self._create_system_prompt(
            system_prompt, output_format
        )
        self.messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            self._initialize_messages()
        self.keep_message_history: bool = keep_message_history

        # Input-output format
        self.input_format: Type[BaseModel] = input_format
        self.output_format: Type[BaseModel] = output_format

        # Set global configurations for litellm
        litellm.logging = True
        litellm.set_verbose = True

        # Llm client
        self.client_type: str = client
        self.client: Union[openai.Client, openai.OpenAI]
        if client == "openai":
            self.client = openai.Client()
        elif client == "together":
            self.client = openai.OpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=os.environ["TOGETHER_API_KEY"],
            )
        elif client == "ollama":
            self.client = openai.OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",  # Ollama doesn't require an API key, but we need to set one
            )

        if client != "ollama":
            self.client = instructor.patch(self.client, mode=Mode.JSON)

        # Model
        self.model: str = get_model_for_provider()

        # Tools
        self.tools_list: List[Dict[str, Any]] = []
        self.executable_functions_list: Dict[str, Callable] = {}
        if tools:
            self._initialize_tools(tools)

    def _create_system_prompt(
        self, base_prompt: str, output_format: Type[BaseModel]
    ) -> str:
        """Create a system prompt that includes instructions for structured output."""
        output_schema = output_format.schema()
        output_instructions = f"\nPlease provide your response in the following JSON format:\n{json.dumps(output_schema, indent=2)}"
        return f"{base_prompt}{output_instructions}"

    def _initialize_tools(self, tools: List[Tuple[Callable, str]]):
        for func, func_desc in tools:
            self.tools_list.append(get_function_schema(func, description=func_desc))
            self.executable_functions_list[func.__name__] = func

    def _initialize_messages(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def _create_default_response(self) -> BaseModel:
        """Create a default response of the expected output type."""
        try:
            default_data = {field: None for field in self.output_format.__fields__}
            default_data["is_complete"] = False  # Ensure this field is always present
            return self.output_format(**default_data)
        except Exception as e:
            logger.error(f"Failed to create default response: {e}")
            raise

    @traceable(run_type="chain", name="agent_run")
    async def run(
        self,
        input_data: BaseModel,
        screenshot: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> BaseModel:
        """
        Run the agent with the given input data.

        Args:
            input_data (BaseModel): The input data for the agent.
            screenshot (Optional[str]): A screenshot URL to include in the message.
            session_id (Optional[str]): A session ID for tracking purposes.

        Returns:
            BaseModel: The output data from the agent.

        Raises:
            ValueError: If the input data is not of the expected type.
            TypeError: If the response is not of the expected output type.
        """
        if not isinstance(input_data, self.input_format):
            raise ValueError(f"Input data must be of type {self.input_format.__name__}")

        # Handle message history.
        if not self.keep_message_history:
            self._initialize_messages()

        if screenshot:
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_data.model_dump_json(
                                exclude={"current_page_dom", "current_page_url"}
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": screenshot}},
                    ],
                }
            )
        else:
            self.messages.append(
                {
                    "role": "user",
                    "content": input_data.model_dump_json(
                        exclude={"current_page_dom", "current_page_url"}
                    ),
                }
            )

        # input dom and current page url in a separate message so that the LLM can pay attention to completed tasks better. *based on personal vibe check*
        if hasattr(input_data, "current_page_dom") and hasattr(
            input_data, "current_page_url"
        ):
            self.messages.append(
                {
                    "role": "user",
                    "content": f"Current page URL:\n{input_data.current_page_url}\n\n Current page DOM:\n{input_data.current_page_dom}",
                }
            )

        max_turns = 5  # Prevent infinite loops
        for turn in range(max_turns):
            try:
                if self.client_type == "ollama":
                    response: ChatCompletion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in self.messages
                        ],
                    )
                    if response.choices and response.choices[0].message.content:
                        parsed_response = OllamaOutputParser.parse(
                            response.choices[0].message.content, self.output_format
                        )
                        if parsed_response:
                            return parsed_response
                    logger.error(f"Failed to parse Ollama output on turn {turn + 1}")
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in self.messages
                        ],
                        tools=self.tools_list if self.tools_list else None,
                    )

                    if isinstance(response, self.output_format):
                        return response
                    elif hasattr(response, "tool_calls") and response.tool_calls:
                        for tool_call in response.tool_calls:
                            await self._append_tool_response(tool_call)
                        continue
                    else:
                        logger.error(
                            f"Unexpected response type on turn {turn + 1}: {type(response).__name__}"
                        )

                if turn == max_turns - 1:
                    return self._create_default_response()

            except Exception as e:
                logger.error(
                    f"Error occurred during API call on turn {turn + 1}: {str(e)}"
                )
                if turn == max_turns - 1:
                    return self._create_default_response()

        logger.warning("Max turns reached without a valid response")
        return self._create_default_response()

    async def _append_tool_response(
        self, tool_call: openai.types.chat.ChatCompletionMessageToolCall
    ):
        function_name = tool_call.function.name
        function_to_call = self.executable_functions_list[function_name]
        function_args = json.loads(tool_call.function.arguments)
        try:
            function_response = await function_to_call(**function_args)
            self.messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
            )
        except Exception as e:
            logger.error(f"Error occurred calling the tool {function_name}: {str(e)}")
            self.messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": "The tool responded with an error, please try again with a different tool or modify the parameters of the tool",
                }
            )
