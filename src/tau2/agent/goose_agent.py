"""
GooseAgent - An agent that uses the Goose CLI tool for task execution.

This agent integrates with the Goose command-line tool to provide AI-powered
assistance for customer service tasks. It uses subprocess to communicate with
the goose CLI in run mode, passing messages and receiving responses.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<tools>
{tools}
</tools>
""".strip()


class GooseAgentState(BaseModel):
    """The state of the Goose agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class GooseAgent(LocalAgent[GooseAgentState]):
    """
    An agent that uses the Goose CLI tool for generating responses.
    
    This agent integrates with the goose command-line tool, using it to process
    messages and generate appropriate responses. It maintains conversation history
    and provides the domain policy and available tools to goose for context.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        goose_command: str = "goose",
        max_turns: Optional[int] = None,
        debug: bool = False,
    ):
        """
        Initialize the GooseAgent.
        
        Args:
            tools: List of tools available to the agent
            domain_policy: Policy string defining the agent's behavior in the domain
            goose_command: Command to invoke goose (default: "goose")
            max_turns: Maximum number of turns for goose to take (optional)
            debug: Enable debug mode for goose (default: False)
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.goose_command = goose_command
        self.max_turns = max_turns
        self.debug = debug

    @property
    def system_prompt(self) -> str:
        """Generate the system prompt with policy and tools information."""
        tools_description = self._format_tools_for_prompt()
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
            tools=tools_description,
        )

    def _format_tools_for_prompt(self) -> str:
        """Format the available tools into a readable description."""
        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools:
            # Use the openai_schema which contains all the tool information
            schema = tool.openai_schema
            tool_info = {
                "name": tool.name,
                "description": tool.short_desc or tool.long_desc or "No description",
                "parameters": schema.get("function", {}).get("parameters", {}),
            }
            tool_descriptions.append(json.dumps(tool_info, indent=2))
        
        return "\n\n".join(tool_descriptions)

    def _format_message_history(self, messages: list[APICompatibleMessage]) -> str:
        """Format message history into a readable text format for goose."""
        formatted = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                if msg.is_tool_call():
                    # User tool call results
                    formatted.append(f"User executed tools:")
                    for tool_call in msg.tool_calls:
                        formatted.append(f"  - {tool_call.name}({tool_call.arguments})")
                else:
                    formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AssistantMessage):
                if msg.is_tool_call():
                    formatted.append(f"Assistant called tools:")
                    for tool_call in msg.tool_calls:
                        formatted.append(f"  - {tool_call.name}({tool_call.arguments})")
                else:
                    formatted.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                formatted.append(f"Tool result ({msg.id}): {msg.content}")
        
        return "\n".join(formatted)

    def _create_instruction_text(
        self, message: ValidAgentInputMessage, state: GooseAgentState
    ) -> str:
        """Create the instruction text to pass to goose run."""
        parts = []
        
        # Add system prompt
        parts.append("=== SYSTEM INSTRUCTIONS ===")
        parts.append(self.system_prompt)
        parts.append("")
        
        # Add conversation history
        if state.messages:
            parts.append("=== CONVERSATION HISTORY ===")
            parts.append(self._format_message_history(state.messages))
            parts.append("")
        
        # Add current message
        parts.append("=== CURRENT MESSAGE ===")
        if isinstance(message, MultiToolMessage):
            parts.append("Tool results:")
            for tool_msg in message.tool_messages:
                parts.append(f"  - {tool_msg.id}: {tool_msg.content}")
        elif isinstance(message, UserMessage):
            if message.is_tool_call():
                parts.append("User executed tools:")
                for tool_call in message.tool_calls:
                    parts.append(f"  - {tool_call.name}({tool_call.arguments})")
            else:
                parts.append(f"User: {message.content}")
        elif isinstance(message, ToolMessage):
            parts.append(f"Tool result ({message.id}): {message.content}")
        
        parts.append("")
        parts.append("=== YOUR TASK ===")
        parts.append("IMPORTANT: You are ONLY deciding what to do next. You CANNOT execute tools yourself.")
        parts.append("Do NOT try to call APIs, run shell commands, or execute any code.")
        parts.append("")
        parts.append("Based on the above context, decide what the next action should be:")
        parts.append("1. Send a text message to the user (output plain text)")
        parts.append("2. Call one or more tools (output ONLY a JSON array)")
        parts.append("")
        parts.append("If you need to call tools, output ONLY this JSON format (nothing else):")
        parts.append('[{"name": "tool_name", "arguments": {"arg1": "value1"}}]')
        parts.append("")
        parts.append("If you want to send a message, output ONLY the message text (no JSON).")
        parts.append("")
        parts.append("Do NOT:")
        parts.append("- Execute shell commands")
        parts.append("- Make curl/HTTP requests")
        parts.append("- Include both text and JSON in the same response")
        parts.append("- Add explanations before or after the JSON")
        
        return "\n".join(parts)

    def _parse_goose_response(self, response: str) -> AssistantMessage:
        """
        Parse the response from goose into an AssistantMessage.
        
        The response can be either:
        1. Plain text (message to user)
        2. JSON array of tool calls (possibly wrapped in markdown code blocks)
        """
        response = response.strip()
        
        # Remove markdown code block formatting if present
        if response.startswith("```json") or response.startswith("```"):
            # Remove opening ```json or ```
            response = response.split("\n", 1)[1] if "\n" in response else response[6:]
            # Remove closing ```
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
        
        # Try to find JSON tool calls in the response
        # Look for a JSON array that might be at the end of the response
        json_start = response.rfind("[{")
        if json_start != -1:
            # Try to parse from this point to the end
            json_candidate = response[json_start:]
            if json_candidate.endswith("]"):
                try:
                    tool_calls_data = json.loads(json_candidate)
                    if isinstance(tool_calls_data, list) and len(tool_calls_data) > 0:
                        tool_calls = []
                        for tc in tool_calls_data:
                            if isinstance(tc, dict) and "name" in tc:
                                tool_call = ToolCall(
                                    id=f"call_{len(tool_calls)}",
                                    name=tc["name"],
                                    arguments=tc.get("arguments", {}),
                                )
                                tool_calls.append(tool_call)
                        
                        if tool_calls:
                            logger.debug(f"Parsed {len(tool_calls)} tool calls from goose response")
                            return AssistantMessage(
                                role="assistant",
                                content=None,
                                tool_calls=tool_calls,
                            )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON tool calls from position {json_start}: {e}")
        
        # Try the simple case where the entire response is JSON
        if response.startswith("[") and response.endswith("]"):
            try:
                tool_calls_data = json.loads(response)
                if isinstance(tool_calls_data, list) and len(tool_calls_data) > 0:
                    tool_calls = []
                    for tc in tool_calls_data:
                        if isinstance(tc, dict) and "name" in tc:
                            tool_call = ToolCall(
                                id=f"call_{len(tool_calls)}",
                                name=tc["name"],
                                arguments=tc.get("arguments", {}),
                            )
                            tool_calls.append(tool_call)
                    
                    if tool_calls:
                        logger.debug(f"Parsed {len(tool_calls)} tool calls from clean JSON response")
                        return AssistantMessage(
                            role="assistant",
                            content=None,
                            tool_calls=tool_calls,
                        )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse response as JSON tool calls: {response}")
        
        # Otherwise treat as plain text message
        return AssistantMessage(
            role="assistant",
            content=response,
            tool_calls=None,
        )

    def _call_goose(self, instruction_text: str) -> str:
        """
        Call the goose CLI with the given instruction text.
        
        Args:
            instruction_text: The instruction text to pass to goose
            
        Returns:
            The response text from goose's assistant message
            
        Raises:
            RuntimeError: If goose command fails
        """
        # Create a temporary file for the instructions
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write(instruction_text)
            tmp_file_path = tmp_file.name
        
        try:
            # Build the goose command with JSON output for structured parsing
            cmd = [
                self.goose_command,
                "run",
                "--instructions", tmp_file_path,
                "--quiet",  # Only output the response
                "--output-format", "json",  # Get structured JSON output
                "--max-turns", "1",  # Only allow one turn to prevent goose from looping
            ]
            
            if self.max_turns is not None and self.max_turns > 1:
                # Override the default if user specified a different value
                cmd[-1] = str(self.max_turns)
            
            if self.debug:
                cmd.append("--debug")
            
            logger.debug(f"Executing goose command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Goose command failed with return code {result.returncode}\n"
                error_msg += f"STDERR: {result.stderr}\n"
                error_msg += f"STDOUT: {result.stdout}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Parse the JSON output
            try:
                output = json.loads(result.stdout)
                # Extract the assistant's response from the messages
                messages = output.get("messages", [])
                for msg in reversed(messages):  # Start from the end to get the latest assistant message
                    if msg.get("role") == "assistant":
                        content = msg.get("content", [])
                        if content and len(content) > 0:
                            response = content[0].get("text", "")
                            logger.debug(f"Goose response: {response}")
                            return response
                
                # If no assistant message found, return empty string
                logger.warning("No assistant message found in goose output")
                return ""
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse goose JSON output: {e}\nOutput: {result.stdout}")
                raise RuntimeError(f"Failed to parse goose JSON output: {e}")
            
        finally:
            # Clean up the temporary file
            try:
                Path(tmp_file_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_file_path}: {e}")

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> GooseAgentState:
        """
        Get the initial state of the agent.
        
        Args:
            message_history: The message history of the conversation.
            
        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, "
            "or ToolMessage to Agent."
        )
        
        return GooseAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GooseAgentState
    ) -> tuple[AssistantMessage, GooseAgentState]:
        """
        Generate the next message using the Goose CLI.
        
        Args:
            message: The user message or tool message(s).
            state: The agent state.
            
        Returns:
            A tuple of an assistant message and an updated agent state.
        """
        # Update state with incoming message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        
        # Create instruction text for goose
        instruction_text = self._create_instruction_text(message, state)
        
        # Call goose and get response
        try:
            response_text = self._call_goose(instruction_text)
            assistant_message = self._parse_goose_response(response_text)
        except Exception as e:
            logger.error(f"Error calling goose: {e}")
            # Return an error message to the user
            assistant_message = AssistantMessage(
                role="assistant",
                content=f"I apologize, but I encountered an error processing your request. Please try again.",
                tool_calls=None,
            )
        
        # Update state with assistant message
        state.messages.append(assistant_message)
        
        return assistant_message, state

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[GooseAgentState] = None,
    ) -> None:
        """
        Stop the agent.
        
        Args:
            message: The last message to the agent.
            state: The agent state.
        """
        logger.info("GooseAgent stopped")
        pass
