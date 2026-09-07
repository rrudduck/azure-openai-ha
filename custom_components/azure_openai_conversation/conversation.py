"""Conversation support for Azure OpenAI."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterable
import json
from typing import Any, Literal

import openai
from openai._streaming import AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearch,
    ResponseFunctionWebSearchParam,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ToolParam,
    WebSearchToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from probatio import to_openapi

from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.json import json_dumps

from . import AzureOpenAIConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
)

MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: AzureOpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Azure OpenAI conversation entity."""
    async_add_entities([AzureOpenAIConversationEntity(config_entry)])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> FunctionToolParam:
    """Convert a Home Assistant tool schema to an OpenAI function tool."""
    unsupported_keys = {"oneOf", "anyOf", "allOf", "enum", "not"}
    schema = to_openapi(tool.parameters, custom_serializer=custom_serializer)
    if unsupported_keys.intersection(schema):
        schema = {key: value for key, value in schema.items() if key not in unsupported_keys}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})

    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=schema,
        description=tool.description,
        strict=False,
    )


def _convert_content_to_param(
    chat_content: Iterable[conversation.Content],
) -> ResponseInputParam:
    """Convert Home Assistant chat content to Responses API input."""
    messages: ResponseInputParam = []
    web_search_calls: dict[str, ResponseFunctionWebSearchParam] = {}

    for content in chat_content:
        if isinstance(content, conversation.ToolResultContent):
            if (
                content.tool_name == "web_search_call"
                and content.tool_call_id in web_search_calls
            ):
                web_search_call = web_search_calls.pop(content.tool_call_id)
                web_search_call["status"] = content.tool_result.get(  # type: ignore[typeddict-item]
                    "status", "completed"
                )
                messages.append(web_search_call)
            else:
                messages.append(
                    FunctionCallOutput(
                        type="function_call_output",
                        call_id=content.tool_call_id,
                        output=json_dumps(content.tool_result),
                    )
                )
            continue

        if content.content:
            role: Literal["user", "assistant", "system", "developer"] = content.role
            if role == "system":
                role = "developer"
            messages.append(
                EasyInputMessageParam(
                    type="message", role=role, content=content.content
                )
            )

        if not isinstance(content, conversation.AssistantContent):
            continue

        for tool_call in content.tool_calls or ():
            if (
                tool_call.external
                and tool_call.tool_name == "web_search_call"
                and "action" in tool_call.tool_args
            ):
                web_search_calls[tool_call.id] = ResponseFunctionWebSearchParam(
                    type="web_search_call",
                    id=tool_call.id,
                    action=tool_call.tool_args["action"],
                    status="completed",
                )
            else:
                messages.append(
                    ResponseFunctionToolCallParam(
                        type="function_call",
                        name=tool_call.tool_name,
                        arguments=json_dumps(tool_call.tool_args),
                        call_id=tool_call.id,
                    )
                )

        if isinstance(content.native, ResponseReasoningItem):
            messages.append(
                ResponseReasoningItemParam(
                    type="reasoning",
                    id=content.native.id,
                    summary=[],
                    encrypted_content=content.native.encrypted_content,
                )
            )

    return messages


def _trace_usage(
    chat_log: conversation.ChatLog,
    event: ResponseCompletedEvent | ResponseIncompleteEvent | ResponseFailedEvent,
) -> None:
    """Add token usage to the Home Assistant conversation trace."""
    if event.response.usage is not None:
        chat_log.async_trace(
            {
                "stats": {
                    "input_tokens": event.response.usage.input_tokens,
                    "output_tokens": event.response.usage.output_tokens,
                }
            }
        )


async def _transform_stream(
    chat_log: conversation.ChatLog,
    stream: AsyncStream[ResponseStreamEvent],
) -> AsyncGenerator[
    conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
]:
    """Transform a Responses API stream into Home Assistant chat content."""
    last_role: Literal["assistant", "tool_result"] | None = None

    async for event in stream:
        LOGGER.debug("Received event: %s", event)

        if isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                yield {"role": "assistant"}
                last_role = "assistant"
                current_tool_call = event.item
            elif isinstance(
                event.item, (ResponseOutputMessage, ResponseReasoningItem)
            ) and last_role != "assistant":
                yield {"role": "assistant"}
                last_role = "assistant"
        elif isinstance(event, ResponseOutputItemDoneEvent):
            if isinstance(event.item, ResponseReasoningItem):
                yield {
                    "native": ResponseReasoningItem(
                        type="reasoning",
                        id=event.item.id,
                        summary=[],
                        encrypted_content=event.item.encrypted_content,
                    )
                }
            elif isinstance(event.item, ResponseFunctionWebSearch):
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=event.item.id,
                            tool_name="web_search_call",
                            tool_args={
                                "action": event.item.action.to_dict()
                                if event.item.action
                                else None
                            },
                            external=True,
                        )
                    ]
                }
                yield {
                    "role": "tool_result",
                    "tool_call_id": event.item.id,
                    "tool_name": "web_search_call",
                    "tool_result": {"status": event.item.status},
                }
                last_role = "tool_result"
        elif isinstance(event, ResponseTextDeltaEvent):
            yield {"content": event.delta}
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            current_tool_call.status = "completed"
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call.call_id,
                        tool_name=current_tool_call.name,
                        tool_args=json.loads(current_tool_call.arguments),
                    )
                ]
            }
        elif isinstance(event, ResponseCompletedEvent):
            _trace_usage(chat_log, event)
        elif isinstance(event, ResponseIncompleteEvent):
            _trace_usage(chat_log, event)
            reason = (
                event.response.incomplete_details.reason
                if event.response.incomplete_details
                else "unknown reason"
            )
            raise HomeAssistantError(f"Azure OpenAI response incomplete: {reason}")
        elif isinstance(event, ResponseFailedEvent):
            _trace_usage(chat_log, event)
            reason = (
                event.response.error.message
                if event.response.error is not None
                else "unknown reason"
            )
            raise HomeAssistantError(f"Azure OpenAI response failed: {reason}")
        elif isinstance(event, ResponseErrorEvent):
            raise HomeAssistantError(f"Azure OpenAI response error: {event.message}")


class AzureOpenAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Azure OpenAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: AzureOpenAIConfigEntry) -> None:
        """Initialize the conversation agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Microsoft Azure",
            model="Azure OpenAI",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return the supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Register this entity as a conversation agent."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister this conversation agent."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process user input with Azure OpenAI."""
        options = self.entry.options
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_handle_chat_log(self, chat_log: conversation.ChatLog) -> None:
        """Generate and stream a response for a Home Assistant chat log."""
        options = self.entry.options
        messages = _convert_content_to_param(chat_log.content)
        tools: list[ToolParam] = []

        if chat_log.llm_api:
            tools.extend(
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            )
        if options.get(CONF_WEB_SEARCH):
            tools.append(
                WebSearchToolParam(
                    type="web_search",
                    search_context_size=options.get(
                        CONF_WEB_SEARCH_CONTEXT_SIZE,
                        RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
                    ),
                )
            )

        model_args: dict[str, Any] = {
            "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "input": messages,
            "max_output_tokens": options.get(
                CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
            ),
            "user": chat_log.conversation_id,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            model_args["tools"] = tools

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                stream = await self.entry.runtime_data.responses.create(**model_args)
                content_stream = chat_log.async_add_delta_content_stream(
                    self.entity_id, _transform_stream(chat_log, stream)
                )
                messages.extend(
                    _convert_content_to_param(
                        [content async for content in content_stream]
                    )
                )
            except openai.AuthenticationError as err:
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError("Azure OpenAI authentication failed") from err
            except openai.RateLimitError as err:
                raise HomeAssistantError(
                    "Azure OpenAI rate limit or quota exceeded"
                ) from err
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to Azure OpenAI: %s", err)
                raise HomeAssistantError("Error talking to Azure OpenAI") from err

            if not chat_log.unresponded_tool_results:
                break
