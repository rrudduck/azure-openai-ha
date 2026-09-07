"""Constants for Azure OpenAI Conversation."""

import logging

DOMAIN = "azure_openai_conversation"
LOGGER = logging.getLogger(__package__)

CONF_API_BASE = "api_base"
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"
CONF_WEB_SEARCH = "web_search"
CONF_WEB_SEARCH_CONTEXT_SIZE = "search_context_size"

RECOMMENDED_CHAT_MODEL = "gpt-4o-mini"
RECOMMENDED_MAX_TOKENS = 1024
RECOMMENDED_WEB_SEARCH = False
RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE = "medium"
