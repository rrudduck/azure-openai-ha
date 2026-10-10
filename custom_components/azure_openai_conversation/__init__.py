"""The Azure OpenAI Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_API_BASE,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_WEB_SEARCH,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
)

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type AzureOpenAIConfigEntry = ConfigEntry[openai.AsyncOpenAI]


def normalize_azure_endpoint(endpoint: str) -> str:
    """Return an Azure OpenAI v1 endpoint with a trailing slash."""
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint.endswith("/openai/v1"):
        endpoint = f"{endpoint}/openai/v1"
    return f"{endpoint}/"


def create_client(
    hass: HomeAssistant, data: Mapping[str, str]
) -> openai.AsyncOpenAI:
    """Create an Azure OpenAI client using Home Assistant's HTTP session."""
    return openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY],
        base_url=normalize_azure_endpoint(data[CONF_API_BASE]),
        http_client=cast(Any, get_async_client(hass)),
    )


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Azure OpenAI Conversation."""
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: AzureOpenAIConfigEntry
) -> bool:
    """Set up Azure OpenAI Conversation from a config entry."""
    client = create_client(hass, entry.data)

    try:
        await client.models.list(timeout=10.0)
    except openai.AuthenticationError as err:
        raise ConfigEntryAuthFailed(err) from err
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: AzureOpenAIConfigEntry
) -> bool:
    """Unload Azure OpenAI Conversation."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_migrate_entry(
    hass: HomeAssistant, entry: AzureOpenAIConfigEntry
) -> bool:
    """Normalize options from earlier integration versions."""
    if entry.version > 2:
        return True

    retired_options = {
        "city",
        "country",
        "reasoning_effort",
        "recommended",
        "region",
        "temperature",
        "timezone",
        "top_p",
        "user_location",
    }
    options = {
        key: value
        for key, value in entry.options.items()
        if key not in retired_options
    }
    if isinstance(options.get(CONF_LLM_HASS_API), str):
        options[CONF_LLM_HASS_API] = [options[CONF_LLM_HASS_API]]
    options.setdefault(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
    options.setdefault(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
    options[CONF_MAX_TOKENS] = int(options[CONF_MAX_TOKENS])
    options.setdefault(CONF_WEB_SEARCH, RECOMMENDED_WEB_SEARCH)
    options.setdefault(
        CONF_WEB_SEARCH_CONTEXT_SIZE, RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE
    )

    hass.config_entries.async_update_entry(entry, options=options, version=3)
    return True


async def async_update_options(
    hass: HomeAssistant, entry: AzureOpenAIConfigEntry
) -> None:
    """Reload the integration when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)
