# Azure OpenAI Conversation for Home Assistant

A small Home Assistant custom integration that provides an Assist conversation
agent backed by an Azure OpenAI model deployment.

The integration uses Azure's OpenAI-compatible **Responses API** at
`/openai/v1/responses`. It intentionally sends only model-neutral request options,
so Azure deployment names for GPT-4.1, GPT-5, o-series, and later compatible
models do not depend on fragile name detection.

## Requirements

- Home Assistant 2026.9.1 or newer
- An Azure OpenAI resource and API key
- A model deployment in a region that supports the Responses API

Both Azure endpoint forms are supported:

- `https://<resource>.openai.azure.com`
- `https://<resource>.services.ai.azure.com`

Enter the resource endpoint only. The integration adds `/openai/v1/` when needed.

## Installation

### HACS

Add `https://github.com/rrudduck/azure-openai-ha` as a custom integration
repository in HACS, install **Azure OpenAI Conversation**, and restart Home
Assistant.

### Manual

Copy `custom_components/azure_openai_conversation` into the same path below your
Home Assistant configuration directory, then restart Home Assistant.

## Configuration

1. Open **Settings → Devices & services → Add integration**.
2. Select **Azure OpenAI Conversation**.
3. Enter the Azure resource endpoint and API key.
4. Configure the integration and set **Azure deployment name** to the deployment
   name from Azure AI Foundry.
5. Select the resulting conversation agent in your Assist pipeline.

The options page also controls the instructions, Home Assistant tool access,
maximum output tokens, and optional Azure web search. Web search must be supported
by the selected Azure deployment and region.

## Model compatibility

The integration does not send `temperature`, `top_p`, or a guessed reasoning
configuration. Those values are not portable across current Azure OpenAI models,
especially when the Azure deployment name does not reveal the underlying model.
Reasoning models use their server-side defaults.

Responses are stateless (`store: false`). Encrypted reasoning items are retained
in Home Assistant's chat log and passed back to Azure, which preserves tool-call
context without storing conversations in the Azure service.

## Development

Run the repository checks with Docker:

```sh
make test
```

The checks run Ruff, Home Assistant hassfest, and HACS validation.

## License

[MIT](LICENSE)
