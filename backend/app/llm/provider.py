from collections.abc import Sequence

from openai import OpenAI

from app import config


class LLMConfigurationError(RuntimeError):
    """Raised when the NVIDIA provider is not configured."""


class NVIDIAProvider:
    def __init__(self) -> None:
        if not config.NVIDIA_API_KEY:
            raise LLMConfigurationError("NVIDIA_API_KEY is not configured.")
        if not config.NVIDIA_MODEL:
            raise LLMConfigurationError("NVIDIA_MODEL is not configured.")

        self.model = config.NVIDIA_MODEL
        self.client = OpenAI(
            api_key=config.NVIDIA_API_KEY,
            base_url=config.NVIDIA_BASE_URL,
        )

    def generate_response(self, messages: Sequence[dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
        )
        response = completion.choices[0].message.content
        if not response:
            raise RuntimeError("NVIDIA returned an empty response.")
        return response