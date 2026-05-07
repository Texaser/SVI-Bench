"""Unified model interface so each task's evaluate.py speaks one API.

Tasks call `model = get_model(name); model.generate(prompt, video=...)`.
Provider-specific imports happen inside the subclasses, lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        video: Any | None = None,
        images: list[Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response given a prompt and optional visual inputs."""


class OpenAIModel(BaseModel):
    def __init__(self, name: str = "gpt-4o", **client_kwargs: Any) -> None:
        from openai import OpenAI

        self.name = name
        self._client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        *,
        video: Any | None = None,
        images: list[Any] | None = None,
        **kwargs: Any,
    ) -> str:
        # TODO: build messages with image/video content blocks per OpenAI's
        # multimodal schema. Stubbed for now.
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class AnthropicModel(BaseModel):
    def __init__(self, name: str = "claude-opus-4-7", **client_kwargs: Any) -> None:
        # Lazy import — anthropic is only required for tasks that use it.
        from anthropic import Anthropic

        self.name = name
        self._client = Anthropic(**client_kwargs)

    def generate(
        self,
        prompt: str,
        *,
        video: Any | None = None,
        images: list[Any] | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        resp = self._client.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        # Concatenate text blocks.
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


_REGISTRY: dict[str, type[BaseModel]] = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
}


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """Resolve a model name to an instantiated wrapper.

    Names starting with `gpt-` or `o`-prefixed reasoning models route to OpenAI;
    `claude-*` routes to Anthropic. Pass an explicit `provider=` kwarg to override.
    """
    provider = kwargs.pop("provider", None)
    if provider is None:
        if name.startswith(("gpt-", "o1", "o3", "o4")):
            provider = "openai"
        elif name.startswith("claude-"):
            provider = "anthropic"
        else:
            raise ValueError(
                f"cannot infer provider for model {name!r}; pass provider= explicitly"
            )
    cls = _REGISTRY[provider]
    return cls(name=name, **kwargs)
