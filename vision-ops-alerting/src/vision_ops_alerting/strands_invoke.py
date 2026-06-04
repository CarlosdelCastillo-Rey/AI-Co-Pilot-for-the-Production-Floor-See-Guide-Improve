"""Strands agent helpers — no stdout streaming, clean text extraction."""

from __future__ import annotations

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.models.model import Model


def create_agent(
    *,
    model: Model,
    system_prompt: str,
    tools: list | None = None,
) -> Agent:
    """callback_handler=None disables PrintingCallbackHandler (avoids polluting uvicorn logs)."""
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
        callback_handler=None,
    )


def agent_result_text(result: AgentResult | str) -> str:
    if isinstance(result, AgentResult):
        return str(result).strip()
    return str(result).strip()


def invoke_agent(agent: Agent, prompt: str) -> str:
    return agent_result_text(agent(prompt))
