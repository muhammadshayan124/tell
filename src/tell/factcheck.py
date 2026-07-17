"""Fact-checking: a deliberately separate, unrelated signal from the
deception-cue model in model.py.

Checking whether a specific claim matches retrievable evidence is a solid,
well-understood problem -- unlike inferring truthfulness from behavior, this
one isn't pseudoscience. It's kept as an independent module and an
independent output field on purpose: a statement can be behaviorally
consistent with the deception-cue literature *and* factually correct, or
inconsistent *and* factually wrong, or any other combination -- collapsing
them into one score would hide that these are two different questions.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass


class FactCheckUnavailableError(RuntimeError):
    """Raised when no fact-checking provider is configured (missing SDK/key)."""


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    verdict: str  # "supported" | "contradicted" | "unverifiable"
    explanation: str


class FactCheckProvider(ABC):
    @abstractmethod
    def extract_claims(self, transcript: str) -> list[str]:
        """Pull out specific, checkable factual assertions from free text."""

    @abstractmethod
    def verify_claim(self, claim: str) -> ClaimVerdict:
        """Check one claim against retrievable evidence."""

    def verify_transcript(self, transcript: str) -> list[ClaimVerdict]:
        return [self.verify_claim(claim) for claim in self.extract_claims(transcript)]


_CLAIM_EXTRACTION_PROMPT = """Extract specific, independently checkable factual \
claims from the following statement -- names, dates, locations, quantities, \
events that either did or didn't happen. Skip opinions, feelings, and vague \
statements that can't be verified against a record.

Return ONLY a JSON array of strings, nothing else. If there are no checkable \
claims, return [].

Statement:
{transcript}
"""

_VERIFICATION_PROMPT = """You are checking a single factual claim using your \
web search tool. Search for evidence, then decide: does the evidence support \
this claim, contradict it, or is it unverifiable (no reliable evidence either \
way)?

Claim: {claim}

Reply with ONLY a JSON object: \
{{"verdict": "supported" | "contradicted" | "unverifiable", \
"explanation": "one or two sentences citing what you found"}}
"""


class AnthropicFactCheckProvider(FactCheckProvider):
    """Uses Claude's web_search tool, so verification is grounded in actual
    retrieved pages rather than the model's own (possibly stale or
    hallucinated) knowledge."""

    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise FactCheckUnavailableError(
                "anthropic is not installed. Install it with: pip install 'tell[factcheck]'"
            ) from exc

        import os

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise FactCheckUnavailableError("ANTHROPIC_API_KEY is not set")

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def extract_claims(self, transcript: str) -> list[str]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": _CLAIM_EXTRACTION_PROMPT.format(transcript=transcript)}
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        claims = json.loads(text)
        if not isinstance(claims, list):
            raise ValueError(f"expected a JSON array of claims, got: {text!r}")
        return claims

    def verify_claim(self, claim: str) -> ClaimVerdict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": _VERIFICATION_PROMPT.format(claim=claim)}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        data = json.loads(text[text.index("{") : text.rindex("}") + 1])
        return ClaimVerdict(
            claim=claim, verdict=data["verdict"], explanation=data.get("explanation", "")
        )


def get_factcheck_provider(provider: str = "anthropic", **kwargs) -> FactCheckProvider:
    providers = {"anthropic": AnthropicFactCheckProvider}
    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(providers)}")
    return cls(**kwargs)
