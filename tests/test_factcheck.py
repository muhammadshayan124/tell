import pytest

from tell.factcheck import (
    AnthropicFactCheckProvider,
    ClaimVerdict,
    FactCheckProvider,
    FactCheckUnavailableError,
    get_factcheck_provider,
)


class _FakeFactCheckProvider(FactCheckProvider):
    """A scripted provider, standing in for a real LLM+search backend --
    verify_transcript's own logic (looping claims through verify_claim) is
    exercised for real, only the network-calling methods are faked."""

    def __init__(self, claims: list[str], verdicts: dict[str, ClaimVerdict]):
        self._claims = claims
        self._verdicts = verdicts

    def extract_claims(self, transcript: str) -> list[str]:
        return self._claims

    def verify_claim(self, claim: str) -> ClaimVerdict:
        return self._verdicts[claim]


def test_verify_transcript_checks_every_extracted_claim():
    provider = _FakeFactCheckProvider(
        claims=["The meeting was on Tuesday", "The report was 40 pages"],
        verdicts={
            "The meeting was on Tuesday": ClaimVerdict(
                claim="The meeting was on Tuesday",
                verdict="supported",
                explanation="Calendar confirms.",
            ),
            "The report was 40 pages": ClaimVerdict(
                claim="The report was 40 pages",
                verdict="contradicted",
                explanation="Actual report is 12 pages.",
            ),
        },
    )

    verdicts = provider.verify_transcript("some transcript")
    assert [v.verdict for v in verdicts] == ["supported", "contradicted"]


def test_verify_transcript_returns_empty_list_when_no_claims_extracted():
    provider = _FakeFactCheckProvider(claims=[], verdicts={})
    assert provider.verify_transcript("I feel like today was a good day") == []


def test_anthropic_provider_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(FactCheckUnavailableError, match="ANTHROPIC_API_KEY"):
        AnthropicFactCheckProvider(api_key=None)


def test_get_factcheck_provider_rejects_unknown_provider_name():
    with pytest.raises(ValueError, match="unknown provider"):
        get_factcheck_provider("not-a-real-provider")
